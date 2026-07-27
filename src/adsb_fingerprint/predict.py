"""adsb-predict: live identification — score incoming messages against enrolled signatures.

Follows the messages index as adsb-collect writes it (rows land ~1 s after
reception), embeds each new message with a pinned training run's model, and
scores it against per-aircraft signature centroids:

  signature   recency-weighted mean of an aircraft's normalized embeddings
              (exponential decay, --half-life-days), so stale eras age out
              on their own after hardware or station changes
  enrollment  any aircraft with >= --min-enroll messages in the store —
              data-driven, not limited to the model's training classes
  prediction  nearest signature by cosine, written per message to the
              predictions table with the similarity, the margin over the
              runner-up, and a type vote (registry models of the
              --neighbors nearest signatures, similarity-weighted)

A message is scored BEFORE its own embedding joins the store, so it never
matches itself. The startup pass embeds --lookback-days of history to seed
the signatures, then backfills predictions for --backfill-hours so the
/live page has rows immediately (backfilled messages are the one place a
message sits inside its own signature — bootstrap only, ~1 part in
--min-enroll optimistic). The checkpoint's own ablation variant and window
are applied, so live inference sees exactly what training saw.

predictions and signatures are derived tables: drop them and re-run with a
longer --backfill-hours to rebuild.
"""

import argparse
import time
from collections import defaultdict
from datetime import (
    datetime,
    timezone,
)
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from adsb_fingerprint import (
    config,
    dataset,
    db,
)
from adsb_fingerprint.evaluate import load_run
from adsb_fingerprint.model import (
    iq_channels,
    pick_device,
)
from adsb_fingerprint.verify import embed_features

HISTORY_SQL = """
    select
        id,
        capture_file,
        sample_offset,
        icao,
        captured_at
    from messages
    where crc_ok
    and icao is not null
    and captured_at > now() - %(days)s * interval '1 day'
    order by id
"""

NEW_MESSAGES_SQL = """
    select
        id,
        capture_file,
        sample_offset,
        icao,
        captured_at
    from messages
    where crc_ok
    and icao is not null
    and id > %(after_id)s
    order by id
"""

REGISTRY_SQL = """
    select
        icao,
        model,
        type
    from aircraft
    where icao = any(%(icaos)s)
"""

INSERT_SQL = """
    insert into predictions (
        message_id,
        model_run,
        predicted_icao,
        similarity,
        margin,
        predicted_type
    )
    values (
        %(message_id)s,
        %(model_run)s,
        %(predicted_icao)s,
        %(similarity)s,
        %(margin)s,
        %(predicted_type)s
    )
    on conflict (message_id) do nothing
"""

UPSERT_SIGNATURE_SQL = """
    insert into signatures (
        icao,
        model_run,
        weight,
        messages,
        updated_at
    )
    values (
        %(icao)s,
        %(model_run)s,
        %(weight)s,
        %(messages)s,
        %(updated_at)s
    )
    on conflict (icao) do update set
        model_run = excluded.model_run,
        weight = excluded.weight,
        messages = excluded.messages,
        updated_at = excluded.updated_at
"""


def newest_loadable_run():
    """The newest run dir whose best.pt loads with the current code."""
    for run_dir in sorted(
        (d for d in config.MODEL_DIR.iterdir() if d.is_dir()),
        reverse=True,
    ):
        try:
            load_run(run_dir, "best")
            return run_dir
        except Exception as error:
            print(f"skipping {run_dir.name}: {error}")
    raise SystemExit("no loadable training run under paths.models")


def read_snippets(rows, window):
    """(n, window) complex64 IQ, sliced and normalized like dataset.load_examples."""
    out = np.zeros((len(rows), window), np.complex64)
    by_file = defaultdict(list)
    for i, row in enumerate(rows):
        by_file[row["capture_file"]].append(i)
    for capture_file, indices in by_file.items():
        with open(config.CAPTURE_DIR / capture_file, "rb") as f:
            for i in indices:
                f.seek(rows[i]["sample_offset"] * 8)
                snippet = np.fromfile(f, dtype=np.complex64, count=window)
                peak = np.abs(snippet).max() if len(snippet) else 0.0
                if peak > 0:
                    out[i, : len(snippet)] = snippet / peak
    return out


class SignatureStore:
    """Per-aircraft exponentially-decayed embedding sums.

    Sums are valued as of t_ref; advancing t_ref decays every aircraft by
    the elapsed time, so a signature is always the recency-weighted mean
    of everything heard, with old eras fading on the half-life.
    """

    def __init__(self, dim, half_life_days, min_enroll):
        self.dim = dim
        self.half_life_s = half_life_days * 86400
        self.min_enroll = min_enroll
        self.sums = {}
        self.weights = {}
        self.counts = {}
        self.t_ref = None

    def add_batch(self, embeddings, icaos, stamps):
        newest = max(stamps)
        if self.t_ref is None:
            self.t_ref = newest
        elif newest > self.t_ref:
            factor = 0.5 ** ((newest - self.t_ref).total_seconds() / self.half_life_s)
            for icao in self.sums:
                self.sums[icao] *= factor
                self.weights[icao] *= factor
            self.t_ref = newest
        for emb, icao, stamp in zip(embeddings, icaos, stamps):
            weight = 0.5 ** ((self.t_ref - stamp).total_seconds() / self.half_life_s)
            if icao not in self.sums:
                self.sums[icao] = np.zeros(self.dim)
                self.weights[icao] = 0.0
                self.counts[icao] = 0
            self.sums[icao] += weight * emb
            self.weights[icao] += weight
            self.counts[icao] += 1

    def enrolled(self):
        """(icaos list, (n, dim) normalized centroid matrix), roster order."""
        icaos = sorted(
            icao
            for icao, count in self.counts.items()
            if count >= self.min_enroll
        )
        if not icaos:
            return [], np.zeros((0, self.dim))
        matrix = np.stack([self.sums[icao] for icao in icaos])
        return icaos, matrix / np.maximum(
            np.linalg.norm(matrix, axis=1, keepdims=True),
            1e-12,
        )


def type_labels(conn, icaos, cache):
    """Registry model-or-type label per ICAO, cached across ticks."""
    missing = [icao for icao in icaos if icao not in cache]
    if missing:
        for row in conn.execute(REGISTRY_SQL, {"icaos": missing}).fetchall():
            cache[row["icao"]] = row["model"] or row["type"]
        for icao in missing:
            cache.setdefault(icao, None)
    return cache


def score(embeddings, roster, centroids, labels, neighbors):
    """Prediction dicts (sans message ids) for a batch of embeddings."""
    sims = embeddings @ centroids.T
    k = min(neighbors, len(roster))
    out = []
    for row in sims:
        top = np.argsort(-row)[:k]
        votes = defaultdict(float)
        for i in top:
            votes[labels.get(roster[i]) or "?"] += float(row[i])
        out.append(
            {
                "predicted_icao": roster[int(top[0])],
                "similarity": float(row[top[0]]),
                "margin": float(row[top[0]] - row[top[1]]) if k > 1 else None,
                "predicted_type": max(votes, key=votes.get),
            }
        )
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Score incoming messages against enrolled aircraft signatures, continuously.",
    )
    parser.add_argument(
        "--run",
        default=None,
        help="Training run directory or bare name (default: newest loadable run).",
    )
    parser.add_argument("--interval", type=float, default=5.0, help="Seconds between polls.")
    parser.add_argument(
        "--half-life-days",
        type=float,
        default=2.0,
        help="Signature recency half-life (2 d matched pooled enrollment and "
        "self-heals across hardware-era boundaries).",
    )
    parser.add_argument("--min-enroll", type=int, default=50, help="Messages before an aircraft enrolls.")
    parser.add_argument("--lookback-days", type=float, default=7.0, help="History window that seeds signatures.")
    parser.add_argument(
        "--backfill-hours",
        type=float,
        default=2.0,
        help="Also write predictions for this recent slice at startup (0 = none).",
    )
    parser.add_argument("--neighbors", type=int, default=5, help="Signatures voting on the type label.")
    parser.add_argument("--batch-size", type=int, default=512, help="Embedding batch size.")
    parser.add_argument("--device", default=None, help="Torch device (default: mps/cuda/cpu auto).")
    args = parser.parse_args()

    if args.run:
        run_dir = Path(args.run).expanduser()
        if not run_dir.is_dir():
            run_dir = config.MODEL_DIR / args.run
        ckpt, net = load_run(run_dir, "best")
    else:
        run_dir = newest_loadable_run()
        ckpt, net = load_run(run_dir, "best")
    window = ckpt["window"]
    device = torch.device(args.device) if args.device else pick_device()

    store = SignatureStore(
        dim=net.classify.in_features,
        half_life_days=args.half_life_days,
        min_enroll=args.min_enroll,
    )
    labels = {}
    model_run = run_dir.name
    print(f"model    : {model_run} ({ckpt['variant']}, window {window})")

    with db.connect() as conn:
        history = conn.execute(HISTORY_SQL, {"days": args.lookback_days}).fetchall()
        print(f"history  : {len(history)} messages in the last {args.lookback_days:g} days")
        if history:
            last_id = history[-1]["id"]
        else:
            newest = conn.execute("select coalesce(max(id), 0) as last_id from messages").fetchone()
            last_id = newest["last_id"]
        for lo in tqdm(
            range(0, len(history), 4096),
            desc="seed signatures",
            leave=False,
        ):
            batch = history[lo : lo + 4096]
            iq = dataset.apply_variant(read_snippets(batch, window), ckpt["variant"])
            emb = embed_features(net, iq_channels(iq), device, args.batch_size)
            store.add_batch(
                emb,
                [row["icao"] for row in batch],
                [row["captured_at"] for row in batch],
            )

        roster, centroids = store.enrolled()
        labels = type_labels(conn, roster, labels)
        now = datetime.now(timezone.utc)
        for icao in roster:
            conn.execute(
                UPSERT_SIGNATURE_SQL,
                {
                    "icao": icao,
                    "model_run": model_run,
                    "weight": round(store.weights[icao], 2),
                    "messages": store.counts[icao],
                    "updated_at": now,
                },
            )
        conn.commit()
        print(f"enrolled : {len(roster)} aircraft (>= {args.min_enroll} msgs, half-life {args.half_life_days:g} d)")

        if args.backfill_hours > 0 and roster:
            cutoff = datetime.now(timezone.utc).timestamp() - args.backfill_hours * 3600
            recent = [
                row
                for row in history
                if row["captured_at"].timestamp() >= cutoff
            ]
            for lo in tqdm(
                range(0, len(recent), 4096),
                desc="backfill",
                leave=False,
            ):
                batch = recent[lo : lo + 4096]
                iq = dataset.apply_variant(read_snippets(batch, window), ckpt["variant"])
                emb = embed_features(net, iq_channels(iq), device, args.batch_size)
                conn.cursor().executemany(
                    INSERT_SQL,
                    [
                        {
                            "message_id": row["id"],
                            "model_run": model_run,
                            **scored,
                        }
                        for row, scored in zip(
                            batch,
                            score(emb, roster, centroids, labels, args.neighbors),
                        )
                    ],
                )
            conn.commit()
            print(f"backfill : {len(recent)} messages over the last {args.backfill_hours:g} h")

        print("following the index — Ctrl+C to stop")
        while True:
            fresh = conn.execute(NEW_MESSAGES_SQL, {"after_id": last_id}).fetchall()
            if not fresh:
                conn.commit()
                time.sleep(args.interval)
                continue
            last_id = fresh[-1]["id"]
            iq = dataset.apply_variant(read_snippets(fresh, window), ckpt["variant"])
            emb = embed_features(net, iq_channels(iq), device, args.batch_size)
            roster, centroids = store.enrolled()
            if roster:
                labels = type_labels(
                    conn,
                    roster + [row["icao"] for row in fresh],
                    labels,
                )
                conn.cursor().executemany(
                    INSERT_SQL,
                    [
                        {
                            "message_id": row["id"],
                            "model_run": model_run,
                            **scored,
                        }
                        for row, scored in zip(
                            fresh,
                            score(emb, roster, centroids, labels, args.neighbors),
                        )
                    ],
                )
            store.add_batch(
                emb,
                [row["icao"] for row in fresh],
                [row["captured_at"] for row in fresh],
            )
            now = datetime.now(timezone.utc)
            for icao in sorted({row["icao"] for row in fresh} & set(store.counts)):
                if store.counts[icao] >= args.min_enroll:
                    conn.execute(
                        UPSERT_SIGNATURE_SQL,
                        {
                            "icao": icao,
                            "model_run": model_run,
                            "weight": round(store.weights[icao], 2),
                            "messages": store.counts[icao],
                            "updated_at": now,
                        },
                    )
            conn.commit()
            stamp = datetime.now().strftime("%H:%M:%S")
            print(f"{stamp}  scored {len(fresh):4d} msgs · {len(roster)} enrolled")
            time.sleep(args.interval)


if __name__ == "__main__":
    main()
