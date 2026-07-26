"""adsb-train: train the ADCC model with whole sessions held out for eval.

Splits are by session, never random: the newest sessions (or an explicit
list) are held out, so the reported accuracy is always cross-session —
same transponders, different flight geometry. Classes are the ICAOs with
enough messages on *both* sides of the split, so every class is actually
measured under a changed channel.

Caveat: consecutive 30-minute sessions can split a single overflight, so
train and held-out examples of one aircraft may still share near-identical
geometry. Held-out numbers here are optimistic; adsb-eval (P6) is where
ablations and the channel-only baseline pin down what was learned.

Each run writes a directory under paths.models: best.pt (highest held-out
balanced accuracy), last.pt, and a run.yaml sidecar with the arguments,
splits, classes, and per-epoch metrics.
"""

import argparse
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from tqdm import tqdm

from adsb_fingerprint import config, dataset
from adsb_fingerprint.model import (
    ADCC,
    iq_channels,
    pick_device,
)


def split_sessions(all_sessions, test_sessions, test_fraction):
    """Return (train_sessions, test_sessions), holding out the newest by default.

    Session names are UTC timestamps, so lexicographic order is chronological.
    """
    ordered = sorted(all_sessions)
    if test_sessions:
        missing = sorted(set(test_sessions) - set(ordered))
        if missing:
            raise SystemExit(f"unknown test sessions: {', '.join(missing)}")
        held_out = set(test_sessions)
    else:
        n_test = max(1, round(len(ordered) * test_fraction))
        if n_test >= len(ordered):
            raise SystemExit("the split leaves no training sessions")
        held_out = set(ordered[-n_test:])
    return (
        [session for session in ordered if session not in held_out],
        [session for session in ordered if session in held_out],
    )


def select_classes(icaos, train_mask, test_mask, min_train, min_test):
    """ICAOs with enough examples on both sides of the session split."""
    train_counts = Counter(icaos[train_mask].tolist())
    test_counts = Counter(icaos[test_mask].tolist())
    return sorted(
        icao
        for icao in train_counts
        if train_counts[icao] >= min_train and test_counts.get(icao, 0) >= min_test
    )


def cap_per_class_day(train_idx, icaos, days, cap, rng):
    """Randomly cap training examples per (ICAO, local day).

    Spreads each class's training data across the days it was heard, so no
    single day's channel and oscillator state dominates a class prototype.
    """
    capped = []
    for icao in np.unique(icaos):
        mask = icaos == icao
        mine = train_idx[mask]
        mine_days = days[mask]
        for day in np.unique(mine_days):
            rows = mine[mine_days == day]
            if len(rows) > cap:
                rows = rng.choice(rows, cap, replace=False)
            capped.append(rows)
    return np.sort(np.concatenate(capped))


def evaluate(model, x, y, batch_size, device, n_classes):
    """Held-out overall accuracy, balanced accuracy, and per-class recall."""
    model.eval()
    preds = []
    with torch.no_grad():
        for lo in range(0, len(x), batch_size):
            logits = model(x[lo : lo + batch_size].to(device))
            preds.append(logits.argmax(dim=1).cpu())
    model.train()
    preds = torch.cat(preds).numpy()
    truth = y.numpy()
    recalls = np.array(
        [(preds[truth == c] == c).mean() for c in range(n_classes)],
        dtype=float,
    )
    return float((preds == truth).mean()), float(recalls.mean()), recalls


def main():
    parser = argparse.ArgumentParser(
        description="Train the ADCC fingerprinting model with whole sessions held out.",
    )
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Adam learning rate.")
    parser.add_argument("--seed", type=int, default=0, help="Seed for init and shuffling.")
    parser.add_argument(
        "--variant",
        choices=("whole", "icao_masked", "preamble"),
        default="whole",
        help="Ablation view of the IQ (default: whole message).",
    )
    parser.add_argument(
        "--test-sessions",
        nargs="+",
        default=None,
        help="Sessions to hold out (default: the newest --test-fraction of them).",
    )
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=0.2,
        help="Fraction of sessions (newest) to hold out when --test-sessions is not given.",
    )
    parser.add_argument(
        "--match-run",
        default=None,
        help="Path to a previous run directory: reuse its exact train/test "
        "sessions, ignoring any newer ones, so ablation runs stay comparable "
        "(overrides --test-sessions/--test-fraction).",
    )
    parser.add_argument("--min-train", type=int, default=50, help="Min training messages per ICAO.")
    parser.add_argument("--min-test", type=int, default=10, help="Min held-out messages per ICAO.")
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=0,
        help="Cap on training messages per ICAO, randomly subsampled (0 = no cap).",
    )
    parser.add_argument(
        "--max-per-class-day",
        type=int,
        default=0,
        help="Cap on training messages per ICAO per local day, randomly "
        "subsampled (0 = no cap): spreads each class's training data across "
        "the days it was heard instead of letting one busy day dominate.",
    )
    parser.add_argument("--device", default=None, help="Torch device (default: mps/cuda/cpu auto).")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    data = dataset.load_examples()
    if len(data["icao"]) == 0:
        raise SystemExit("no examples — capture and index some messages first")

    if args.match_run:
        matched = yaml.safe_load((Path(args.match_run) / "run.yaml").read_text())
        universe = sorted(
            set(matched["sessions"]["train"]) | set(matched["sessions"]["test"])
        )
        missing = sorted(set(universe) - set(data["session"].tolist()))
        if missing:
            raise SystemExit(f"matched run's sessions have no data: {', '.join(missing)}")
        in_universe = np.isin(data["session"], universe)
        for key in ("iq", "icao", "session", "captured_at", "rssi_db"):
            data[key] = data[key][in_universe]
        args.test_sessions = matched["sessions"]["test"]

    iq = dataset.apply_variant(data["iq"], args.variant)

    train_sessions, test_sessions = split_sessions(
        set(data["session"].tolist()),
        args.test_sessions,
        args.test_fraction,
    )
    train_mask, test_mask = dataset.session_split(data["session"], test_sessions)
    classes = select_classes(
        data["icao"],
        train_mask,
        test_mask,
        args.min_train,
        args.min_test,
    )
    if not classes:
        raise SystemExit(
            "no ICAO has enough messages on both sides of the split — "
            "collect longer or lower --min-train/--min-test"
        )

    keep = np.isin(data["icao"], classes)
    train_idx = np.where(train_mask & keep)[0]
    test_idx = np.where(test_mask & keep)[0]
    if args.max_per_class_day > 0:
        days = np.array(
            [t.astimezone().date().isoformat() for t in data["captured_at"][train_idx]],
        )
        train_idx = cap_per_class_day(
            train_idx,
            data["icao"][train_idx],
            days,
            args.max_per_class_day,
            rng,
        )
    if args.max_per_class > 0:
        capped = []
        for icao in classes:
            mine = train_idx[data["icao"][train_idx] == icao]
            if len(mine) > args.max_per_class:
                mine = rng.choice(mine, args.max_per_class, replace=False)
            capped.append(mine)
        train_idx = np.sort(np.concatenate(capped))

    x_train = torch.from_numpy(iq_channels(iq[train_idx]))
    y_train = torch.from_numpy(np.searchsorted(classes, data["icao"][train_idx]))
    x_test = torch.from_numpy(iq_channels(iq[test_idx]))
    y_test = torch.from_numpy(np.searchsorted(classes, data["icao"][test_idx]))

    device = torch.device(args.device) if args.device else pick_device()
    model_kwargs = {
        "n_classes": len(classes),
        "in_channels": 2,
        "channels": 48,
        "skip_channels": 48,
        "kernel_size": 4,
        "dilations": [2, 4, 8, 16, 32, 64],
    }
    model = ADCC(**model_kwargs).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    n_params = sum(p.numel() for p in model.parameters())

    started = datetime.now(timezone.utc)
    run_dir = config.MODEL_DIR / f"{started.strftime('%Y%m%dT%H%M%SZ')}-{args.variant}"
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"variant  : {args.variant}")
    print(f"sessions : {len(train_sessions)} train, {len(test_sessions)} held out ({', '.join(test_sessions)})")
    print(f"classes  : {len(classes)} ICAOs (>= {args.min_train} train / >= {args.min_test} held-out msgs)")
    print(f"examples : {len(train_idx)} train, {len(test_idx)} held out")
    print(f"model    : {n_params:,} params on {device.type}")
    print(f"run dir  : {run_dir}")

    history = []
    best = {"balanced_accuracy": -1.0, "epoch": 0}

    def write_meta(outcome=None):
        meta = {
            "tool": "adsb-train",
            "started_at": started.isoformat(),
            "args": vars(args),
            "sessions": {
                "train": train_sessions,
                "test": test_sessions,
            },
            "classes": classes,
            "examples": {
                "train": int(len(train_idx)),
                "test": int(len(test_idx)),
            },
            "parameters": n_params,
            "device": device.type,
            "epochs": history,
        }
        if outcome is not None:
            meta["outcome"] = outcome
        (run_dir / "run.yaml").write_text(yaml.safe_dump(meta, sort_keys=False))

    def checkpoint(path, epoch, accuracy, balanced):
        torch.save(
            {
                "model": model_kwargs,
                "state_dict": model.state_dict(),
                "classes": classes,
                "variant": args.variant,
                "window": int(data["window"]),
                "epoch": epoch,
                "accuracy": accuracy,
                "balanced_accuracy": balanced,
                "train_sessions": train_sessions,
                "test_sessions": test_sessions,
            },
            path,
        )

    write_meta()
    for epoch in range(1, args.epochs + 1):
        tick = time.perf_counter()
        perm = rng.permutation(len(train_idx))
        total_loss = 0.0
        for lo in tqdm(
            range(0, len(perm), args.batch_size),
            desc=f"epoch {epoch}",
            leave=False,
        ):
            batch = perm[lo : lo + args.batch_size]
            optimizer.zero_grad()
            loss = F.cross_entropy(
                model(x_train[batch].to(device)),
                y_train[batch].to(device),
            )
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch)
        train_loss = total_loss / len(perm)

        accuracy, balanced, _ = evaluate(
            model,
            x_test,
            y_test,
            args.batch_size,
            device,
            len(classes),
        )
        seconds = time.perf_counter() - tick
        marker = ""
        if balanced > best["balanced_accuracy"]:
            best = {"balanced_accuracy": balanced, "epoch": epoch, "accuracy": accuracy}
            checkpoint(run_dir / "best.pt", epoch, accuracy, balanced)
            marker = "  *"
        checkpoint(run_dir / "last.pt", epoch, accuracy, balanced)
        history.append(
            {
                "epoch": epoch,
                "train_loss": round(train_loss, 4),
                "accuracy": round(accuracy, 4),
                "balanced_accuracy": round(balanced, 4),
                "seconds": round(seconds, 1),
            }
        )
        write_meta()
        print(
            f"epoch {epoch:3d}/{args.epochs}  loss {train_loss:.4f}  "
            f"held-out acc {accuracy:.3f}  balanced {balanced:.3f}  ({seconds:.1f}s){marker}"
        )

    write_meta(
        outcome={
            "ended_at": datetime.now(timezone.utc).isoformat(),
            "best_epoch": best["epoch"],
            "best_accuracy": round(best.get("accuracy", 0.0), 4),
            "best_balanced_accuracy": round(best["balanced_accuracy"], 4),
        },
    )
    print(
        f"\nbest: epoch {best['epoch']}  "
        f"held-out acc {best.get('accuracy', 0.0):.3f}  balanced {best['balanced_accuracy']:.3f}"
    )
    print(f"checkpoints in {run_dir}")


if __name__ == "__main__":
    main()
