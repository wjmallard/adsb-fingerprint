"""adsb-verify: open-set metrics — claimed-ID verification and stranger detection.

Rebuilds a run's exact split, embeds every message under the checkpoint's
own ablation variant, and enrolls each class as the normalized mean of its
train-split embeddings. Held-out-session messages are then scored by cosine
similarity to the enrolled prototypes:

  verify (tier 1)    genuine trial = a held-out message against its own
                     class's prototype; impostor trials = the same message
                     against every other prototype. Reports EER, the
                     threshold-free 1-vs-1 anti-spoof number.
  stranger (tier 3)  score = best similarity to any enrolled prototype.
                     Knowns are held-out messages of enrolled classes;
                     strangers are the run's --exclude ICAOs (never
                     trained on), scored in the same held-out sessions so
                     both sides face the same days and channel. Reports
                     AUROC: 0.5 = strangers look enrolled, 1.0 = cleanly
                     gateable.

Strangers default to the run's recorded --exclude list (from run.yaml);
--strangers overrides. Runs trained without excludes report only the
verify numbers. Given several runs over the same split, ends with a
comparison table.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
)
from tqdm import tqdm

from adsb_fingerprint import config, dataset
from adsb_fingerprint.evaluate import load_run, split_for
from adsb_fingerprint.model import (
    iq_channels,
    pick_device,
)


def embed_features(net, x, device, batch_size):
    """L2-normalized penultimate-layer embeddings for (n, 2, t) input."""
    net = net.to(device)
    outs = []
    with torch.no_grad():
        for lo in tqdm(
            range(0, len(x), batch_size),
            desc="embed",
            leave=False,
        ):
            batch = torch.from_numpy(x[lo : lo + batch_size]).to(device)
            outs.append(net.features(batch).cpu().numpy())
    emb = np.concatenate(outs)
    return emb / np.maximum(
        np.linalg.norm(emb, axis=1, keepdims=True),
        1e-12,
    )


def enroll(embeddings, class_idx, n_classes):
    """(n_classes, d) prototypes: normalized mean of each class's embeddings."""
    protos = np.zeros((n_classes, embeddings.shape[1]))
    np.add.at(protos, class_idx, embeddings)
    return protos / np.maximum(
        np.linalg.norm(protos, axis=1, keepdims=True),
        1e-12,
    )


def eer(genuine, impostor):
    """Equal error rate for two score arrays (higher = more genuine)."""
    labels = np.concatenate(
        [
            np.ones(len(genuine)),
            np.zeros(len(impostor)),
        ]
    )
    fpr, tpr, _ = roc_curve(labels, np.concatenate([genuine, impostor]))
    i = int(np.argmin(np.abs(fpr - (1 - tpr))))
    return float((fpr[i] + 1 - tpr[i]) / 2)


def strangers_for(run_dir, override):
    """The stranger ICAO list: --strangers if given, else the run's --exclude."""
    if override:
        return sorted(override)
    run_yaml = run_dir / "run.yaml"
    if not run_yaml.exists():
        return []
    recorded = yaml.safe_load(run_yaml.read_text())["args"].get("exclude")
    return sorted(recorded) if recorded else []


def main():
    parser = argparse.ArgumentParser(
        description="Open-set evaluation: verification EER and stranger-detection AUROC.",
    )
    parser.add_argument("runs", nargs="+", help="Run directories (from adsb-train).")
    parser.add_argument(
        "--checkpoint",
        choices=("best", "last"),
        default="best",
        help="Which checkpoint to load from each run.",
    )
    parser.add_argument(
        "--strangers",
        nargs="+",
        default=None,
        metavar="ICAO",
        help="ICAOs to score as strangers (default: the run's --exclude list).",
    )
    parser.add_argument("--batch-size", type=int, default=512, help="Embedding batch size.")
    args = parser.parse_args()

    run_dirs = []
    for run in args.runs:
        run_dir = Path(run).expanduser()
        if not run_dir.is_dir():
            run_dir = config.MODEL_DIR / run
        run_dirs.append(run_dir)
    loaded = [load_run(run_dir, args.checkpoint) for run_dir in run_dirs]
    device = pick_device()
    data = dataset.load_examples()

    results = []
    for run_dir, (ckpt, net) in zip(run_dirs, loaded):
        classes = ckpt["classes"]
        train_idx, test_idx = split_for(ckpt, data)
        strangers = strangers_for(run_dir, args.strangers)
        stranger_idx = np.where(
            np.isin(data["icao"], strangers)
            & np.isin(data["session"], ckpt["test_sessions"])
        )[0]

        rows = np.concatenate([train_idx, test_idx, stranger_idx])
        iq = dataset.apply_variant(data["iq"][rows], ckpt["variant"])
        emb = embed_features(
            net,
            iq_channels(iq),
            device,
            args.batch_size,
        )
        emb_train = emb[: len(train_idx)]
        emb_test = emb[len(train_idx) : len(train_idx) + len(test_idx)]
        emb_stranger = emb[len(train_idx) + len(test_idx) :]

        protos = enroll(
            emb_train,
            np.searchsorted(classes, data["icao"][train_idx]),
            len(classes),
        )
        sims = emb_test @ protos.T
        truth = np.searchsorted(classes, data["icao"][test_idx])
        genuine = sims[np.arange(len(truth)), truth]
        impostor_mask = np.ones_like(sims, dtype=bool)
        impostor_mask[np.arange(len(truth)), truth] = False
        verify_eer = eer(genuine, sims[impostor_mask])

        print(f"== {run_dir.name}  ({args.checkpoint}.pt, epoch {ckpt['epoch']})")
        print(
            f"   variant {ckpt['variant']} · {len(classes)} classes · "
            f"{len(ckpt['test_sessions'])} held-out sessions · "
            f"enrolled from {len(train_idx)} train messages"
        )
        print(
            f"   verify   : EER {verify_eer:.3f}   "
            f"({len(genuine)} genuine, {int(impostor_mask.sum())} impostor trials)"
        )

        auroc = None
        if len(emb_stranger):
            known_scores = sims.max(axis=1)
            stranger_scores = (emb_stranger @ protos.T).max(axis=1)
            auroc = float(
                roc_auc_score(
                    np.concatenate(
                        [
                            np.ones(len(known_scores)),
                            np.zeros(len(stranger_scores)),
                        ]
                    ),
                    np.concatenate([known_scores, stranger_scores]),
                )
            )
            heard = len(set(data["icao"][stranger_idx].tolist()))
            print(
                f"   stranger : AUROC {auroc:.3f}   "
                f"({len(known_scores)} known vs {len(stranger_scores)} stranger "
                f"messages from {heard}/{len(strangers)} excluded ICAOs)"
            )
        elif strangers:
            print("   stranger : no messages from the stranger ICAOs in the held-out sessions")
        else:
            print("   stranger : skipped (run has no --exclude list; pass --strangers)")
        print()

        results.append(
            {
                "run": run_dir.name,
                "eer": verify_eer,
                "auroc": auroc,
                "test_sessions": ckpt["test_sessions"],
                "classes": classes,
            }
        )

    same_split = all(
        r["test_sessions"] == results[0]["test_sessions"]
        and r["classes"] == results[0]["classes"]
        for r in results
    )
    if len(results) > 1 and same_split:
        print("comparison (same split):")
        print(f"  {'run':<40} {'EER':>6} {'AUROC':>6}")
        for r in results:
            auroc = f"{r['auroc']:.3f}" if r["auroc"] is not None else "—"
            print(f"  {r['run']:<40} {r['eer']:>6.3f} {auroc:>6}")
    elif len(results) > 1:
        print("runs differ in split/classes — no comparison")


if __name__ == "__main__":
    main()
