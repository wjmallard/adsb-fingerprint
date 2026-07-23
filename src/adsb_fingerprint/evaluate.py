"""adsb-eval: cross-session evaluation, ablation comparison, and baselines.

Loads adsb-train checkpoints, rebuilds each one's exact held-out set (its
recorded sessions, classes, and ablation variant), and reports overall +
balanced accuracy, a per-class breakdown with registry labels, and the top
confusions. Given several runs over the same split — the whole /
icao_masked / preamble ablation trio — it adds a comparison table.

Two reference classifiers score the same split from cheap per-message
features (features.py), no waveform learning:

  channel-only    {rssi_db, snr_db} — pure propagation. If this scores
                  high, the deep model may just be a range detector.
  cheap-features  {rssi_db, snr_db, cfo_hz} — measured CFO is mostly
                  oscillator offset (hardware), so this is a trivial
                  handcrafted fingerprinter the deep model has to beat.

Baseline features are always measured on the unablated IQ: they describe
the received message, not the model's masked view of it.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm

from adsb_fingerprint import config, dataset, features
from adsb_fingerprint.model import (
    ADCC,
    iq_channels,
    pick_device,
)

BASELINES = {
    "channel-only": ["rssi_db", "snr_db"],
    "cheap-features": ["rssi_db", "snr_db", "cfo_hz"],
}


def load_run(run_dir, which):
    """Return (checkpoint dict, model in eval mode) from a run directory."""
    ckpt = torch.load(run_dir / f"{which}.pt", map_location="cpu")
    net = ADCC(**ckpt["model"])
    net.load_state_dict(ckpt["state_dict"])
    net.eval()
    return ckpt, net


def split_for(ckpt, data):
    """(train_idx, test_idx) into the loaded arrays for a checkpoint's split."""
    keep = np.isin(data["icao"], ckpt["classes"])
    train_idx = np.where(keep & np.isin(data["session"], ckpt["train_sessions"]))[0]
    test_idx = np.where(keep & np.isin(data["session"], ckpt["test_sessions"]))[0]
    return train_idx, test_idx


def predict(net, x, device, batch_size):
    """Class predictions for (n, 2, t) float32 input, batched on device."""
    net = net.to(device)
    preds = []
    with torch.no_grad():
        for lo in range(0, len(x), batch_size):
            batch = torch.from_numpy(x[lo : lo + batch_size]).to(device)
            preds.append(net(batch).argmax(dim=1).cpu().numpy())
    return np.concatenate(preds) if preds else np.empty(0, dtype=int)


def metrics(preds, truth, n_classes):
    """(accuracy, balanced accuracy, per-class recall, confusion matrix)."""
    confusion = np.zeros((n_classes, n_classes), dtype=int)
    np.add.at(confusion, (truth, preds), 1)
    support = confusion.sum(axis=1)
    recalls = np.divide(
        np.diag(confusion),
        support,
        out=np.full(n_classes, np.nan),
        where=support > 0,
    )
    return (
        float((preds == truth).mean()),
        float(np.nanmean(recalls)),
        recalls,
        confusion,
    )


def feature_matrix(iq, rssi_db, index):
    """(len(index), n_features) matrix of cheap features for the given rows."""
    rows = np.empty((len(index), 1 + len(features.FEATURES)))
    for out_row, i in enumerate(
        tqdm(
            index,
            desc="features",
            leave=False,
        )
    ):
        measured = features.extract(iq[i], config.SAMPLE_RATE_HZ)
        rows[out_row, 0] = rssi_db[i]
        rows[out_row, 1:] = [measured[name] for name in features.FEATURES]
    return rows


def fit_baseline(f_train, y_train, f_test, n_classes, seed, steps=800):
    """Small-MLP predictions on standardized features (NaN imputed at mean)."""
    torch.manual_seed(seed)
    mean = np.nanmean(f_train, axis=0)
    std = np.nanstd(f_train, axis=0)
    std[std == 0] = 1.0

    def prep(f):
        z = (f - mean) / std
        return torch.from_numpy(np.nan_to_num(z).astype(np.float32))

    x_train = prep(f_train)
    x_test = prep(f_test)
    y = torch.from_numpy(y_train)
    net = nn.Sequential(
        nn.Linear(x_train.shape[1], 32),
        nn.ReLU(),
        nn.Linear(32, n_classes),
    )
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-2)
    for _ in range(steps):
        optimizer.zero_grad()
        loss = F.cross_entropy(net(x_train), y)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        return net(x_test).argmax(dim=1).numpy()


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate adsb-train checkpoints on their held-out sessions.",
    )
    parser.add_argument("runs", nargs="+", help="Run directories (from adsb-train).")
    parser.add_argument(
        "--checkpoint",
        choices=("best", "last"),
        default="best",
        help="Which checkpoint to load from each run.",
    )
    parser.add_argument("--no-baseline", action="store_true", help="Skip the feature baselines.")
    parser.add_argument(
        "--per-class",
        action="store_true",
        help="Print the per-class table for every run, not just the first.",
    )
    parser.add_argument("--batch-size", type=int, default=512, help="Eval batch size.")
    parser.add_argument("--seed", type=int, default=0, help="Baseline training seed.")
    parser.add_argument("--top", type=int, default=10, help="Per-class rows to show (0 = all).")
    args = parser.parse_args()

    run_dirs = [Path(run).expanduser() for run in args.runs]
    loaded = [load_run(run_dir, args.checkpoint) for run_dir in run_dirs]
    device = pick_device()
    data = dataset.load_examples()
    labels = dataset._labels(sorted({icao for ckpt, _ in loaded for icao in ckpt["classes"]}))

    results = []
    for run_dir, (ckpt, net) in zip(run_dirs, loaded):
        classes = ckpt["classes"]
        train_idx, test_idx = split_for(ckpt, data)
        iq = dataset.apply_variant(data["iq"][test_idx], ckpt["variant"])
        preds = predict(net, iq_channels(iq), device, args.batch_size)
        truth = np.searchsorted(classes, data["icao"][test_idx])
        accuracy, balanced, recalls, confusion = metrics(preds, truth, len(classes))
        results.append(
            {
                "run": run_dir.name,
                "variant": ckpt["variant"],
                "accuracy": accuracy,
                "balanced": balanced,
                "ckpt": ckpt,
                "train_idx": train_idx,
                "test_idx": test_idx,
            }
        )

        print(f"== {run_dir.name}  ({args.checkpoint}.pt, epoch {ckpt['epoch']})")
        print(
            f"   variant {ckpt['variant']} · {len(classes)} classes · "
            f"{len(ckpt['test_sessions'])} held-out sessions · {len(test_idx)} messages"
        )
        print(
            f"   accuracy {accuracy:.3f}  balanced {balanced:.3f}   "
            f"(train-time: {ckpt['accuracy']:.3f} / {ckpt['balanced_accuracy']:.3f})"
        )

        if args.per_class or run_dir is run_dirs[0]:
            order = np.argsort(-confusion.sum(axis=1))
            shown = order if args.top == 0 else order[: args.top]
            print("   per class (by held-out messages):")
            for c in shown:
                row = confusion[c].copy()
                row[c] = 0
                confused = ""
                if row.sum() > 0:
                    worst = int(row.argmax())
                    confused = f"→ {classes[worst]} ×{row[worst]}"
                print(
                    f"     {classes[c]}  n {confusion[c].sum():4d}  "
                    f"recall {recalls[c]:.2f}  {confused:<18}  {labels.get(classes[c], '')}"
                )
            if len(shown) < len(classes):
                print(f"     ... {len(classes) - len(shown)} more (--top 0 for all)")
        print()

    same_split = all(
        r["ckpt"]["test_sessions"] == results[0]["ckpt"]["test_sessions"]
        and r["ckpt"]["classes"] == results[0]["ckpt"]["classes"]
        for r in results
    )

    baseline_rows = []
    if not args.no_baseline and same_split:
        ckpt = results[0]["ckpt"]
        train_idx = results[0]["train_idx"]
        test_idx = results[0]["test_idx"]
        f_train = feature_matrix(data["iq"], data["rssi_db"], train_idx)
        f_test = feature_matrix(data["iq"], data["rssi_db"], test_idx)
        names = ["rssi_db"] + features.FEATURES
        y_train = np.searchsorted(ckpt["classes"], data["icao"][train_idx])
        truth = np.searchsorted(ckpt["classes"], data["icao"][test_idx])
        for baseline, wanted in BASELINES.items():
            cols = [names.index(name) for name in wanted]
            preds = fit_baseline(
                f_train[:, cols],
                y_train,
                f_test[:, cols],
                len(ckpt["classes"]),
                args.seed,
            )
            accuracy, balanced, _, _ = metrics(preds, truth, len(ckpt["classes"]))
            baseline_rows.append((baseline, " ".join(wanted), accuracy, balanced))
    elif not args.no_baseline:
        print("runs differ in split/classes — skipping baselines and comparison\n")

    if same_split and (len(results) > 1 or baseline_rows):
        print(f"comparison (same {len(results[0]['ckpt']['test_sessions'])} held-out sessions, "
              f"{len(results[0]['ckpt']['classes'])} classes):")
        print(f"  {'model':<24} {'accuracy':>8} {'balanced':>9}")
        for r in results:
            print(f"  {r['variant']:<24} {r['accuracy']:>8.3f} {r['balanced']:>9.3f}")
        for baseline, wanted, accuracy, balanced in baseline_rows:
            print(f"  {baseline:<24} {accuracy:>8.3f} {balanced:>9.3f}   [{wanted}]")


if __name__ == "__main__":
    main()
