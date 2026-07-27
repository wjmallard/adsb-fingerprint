"""Rescaled ADCC fingerprinting network, after US 2022/0217619 A1.

One stack of gated dilated causal convolution (GDCC) residual blocks over
per-message IQ windows. The patent's second (long-signal subsequence) stage
is dropped: a whole extended squitter is only ~288 samples at 2.4 MSPS, and
the dilation ladder here already sees the entire window. The model emits
logits over ICAO classes; softmax lives in the loss.

Run as a module for a smoke test that overfits one real batch:
`python -m adsb_fingerprint.model`.
"""

import argparse
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from adsb_fingerprint import dataset


def iq_channels(iq):
    """(n, t) complex IQ windows -> (n, 2, t) float32 with I and Q planes."""
    return np.stack(
        [iq.real, iq.imag],
        axis=1,
    ).astype(np.float32)


def pick_device():
    """The best available torch device (MPS on Apple Silicon)."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class GDCCBlock(nn.Module):
    """Gated dilated causal convolution residual block.

    tanh(W_f * x) ⊙ sigmoid(W_g * x), then 1x1 convs to a residual update
    and a skip contribution — the patent's "augmented" (WaveNet-style
    gated) unit. Left-padding keeps the convolution causal.
    """

    def __init__(self, channels, skip_channels, kernel_size, dilation):
        super().__init__()
        self.causal_pad = (kernel_size - 1) * dilation
        self.filter = nn.Conv1d(channels, channels, kernel_size, dilation=dilation)
        self.gate = nn.Conv1d(channels, channels, kernel_size, dilation=dilation)
        self.residual = nn.Conv1d(channels, channels, 1)
        self.skip = nn.Conv1d(channels, skip_channels, 1)

    def forward(self, x):
        padded = F.pad(x, (self.causal_pad, 0))
        gated = torch.tanh(self.filter(padded)) * torch.sigmoid(self.gate(padded))
        return x + self.residual(gated), self.skip(gated)


class CosineHead(nn.Module):
    """Bias-free classifier whose logits are cosine similarities.

    Both the embedding and the per-class weight vectors are L2-normalized,
    so each class collapses to a direction and a logit is cos(angle to it).
    Used with a margin loss so distance in embedding space is trained to
    mean identity — a plain linear head only has to rank classes, which
    leaves strangers indistinguishable from enrolled aircraft.
    """

    def __init__(self, in_features, n_classes):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_classes, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x):
        return F.linear(
            F.normalize(x, dim=1),
            F.normalize(self.weight, dim=1),
        )


class ADCC(nn.Module):
    """Dilated-causal-conv classifier over (batch, 2, time) IQ windows.

    Causal input conv -> GDCC blocks with doubling dilations -> summed
    skips -> 1x1 conv -> global average pool over time -> linear logits
    (or cosine-similarity logits with classify="cosine").
    """

    def __init__(
        self,
        n_classes,
        in_channels=2,
        channels=48,
        skip_channels=48,
        kernel_size=4,
        dilations=(2, 4, 8, 16, 32, 64),
        classify="linear",
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.dilations = tuple(dilations)
        self.input_conv = nn.Conv1d(in_channels, channels, kernel_size)
        self.blocks = nn.ModuleList(
            GDCCBlock(channels, skip_channels, kernel_size, dilation)
            for dilation in self.dilations
        )
        self.head = nn.Conv1d(skip_channels, skip_channels, 1)
        if classify == "linear":
            self.classify = nn.Linear(skip_channels, n_classes)
        elif classify == "cosine":
            self.classify = CosineHead(skip_channels, n_classes)
        else:
            raise ValueError(f"unknown classifier: {classify!r}")

    def receptive_field(self):
        """Input samples the final output position can see."""
        return 1 + (self.kernel_size - 1) * (1 + sum(self.dilations))

    def features(self, x):
        """Pooled embedding (the penultimate layer, before class logits)."""
        x = self.input_conv(F.pad(x, (self.kernel_size - 1, 0)))
        skips = 0
        for block in self.blocks:
            x, skip = block(x)
            skips = skips + skip
        y = F.relu(skips)
        y = F.relu(self.head(y))
        return y.mean(dim=2)

    def forward(self, x):
        return self.classify(self.features(x))


def main():
    parser = argparse.ArgumentParser(
        description="Smoke-test the ADCC model by overfitting one real batch from the index.",
    )
    parser.add_argument("--classes", type=int, default=8, help="ICAO classes in the smoke batch.")
    parser.add_argument("--per-class", type=int, default=8, help="Messages per class in the smoke batch.")
    parser.add_argument("--steps", type=int, default=300, help="Adam steps.")
    parser.add_argument("--lr", type=float, default=3e-3, help="Learning rate.")
    args = parser.parse_args()

    data = dataset.load_examples(min_messages=args.per_class)
    by_icao = defaultdict(list)
    for i, icao in enumerate(data["icao"].tolist()):
        by_icao[icao].append(i)
    top = sorted(
        by_icao,
        key=lambda icao: len(by_icao[icao]),
        reverse=True,
    )[: args.classes]
    picks = [i for icao in top for i in by_icao[icao][: args.per_class]]

    x = torch.from_numpy(iq_channels(data["iq"][picks]))
    y = torch.tensor([top.index(icao) for icao in data["icao"][picks].tolist()])

    device = pick_device()
    model = ADCC(n_classes=len(top)).to(device)
    x = x.to(device)
    y = y.to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"device          : {device.type}")
    print(f"parameters      : {n_params:,}")
    print(f"receptive field : {model.receptive_field()} samples (window {data['window']})")
    print(f"smoke batch     : {tuple(x.shape)} over {len(top)} classes")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    for step in range(1, args.steps + 1):
        optimizer.zero_grad()
        loss = F.cross_entropy(model(x), y)
        loss.backward()
        optimizer.step()
        if step == 1 or step % 50 == 0:
            with torch.no_grad():
                accuracy = (model(x).argmax(dim=1) == y).float().mean().item()
            print(f"step {step:4d}  loss {loss.item():.4f}  batch accuracy {accuracy:.2f}")

    with torch.no_grad():
        final = F.cross_entropy(model(x), y).item()
    if final < 0.1:
        print(f"final loss {final:.4f}: OK — gradients flow and the stack can memorize")
    else:
        print(f"final loss {final:.4f}: NOT overfitting — something is wrong")


if __name__ == "__main__":
    main()
