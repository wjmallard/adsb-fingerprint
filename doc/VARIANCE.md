# Variance decomposition: is 3-per-10 s the right sampling policy?

`adsb-variance` answers the replication-vs-coverage question empirically: when
the collector caps storage at 3 back-to-back messages per aircraft per 10 s
window (+1 ident), is that the right split between *replicating* a measurement
and *spreading* samples across time and geometry?

## Method

Cheap deterministic features are extracted per message from the stored IQ
snippets (`features.py`; the preamble starts at sample 0):

- `cfo_hz` — carrier frequency offset, staged phase ladder (below)
- `p14_spacing_ns` — preamble pulse-1 → pulse-4 centroid spacing (tx bit clock)
- `preamble_ratio_db` — late/early preamble pulse amplitude ratio (PA settling)
- `rise_ns` — 10–90% rise of preamble pulse 3 (the only early pulse with a
  guaranteed quiet microsecond before it — the snippet starts mid-rise of pulse 1)
- `snr_db` — message power over the post-message tail
- `rssi_db` — from the index, as the pure-channel control

Each feature's within-aircraft variance is then split into three nested levels
— **session > 10 s window > message** — with the unbalanced-nested method of
moments (Henderson), pooled across aircraft (validated against synthetic
ground truth, including unbalanced and zero-component designs). Per aircraft,
values further than `--screen-sd` (default 10) robust sd from the aircraft
median are dropped and counted: a single corrupted extraction (e.g. an
overlapping transmission that still passes CRC — one C01E10 message measured
−778 kHz) would otherwise dominate the plain-moments components.

The components translate into policy: the within-window ICC gives the
effective sample size of k reps, and k\* = sqrt(var_msg / var_win) is where
extra reps stop beating new windows.

## Results (2026-07-23 snapshot: 41k msgs, 347 aircraft, 25 sessions)

| feature             | sd(msg) | sd(win) | sd(sess) | between-aircraft | ratio |
|---------------------|--------:|--------:|---------:|-----------------:|------:|
| `cfo_hz`            |  620 Hz |  444 Hz |  2.6 kHz |         51.5 kHz |  19.1 |
| `p14_spacing_ns`    |   28 ns |    ~0   |    ~1 ns |           8.1 ns |   0.3 |
| `preamble_ratio_db` | 1.09 dB | 0.33 dB |  0.07 dB |          0.44 dB |   0.4 |
| `rise_ns`           |  142 ns |    ~0   |     ~0   |            40 ns |   0.3 |
| `snr_db`            |  3.4 dB |  3.7 dB |   1.8 dB |                — |     — |
| `rssi_db`           |  4.0 dB |  3.8 dB |   1.7 dB |                — |     — |

- **CFO is already a fingerprint on its own** (between/within ≈ 19). Aircraft
  crystals sit tens of kHz apart (spec tolerance is ±1 MHz) while one aircraft
  holds within a few hundred Hz — e.g. A8990F at −19.6 kHz ± ~150 Hz across 6
  sessions. 92% of its within-aircraft variance is *between sessions*
  (oscillator drift + doppler): cross-session invariance is the real battle,
  and replication adds almost nothing (window ICC 0.95 → 3 reps ≈ 1.04
  independent messages).
- **Envelope-shape features are the mirror image**: ~all measurement noise
  (ICC ≈ 0), so reps average them down at sqrt(N), and k\* ≈ 3.
- **Fast fading is confirmed**: rssi scatters ~4 dB *within* a back-to-back
  burst, so even tight reps sample independent fades; rssi/snr put 50–60% of
  variance at window+session level, as a channel control should.

**Verdict: keep 3-per-10 s.** Replication is sized for the features that need
it (envelope shape); CFO and timing get their information from window and
session spread, which the continuous collection campaign provides anyway.

## CFO estimator notes

Naive lag-1 phase scatters tens of kHz: at 2.4 Msps an on-chip is 1.2 samples,
so adjacent-sample products are mostly noise. The shipped estimator is a
staged ladder, each stage unambiguous far beyond the previous residual:
gated lag-1 seed (±fs/2) → 1 µs segment phasors compared at 1/8/32 µs gaps
(±500/62/16 kHz) → slip-safe weighted phase regression over the full 112 µs
data block. Within-burst repeatability is ~260 Hz (robust) and flat with SNR;
the *error tail* is SNR-driven (p99 |err| 12.8 kHz at 10–15 dB SNR vs 2.6 kHz
at 25–40 dB), which is the empirical case for RSSI-aware curation in
`dataset.py`. The receiver's own E4000 offset is common-mode across aircraft
within a session; per-session receiver drift correction (e.g. referencing a
stable repeat-flyer) would sharpen sd(sess) further but is not needed for the
policy question.

## Regenerating

```
adsb-variance                     # full report, per-aircraft table for cfo_hz
adsb-variance --feature rise_ns   # per-aircraft table for another feature
adsb-variance --screen-sd 0       # no outlier screening (raw moments)
adsb-variance --window-seconds 30 # different middle-level timescale
```

Numbers above are a snapshot; the corpus grows continuously, and pre-cap /
20-per-60 s / 10-per-30 s era sessions (documented in each `session.yaml`)
are all included — the window level is a uniform wall-clock bucketing, so the
decomposition is policy-agnostic.
