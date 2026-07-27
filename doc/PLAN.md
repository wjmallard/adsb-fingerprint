# ADS-B Transponder Fingerprinting — Design

A toy implementation of the RF-fingerprinting idea in US 2022/0217619 A1
("Artificial Intelligence Radio Classifier and Identifier", BAE Systems),
rescaled to what a Nooelec NESDR SMArt XTR (RTL2832U, 8-bit, ~2.4 MSPS,
E4000 tuner) can actually receive: ADS-B / Mode S at 1090 MHz.

## Goal

Capture Mode S extended squitters, label each burst by its 24-bit ICAO address
(free ground truth), and train a rescaled Augmented Dilated Causal Convolution
(ADCC) network to identify the *physical transponder* from its hardware
fingerprint — then test whether that survives with the ID bits masked and,
crucially, across different receive geometries.

The operational question — given IQ from an unidentified transmitter — comes
in tiers, each with a confidence attached:

1. **Verify**: it claims ICAO X; does the waveform match X's enrolled
   fingerprint? (1-vs-1, the anti-spoof check.)
2. **Identify**: which enrolled airframe is this? (1-of-N — what the
   classifier does today.)
3. **Fall back to type/operator**: if it matches no enrolled airframe, what
   kind is it and whose fleet does its embedding neighborhood suggest?
   Same-fleet confusion (sibling A320s, flight-school 172s) is the answer at
   this tier, not a failure.

Tiers 1 and 3 need calibrated distances in embedding space rather than a
closed-set softmax; the evals they imply are genuine-vs-impostor ROC and
held-out-airframe (open-set) tests.

## The channel confound

We cannot control aircraft geometry, so the easy signal correlated with ICAO is
the propagation channel (range / SNR / multipath / Doppler), not transponder
hardware. A high in-session score most likely means we built a "how far away"
detector. The experimental design below exists to disprove that; treat a clean
*negative* result as a legitimate outcome.

## Signal facts (at 2.4 MSPS)

- 1090 MHz, PPM, 1 Mbit/s (0.5 us per half-bit).
- 8 us preamble, fixed for all aircraft -> ~19 samples, pure hardware+channel.
- 112-bit body -> 112 us; whole message 120 us -> ~288 complex samples.
- ICAO in bits 9-32 (us ~16-40 of the body) -> ~58 samples, indices ~38-96.

Consequence: each example is ~288 IQ samples. We drop the patent's second
(long-signal subsequence) stage and keep one rescaled stack.

## Storage

- `data/captures/*.cf32` — raw interleaved float32 IQ. Source of truth.
- Postgres `messages` — one row per detected message (icao, capture_file,
  sample_offset, session, crc_ok, rssi, ...). Rebuild by re-running `adsb-index`.
- Schema in `sql/schema.sql`; psycopg3 + dict_row; local peer auth (dbname only).

## Modules / commands

- `capture.py`  -> `adsb-capture`: pyrtlsdr stream -> timestamped cf32 files.
- `modes.py`: preamble correlator, PPM demod, CRC/ICAO via pyModeS.
- `index.py`   -> `adsb-index`: detect over captures -> insert rows (rebuildable).
- `dataset.py`: query index, split BY SESSION, IQ-slice loader, normalize, masking.
- `model.py`: GDCC block + rescaled ADCC.
- `train.py`   -> `adsb-train`; `evaluate.py` -> `adsb-eval`.

## Model (rescaled ADCC, PyTorch/MPS)

- Input (2, ~320): I and Q channels, padded.
- Initial causal conv, then 5-6 GDCC residual blocks, dilations 2,4,8,16,32(,64),
  kernel 4, ~48 filters. GDCC = tanh(W_f*x) (*) sigmoid(W_g*x) -> 1x1 conv -> residual+skip.
- Sum skips -> ReLU -> conv+avgpool -> dense -> softmax(N ICAOs).
- Receptive field 1 + 3*sum(d): 5 blocks ~= 187 samples; +dilation-64 ~= 379 (covers 288).

## Experimental design

1. Split by session/time, never random. Train on some sessions, test on the same
   ICAOs in held-out sessions (different geometry). Survival = hardware.
2. Ablations: whole message -> ICAO-masked -> preamble-only. Masked ~= full =>
   not reading the ID; preamble-only working => genuinely hardware.
3. Channel-only baseline: classify from {power, SNR, Doppler} alone. If that
   scores high, discount the model until it clearly beats it cross-session.
4. Amplitude-normalize per message (/ max|z|); keep ICAOs with >=N msgs over >=2
   sessions.

## Build phases (atomic chunks)

- P0 scaffold: pyproject/config/schema/db + `adsb-initdb`.  <- current
- P1 `adsb-capture`.
- P2 `adsb-index` (+ validate ICAOs against dump1090).
- P3 dataset (session splits, loader, masking).
- P4 model (adds torch dep).
- P5 `adsb-train`.
- P6 `adsb-eval` (cross-session, ablations, baseline).
- Stretch: open-set/zero-shot, UMAP/DBSCAN viewer (Flask+Jinja), multi-burst.

## Prerequisites

- `brew install librtlsdr` before P1 (pyrtlsdr loads it at runtime).
- A Postgres DB: `createdb radio_classifier`, then `adsb-initdb`.
