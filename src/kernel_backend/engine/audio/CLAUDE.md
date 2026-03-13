# CLAUDE.md — engine/audio/

## Responsibility

Audio-domain watermark embedding and extraction across three layers:
- Layer 0 (pilot_tone.py): 48-bit content identifier via DSSS in DWT approximation band
- Layer 1 (wid_beacon.py): 1 Reed-Solomon symbol per segment via DWT detail band + hopping
- Layer 2 (fingerprint.py): speech-optimized log-mel → DCT → keyed projection → 64-bit hash (Phase 2a)

All functions operate on `np.ndarray` float32 samples in [-1.0, 1.0].
No file I/O — raw samples in, raw samples out.

## Boundary rules

MUST NOT import: `fastapi`, `sqlalchemy`, `arq`, `boto3`, `ffmpeg`
MAY import: `numpy`, `scipy`, `pywt`, `reedsolo`, `engine/codec/`

## DWT configuration

- Wavelet: `db4`, level 2
- Target band (Layer 1): `coeffs[-2]` (detail level 2, ~5.5–11 kHz at 44.1 kHz)
- Pilot band (Layer 0): `coeffs[0]` (approximation, lowest frequency — max robustness)
- Reconstruction: `pywt.waverec(..., mode="periodization")`, trim/pad to original length

## Module contracts

### pilot_tone.py

```python
embed_pilot(samples: np.ndarray, sample_rate: int, hash_48: int,
            global_pn_seed: int, chips_per_bit: int = 64,
            target_snr_db: float = -14.0) -> np.ndarray

detect_pilot(samples: np.ndarray, sample_rate: int,
             global_pn_seed: int, chips_per_bit: int = 64,
             threshold: float = 0.15) -> int | None  # returns 48-bit hash or None
```

### wid_beacon.py

```python
embed_segment(segment: np.ndarray, rs_symbol: int,
              band_config: BandConfig, pn_seed: int,
              chips_per_bit: int = 32,
              target_snr_db: float = -14.0) -> np.ndarray

extract_segment(segment: np.ndarray, band_config: BandConfig,
                pn_seed: int, chips_per_bit: int = 32) -> float  # soft correlation [0,1]
```

### fingerprint.py

```python
extract_hashes(samples: np.ndarray, sample_rate: int, key_material: bytes,
               pepper: bytes, segment_duration_s: float = 2.0,
               overlap: float = 0.5, f_min: float = 300.0,
               f_max: float = 8000.0) -> list[SegmentFingerprint]

hamming_distance(hash_a: str, hash_b: str) -> int
```

`SegmentFingerprint` fields: `time_offset_ms: int`, `hash_hex: str` (16-char hex = 64 bits).

## Phase 2a — Speech-Optimized Fingerprint Pipeline

`fingerprint.py` targets **speech** as its primary domain (f_min=300, f_max=8000 Hz).
The pipeline applies three noise-robustness layers inside `_log_mel_spectrogram()`,
plus one normalization in `_compute_hash()`:

1. **STFT-level spectral subtraction** — before mel filterbank:
   `noise_floor = np.percentile(power, 5.0, axis=1) * 1.5`; clip to ≥1e-10.
   Removes stationary/slowly-varying background noise before it compounds through mel.

2. **Energy-weighted time frames** — softmax(2.0 × frame_energy) × mel matrix.
   Upweights high-SNR (speech-dominated) frames; suppresses silence/babble frames.
   `energy_weight=2.0` chosen empirically to balance babble robustness and VoIP robustness.

3. **Per-band mean removal** — `resized - resized.mean(axis=1, keepdims=True)` on the
   resized 32×32 mel matrix. Removes residual stationary noise per mel frequency band
   (equivalent to per-band CMVN). Replaced the earlier global-mean subtraction.

4. **L2 normalization of DCT block** — before keyed projection. Normalizes the 60-dim
   DCT block to unit norm; projection captures spectral shape (direction), not overall energy.
   Raised clean-vs-10dB-babble correlation from 0.25 → >0.90.

**Release gate (all SPEECH_RECORDINGS):** babble@10dB≥80%, pink@20dB≥85%,
mp3@32kbps≥80%, VoIP-G.711≥70%, different-speaker discrimination≥80%. All pass.

## Internal tuning parameters (not exposed in extract_hashes API)

| Parameter | Value | Purpose |
|---|---|---|
| `noise_floor_pct` | 5.0 | Percentile for STFT noise floor estimate |
| `noise_oversub` | 1.5 | Oversubtraction factor |
| `energy_weight` | 2.0 | Softmax sharpness for frame weighting |
| DCT block shape | 12 × 5 | 60-dim feature vector |
| Projection dim | 60 | Matches DCT block size |
| Pre-emphasis coeff | 0.97 | Standard speech pre-emphasis |

## Test signals — CRITICAL

**NEVER use pure tones (sine/square/sawtooth) in DWT watermarking tests.**
Spectral concentration → near-zero band RMS in detail bands → near-zero embedding amplitude
→ false negatives unrelated to the implementation.

Use ONLY:
- `white_noise(seed)`, `pink_noise(seed)`, `multitone_dwt()` from `tests/fixtures/audio_signals.py`
- `librosa.ex()` recordings for integration tests (libri1, libri2, choice, brahms, trumpet, vibeace)

## Validation

```bash
python -m pytest tests/unit/test_audio_pilot.py -v
python -m pytest tests/unit/test_audio_wid.py -v
python -m pytest tests/unit/test_fingerprint_audio.py -v              # 19 blocking + 5 structural
python -m pytest tests/unit/test_fingerprint_audio.py -m integration  # speech integration (25)
```

Key test cases:
- embed_pilot → detect_pilot roundtrip on white noise at -14 dB SNR
- embed_segment × N_seg → RS decode recovers WID with 30% segments as erasures
- fingerprint deterministic: same audio + same key → same hashes
- fingerprint robustness: babble@10dB, pink@20dB, mp3@32kbps, VoIP all pass release gate
- fingerprint discrimination: different speakers → Hamming ≥ 20 for ≥80% segment pairs
