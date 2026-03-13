# CLAUDE.md — engine/codec/

## Responsibility

Low-level coding primitives used by all signal layers:
- Reed-Solomon encode/decode with erasure support
- DSSS chip stream generation and normalized correlation
- Frequency hopping plan generation (per-segment band configs)

No audio or video domain knowledge here. These are generic coding tools.

## Boundary rules

MUST NOT import: `fastapi`, `sqlalchemy`, `arq`, `boto3`, `ffmpeg`, `cv2`, `pywt`
MAY import: `numpy`, `reedsolo`, Python stdlib, `kernel_backend.core.domain.watermark` (BandConfig)

## Module contracts

### reed_solomon.py

```python
class ReedSolomonCodec:
    def __init__(self, n_symbols: int, k_data: int = 16) -> None
        # Uses reedsolo.RSCodec(n_symbols - k_data) internally

    def encode(self, data: bytes) -> list[int]
        # len(data) must == k_data (16); returns list of n_symbols ints

    def decode(self, symbols: list[int | None]) -> bytes
        # None = erasure; raises ReedSolomonError if uncorrectable
        # Returns exactly k_data bytes, padded: decoded.rjust(16, b'\x00')
```
Key property: `decode(encode(data, N) with any K-or-fewer erasures) == data`
**reedsolo 1.7.0 quirk:** `decode()` returns a 3-tuple `(msg, msgecc, errata)` —
there is NO `nostrip` parameter. Leading zero bytes are preserved via `rjust(16, b'\x00')`.

### spread_spectrum.py

```python
# deterministic PN sequence from integer seed
pn_sequence(length: int, seed: int) -> np.ndarray  # values in {-1.0, +1.0}

# DSSS chip stream: payload bits × chips_per_bit
chip_stream(bits: np.ndarray, chips_per_bit: int, seed: int) -> np.ndarray

# normalized cross-correlation in [-1, 1]
normalized_correlation(window: np.ndarray, template: np.ndarray) -> float
```
Key property: `pn_sequence(L, seed)` is deterministic — same output for same (L, seed).

### hopping.py

```python
# generates per-segment band config from system pepper + content key
plan_audio_hopping(
    n_segments: int, content_id: str, author_pubkey: str, pepper: bytes
) -> list[BandConfig]   # len = n_segments

plan_video_hopping(
    n_segments: int, content_id: str, author_pubkey: str, pepper: bytes
) -> list[BandConfig]   # len = n_segments
```
Key property: same inputs → same output (deterministic PRNG from HMAC seed).
Video hopping must include ≥2 positions from robust set {(0,1),(1,0)} per segment.

## Validation

```bash
python -m pytest tests/unit/test_reed_solomon.py -v
python -m pytest tests/unit/test_spread_spectrum.py -v
python -m pytest tests/unit/test_hopping.py -v
```

Expected test cases:
- RS roundtrip with 0, K//2, K-1, and K erasures
- RS raises `ReedSolomonError` with K+1 erasures
- PN sequence determinism across multiple calls
- Hopping plan determinism (same inputs, two calls, identical output)
- Video hopping robust-set constraint satisfied for all segments
