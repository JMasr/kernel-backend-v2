#!/usr/bin/env bash
# =============================================================================
# Kernel Security v2.0 — Greenfield Repository Scaffold
# Creates src/ layout: all source code lives under src/kernel_backend/
# Run from the repo root: bash scaffold.sh
# =============================================================================
set -euo pipefail

ROOT="$(pwd)"
SRC="$ROOT/src/kernel_backend"
echo "▶ Scaffolding Kernel Security v2.0 at: $ROOT"
echo "  Source root: $SRC"

mkmod() { mkdir -p "$1" && touch "$1/__init__.py"; }

# ── Package root ──────────────────────────────────────────────────────────────
mkmod "$SRC"

# ── core/ ─────────────────────────────────────────────────────────────────────
mkmod "$SRC/core"
mkmod "$SRC/core/domain"
touch  "$SRC/core/domain/manifest.py"
touch  "$SRC/core/domain/watermark.py"
touch  "$SRC/core/domain/identity.py"
touch  "$SRC/core/domain/media.py"
mkmod "$SRC/core/ports"
touch  "$SRC/core/ports/signing.py"
touch  "$SRC/core/ports/verification.py"
touch  "$SRC/core/ports/storage.py"
touch  "$SRC/core/ports/registry.py"
touch  "$SRC/core/ports/embedder.py"
mkmod "$SRC/core/services"
touch  "$SRC/core/services/crypto_service.py"
touch  "$SRC/core/services/signing_service.py"
touch  "$SRC/core/services/verification_service.py"

# ── engine/ ───────────────────────────────────────────────────────────────────
mkmod "$SRC/engine"
mkmod "$SRC/engine/audio"
touch  "$SRC/engine/audio/pilot_tone.py"
touch  "$SRC/engine/audio/wid_beacon.py"
touch  "$SRC/engine/audio/fingerprint.py"
mkmod "$SRC/engine/video"
touch  "$SRC/engine/video/pilot_tone.py"
touch  "$SRC/engine/video/wid_watermark.py"
touch  "$SRC/engine/video/fingerprint.py"
mkmod "$SRC/engine/codec"
touch  "$SRC/engine/codec/reed_solomon.py"
touch  "$SRC/engine/codec/spread_spectrum.py"
touch  "$SRC/engine/codec/hopping.py"
mkmod "$SRC/engine/perceptual"
touch  "$SRC/engine/perceptual/jnd_model.py"
touch  "$SRC/engine/perceptual/psychoacoustic.py"

# ── infrastructure/ ───────────────────────────────────────────────────────────
mkmod "$SRC/infrastructure"
mkmod "$SRC/infrastructure/storage"
touch  "$SRC/infrastructure/storage/local_storage.py"
touch  "$SRC/infrastructure/storage/r2_storage.py"
mkmod "$SRC/infrastructure/database"
touch  "$SRC/infrastructure/database/models.py"
touch  "$SRC/infrastructure/database/repositories.py"
touch  "$SRC/infrastructure/database/session.py"
mkdir -p "$SRC/infrastructure/database/migrations/versions"
touch  "$SRC/infrastructure/database/migrations/__init__.py"
touch  "$SRC/infrastructure/database/migrations/env.py"
mkmod "$SRC/infrastructure/media"
touch  "$SRC/infrastructure/media/media_service.py"
mkmod "$SRC/infrastructure/queue"
touch  "$SRC/infrastructure/queue/worker.py"
touch  "$SRC/infrastructure/queue/jobs.py"
touch  "$SRC/infrastructure/queue/redis_pool.py"

# ── api/ ──────────────────────────────────────────────────────────────────────
mkmod "$SRC/api"
mkmod "$SRC/api/identity"
touch  "$SRC/api/identity/router.py"
touch  "$SRC/api/identity/schemas.py"
mkmod "$SRC/api/signing"
touch  "$SRC/api/signing/router.py"
touch  "$SRC/api/signing/schemas.py"
mkmod "$SRC/api/verification"
touch  "$SRC/api/verification/router.py"
touch  "$SRC/api/verification/schemas.py"

# ── tests/ ────────────────────────────────────────────────────────────────────
mkmod "$ROOT/tests"
mkmod "$ROOT/tests/unit"
for f in test_crypto_service test_reed_solomon test_spread_spectrum test_hopping \
          test_audio_pilot test_audio_wid test_video_pilot test_video_wid \
          test_fingerprint_audio test_fingerprint_video \
          test_storage_port test_score_combiner; do
  touch "$ROOT/tests/unit/$f.py"
done
mkmod "$ROOT/tests/integration"
for f in test_sign_verify_av test_sign_verify_audio_only test_sign_verify_video_only \
          test_wid_recovery_compressed test_wid_recovery_trimmed test_tampered_file_red; do
  touch "$ROOT/tests/integration/$f.py"
done
mkmod "$ROOT/tests/boundary"
touch  "$ROOT/tests/boundary/test_import_boundaries.py"
mkmod "$ROOT/tests/fixtures"
touch  "$ROOT/tests/fixtures/conftest.py"
touch  "$ROOT/tests/fixtures/media_factories.py"

# ── data/ ─────────────────────────────────────────────────────────────────────
mkdir -p "$ROOT/data/fixtures"
touch  "$ROOT/data/fixtures/manifest.yaml"

# ── scripts/ ──────────────────────────────────────────────────────────────────
mkdir -p "$ROOT/scripts"
touch  "$ROOT/scripts/gen_fixtures.py"
touch  "$ROOT/scripts/check_boundaries.py"
touch  "$ROOT/scripts/reset_db.py"

# ── alembic/ ──────────────────────────────────────────────────────────────────
mkdir -p "$ROOT/alembic/versions"
touch  "$ROOT/alembic/env.py"
touch  "$ROOT/alembic/script.py.mako"
touch  "$ROOT/alembic.ini"

# ── root files ────────────────────────────────────────────────────────────────
touch "$ROOT/main.py"
touch "$ROOT/config.py"
touch "$ROOT/dependencies.py"
touch "$ROOT/pyproject.toml"
touch "$ROOT/.env.example"
touch "$ROOT/.gitignore"
touch "$ROOT/Makefile"

echo ""
echo "✔ Scaffold complete (src/ layout)."
echo "  Import style:  from kernel_backend.core.domain.manifest import ..."
echo "  Run tests:     python -m pytest tests/ (pythonpath=src in pyproject.toml)"
