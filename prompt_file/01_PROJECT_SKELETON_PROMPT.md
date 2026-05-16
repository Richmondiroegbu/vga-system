# Prompt 01: Project Skeleton & Directory Structure
**Category:** Infrastructure  
**Files to create:**
- `pyproject.toml`, `requirements.txt`, `requirements-dev.txt`, `requirements.lock`
- `Makefile`
- `README.md`
- `.env.example`
- `DEVIATION_LOG.md`
- `.pre-commit-config.yaml`
- `vga/__init__.py` and all package `__init__.py` files
- `vga/bootstrap.py` (startup singleton initialization)
**Dependencies:** None (Phase 1, Step 1)  
**Spec Reference:** `12_VGA_Project_Skeleton_Anchor_Files_v17.2.md`

## Requirements

### pyproject.toml
```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.backends.legacy:BuildBackend"

[project]
name = "vga"
version = "17.2.0"
description = "Video Generation Automation — Cinematic AI Motivation Pipeline"
requires-python = ">=3.10"
dependencies = [
    "torch>=2.5.1",
    "torchvision",
    "torchaudio",
    "diffusers==0.30.0",
    "transformers==4.45.0",
    "accelerate==0.34.0",
    "peft==0.12.0",
    "safetensors==0.4.3",
    "open-clip-torch>=2.24.0",
    "Pillow>=10.0.0",
    "numpy>=1.24.0",
    "opencv-python>=4.9.0",
    "pydub>=0.25.0",
    "ffmpeg-python>=0.2.0",
    "fastapi>=0.115.0",
    "uvicorn>=0.30.0",
    "pydantic>=2.0.0",
    "pydantic-settings>=2.0.0",
    "streamlit>=1.30.0",
    "scipy",
    "omegaconf",
    "einops",
    "mediapipe",
    "face-alignment",
    "decord",
    "soundfile",
    "python-dotenv>=1.0.0",
    "psutil>=5.9.0",
    "requests>=2.31.0",
    "audiocraft",
    "sentencepiece",
    "protobuf>=3.20.0,<5.0.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.4.0",
    "pytest-asyncio>=0.23.0",
    "black>=23.0.0",
    "ruff>=0.1.0",
    "mypy>=1.5.0",
    "pre-commit>=3.5.0",
]
```

### requirements.txt
Pin all dependencies from pyproject.toml plus:
- `unsloth`, `bitsandbytes`
- `paramiko` (for session_controller SSH)
- `torchaudio>=2.5.1`

### vga/bootstrap.py
Initialize ALL singleton instances in order (steps 6A through 6Z-z):
```python
"""
VGA Bootstrap — Singleton initialization sequence.
Called once at FastAPI startup (RULE-85).
Order matters — dependencies must be initialized before dependents.
"""
def initialize_all_singletons() -> dict:
    """
    v17.0 singletons (6Z-p through 6Z-z) — initialize in order:
    - SceneCompositionAgent (6Z-p)
    - TemporalBufferManager (6Z-q)
    - SVIScheduler factory (6Z-r)
    - MotionStateTracker (6Z-s)
    - TemporalRetryController (6Z-t)
    - IdentityStateTracker (6Z-u)
    - AudioQualityValidator (6Z-v)
    - CrossModalAlignmentValidator (6Z-w)
    - CompositionPlanValidator (6Z-x)
    - HRGController (11 checkpoints) (6Z-y)
    - TemporalEngine (6Z-z)
    """
    ...
```

### Makefile targets:
- `make install` — install dependencies
- `make test` — run pytest
- `make lint` — run ruff + mypy
- `make check-arch` — run architecture linter
- `make dev` — start API + Streamlit in dev mode
- `make download-models` — run download_all_models.sh

### .pre-commit-config.yaml
Configure: ruff, black, mypy, and architecture linter hook

## Acceptance Criteria
- [ ] All `__init__.py` files created for every package
- [ ] `python -c "import vga"` succeeds
- [ ] `pytest` discovers test directory
- [ ] `make lint` runs without import errors
