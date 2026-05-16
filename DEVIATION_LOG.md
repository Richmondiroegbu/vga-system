# VGA v17.2 — Deviation Log

Any deviation from a RULE-XX or FR-XXX constraint MUST be documented here with:
- Rule/requirement violated
- Reason for deviation
- Operator approval status

---

## Format

```
## DEV-NNN — [DATE] — [RULE/FR violated]
**File:** path/to/file.py  
**Rule violated:** RULE-XX / FR-XXX  
**Description:** What was deviated  
**Reason:** Why the deviation was necessary  
**Approval:** [PENDING / APPROVED by <operator> on <date>]  
```

---

## DEV-001 — 2026-05-14 — MMAudio model variant downgrade

**File:** `vga/models/wrappers/mmaudio_wrapper.py`, `system_files/bootstrap_pipeline.py`
**Rule violated:** MASTER_PROMPT_INDEX.md §2 Technology Stack (specifies `large_44k_v2`)
**Description:** MMAudio model switched from `large_44k_v2` (4.12 GB) to `medium_44k` (2.49 GB). The full repo (54.7 GB) is no longer downloaded; only `weights/mmaudio_medium_44k.pth` + `ext_weights/` (~5.8 GB total) are downloaded.
**Reason:** Verified disk size constraints. Pod has 250 GB disk. Full MMAudio repo is 54.7 GB including 39.3 GB of training checkpoints irrelevant to inference. Medium model provides adequate quality for ambient audio mixed at −12 dB. User-approved decision to save ~48.9 GB disk space.
**Approval:** APPROVED by operator on 2026-05-14

---

## DEV-002 — 2026-05-14 — AVON watcher placement

**File:** `vga/client/avon_watcher.py`
**Rule violated:** Prompt 18 specifies `vga_client_watcher.py` as a root-level standalone file
**Description:** AVON watcher implemented at `vga/client/avon_watcher.py` (inside the vga package) rather than `vga_client_watcher.py` at project root.
**Reason:** Cleaner package organisation. The watcher is still fully usable as `python -m vga.client.avon_watcher --job-id <id> --server <url>`. The CLI entry point is identical to the standalone script approach.
**Approval:** APPROVED by operator on 2026-05-14

---

## DEV-004 — 2026-05-16 — PyTorch upgraded to 2.8.0+cu128 for RTX 5090

**Files:** `system_files/bootstrap_pipeline.py`, `vga/config/settings.py`, `controller_config.json`
**Rule violated:** MASTER_PROMPT_INDEX.md §2 Technology Stack (specifies PyTorch ≥ 2.5.1, CUDA 12.4 cu124)
**Description:** Switched GPU from RTX 4090 to RTX 5090 (Blackwell architecture, sm_100). This requires PyTorch ≥ 2.8.0 and CUDA 12.8 (cu128). Changes made: PYTORCH_INDEX cu124→cu128, min version 2.5.1→2.8.0, SVI env PyTorch 2.7.1→2.8.0, RunPod template updated. System RAM on 5090 pod is 41 GB (meets the 32 GB minimum; 4090 pod only had 29 GB which was below minimum).
**Reason:** RTX 4090 pod configuration had only 29 GB RAM — below the 32 GB minimum required to load Wan2.2-I2V-A14B-FP8 (needs 25–30 GB free RAM). RTX 5090 pod provides 41 GB RAM and 32 GB VRAM (vs 4090's 24 GB VRAM), resolving both the RAM constraint and providing more VRAM headroom for LatentSync (which uses 17–23 GB VRAM).
**Approval:** APPROVED by operator on 2026-05-16

---

## DEV-003 — 2026-05-14 — image_edit_agent.py combines S-06A/B/C

**File:** `vga/agents/image_edit_agent.py`
**Rule violated:** VGA Codebase Structure Design §agents — specifies `multi_angle_agent.py`, `image_merge_agent.py`, `scene_expansion_agent.py` as separate files
**Description:** Initial implementation combined all 3 sub-stages inside `image_edit_agent.py`. Corrected by implementing all three as separate files per spec. `image_edit_agent.py` retained as the coordinator but the sub-agents now exist as individual files.
**Reason:** Implementation oversight. Now corrected — all three separate agent files created on 2026-05-15.
**Approval:** APPROVED by operator on 2026-05-15 (corrected)
