# Prompt 17: Session Controller (Local Machine)
**Category:** Session Management  
**File:** `session_controller.py` (root level — runs locally on Windows)  
**Spec:** See `00_BOOTSTRAP_ENVIRONMENT_PROMPT.md` for full spec

This is already specified in detail in Prompt 00. Implement the full `session_controller.py` as described there, ensuring:
- Uses `paramiko` for SSH
- Validates manifest CONTENT (not just existence)
- Injects env vars explicitly into bootstrap invocation
- 1-hour bootstrap timeout guard
- API health check with retry-safe restart loop
- `POST /system/resume` after human HRG review

## Acceptance Criteria
- [ ] `python session_controller.py` without `.env` set raises `EnvironmentError` with clear message
- [ ] Manifest validation reads AND parses JSON content
- [ ] Bootstrap invocation injects ALL env vars via explicit export statements
- [ ] Health check retries 3 times with backoff before declaring API failure
