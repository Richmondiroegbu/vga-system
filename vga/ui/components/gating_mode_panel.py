"""Gating mode panel (v15.0). Shows and allows switching GatingMode."""
from __future__ import annotations

from vga.models.enums import GatingMode


def render(api_base: str = "http://localhost:8000/api/v1") -> None:
    import streamlit as st

    st.subheader("🔒 Gating Mode")

    mode = st.selectbox("Active Mode", [m.value for m in GatingMode], index=0)
    if mode == GatingMode.FAST.value:
        st.warning("⚠️ FAST mode: reduced validations. NOT for production.")
    elif mode == GatingMode.STRICT.value:
        st.success("✅ STRICT mode: all validations enforced. Production default.")
    else:
        st.info("BALANCED mode: standard validations.")

    st.caption("Note: CLIP ≥ 0.93 and CompositionPlan gates are NEVER bypassed regardless of mode.")
