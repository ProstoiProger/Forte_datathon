"""Empirical case memory built from the labeled training videos (see scripts/build_casebase.py).

Turns per-window VLM facts into candidate windows (rules weighted by how often they fire in labeled violations vs
normal work) and picks reference examples (a confirmed violation and a similar normal situation) for the verifier.
"""

from cashwatch.casebase.rules import RULES, Rule, VideoContext
from cashwatch.casebase.store import BUNDLE_FILE, CaseBase, Model, ScoringConfig, WindowScore

__all__ = ["BUNDLE_FILE", "RULES", "CaseBase", "Model", "Rule", "ScoringConfig", "VideoContext", "WindowScore"]
