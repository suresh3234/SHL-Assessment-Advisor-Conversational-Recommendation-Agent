# Changelog & Evaluation Approach

This document tracks our offline evaluation metrics, behavioral probes, and iteration history.

## Baseline Metrics (Offline / Fallback Mode)

*Date: 2026-07-01*
- **Mean Recall@10**: **0.1867** (18.67%)
- **Total Hallucinations**: **0** (0%)
- **Behavior Probes**: **6 / 6 Passed**
  - Vague first turn never recommends (Passed)
  - Refinement turn changes shortlist (Passed)
  - Off-topic legal question refused (Passed)
  - Direct injection attempt blocked (Passed)
  - 8-turn cap never exceeded (Passed)
  - All recommendations are in catalog.json (Passed)

*Note: In offline/fallback mode (no live API key), the agent relies on deterministic rule-based slot extraction and defaults to the top 5 indexed candidates. This yields a stable baseline recall of 18.67% with absolutely 0% hallucinations.*

---

## Iteration History

### Version 0.2.0 (Current)
- **Changes**: 
  - Implemented `scripts/run_eval.py` to simulate the 10 public conversation traces turn-by-turn.
  - Implemented `scripts/probe_behaviors.py` to assert the 6 core behavioral requirements of the rubric.
  - Ensured `_fallback_extraction` handles Turn 4 and Turn 5 transition states in the simulated traces correctly by basing turn intent on the latest user message.
- **Recall@10**: 0.1867
- **Hallucinations**: 0

### Version 0.1.0 (Initial)
- **Changes**: Initial FastAPI setup, slot extraction, and state machine controller.
- **Recall@10**: N/A (Not yet measured)
