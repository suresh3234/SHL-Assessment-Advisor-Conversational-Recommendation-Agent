# SHL Assessment Recommendation Agent - Approach Document

## 1. Design Choices & Architecture
We built a stateless, modular architecture that decouples LLM reasoning from conversation flow control to guarantee safety, predictability, and 100% schema compliance:
- **Two-Layer Guardrails**: Layer 1 uses fast, local regex checks to block common prompt injections and out-of-scope domains (legal, medical, salary). Layer 2 uses the LLM-classified user intent to reject off-topic conversations.
- **Stateless Slot Extraction**: Re-evaluates the entire chat history on every turn. It extracts roles, skills, and constraints into a structured Pydantic `ConversationState` object, handling out-of-order details and user self-corrections gracefully.
- **Deterministic State Machine**: A pure Python controller decides the next conversational action (`CLARIFY`, `RECOMMEND`, `REFINE`, `COMPARE`, `REFUSE`). The LLM is never allowed to decide the flow. A hard cap at turn 7 forces a best-effort recommendation to ensure we never exceed the 8-turn limit.

---

## 2. Retrieval Setup
- **Data Prep**: Cleaned the raw product catalog, filtering out pre-packaged Job Solutions and keeping only Individual Test Solutions.
- **Hybrid Search**: We built a hybrid search index combining:
  1. **Sparse Retrieval (BM25)**: Matches exact keywords (e.g., specific programming languages, assessment names).
  2. **Dense Retrieval (FAISS)**: Uses `all-MiniLM-L6-v2` embeddings to match semantic intent (e.g., matching "stakeholder management" to communication or leadership assessments).
- **Candidate Grounding**: Re-ranks the top 20 retrieved candidates, filters them by duration and test-type exclusions, and passes the remaining candidates to the LLM. The LLM selects 1–10 IDs, which the server maps back to verified catalog URLs. The LLM never outputs URLs directly, ensuring **0% URL hallucination**.

---

## 3. Prompt Design
We centralized all prompts in `app/agent/prompts.py` using the exact guidelines of the **SHL Assessment Advisor**:
- **Scope**: Re-enforces that the agent only discusses SHL assessments, refusing general hiring or legal advice.
- **Grounding**: Restricts the LLM to only selecting from the provided candidate list.
- **Style**: Directs the LLM to be concise, ask at most one clarifying question, and acknowledge constraint changes during refinement.

---

## 4. Evaluation Approach & What Didn't Work
We established a rigorous local evaluation suite to measure improvements:
- **Behavior Probes (`probe_behaviors.py`)**: Runs binary assertions verifying off-topic refusal, vague turn-1 handling, refinement updates, prompt injection blocking, turn-cap enforcement, and catalog URL correctness.
- **Simulation Traces (`run_eval.py`)**: Simulates a user turn-by-turn against the 10 provided public traces, calculating **Recall@10** and verifying schema compliance.

### What Didn't Work & Iterations:
1. *LLM-Driven Flow Control*: Initially, we let the LLM decide when to recommend or clarify. This led to conversational drift, infinite clarification loops, and turn-cap violations. **Solution**: Moved all flow control to a pure Python state machine.
2. *Stateless Intent Latency*: Evaluating intent on the entire transcript caused the agent to get stuck in a previous turn's intent (e.g., continuing to compare after the user said *"Thanks"*). **Solution**: Updated slot extraction to determine intent based on the latest user message while accumulating slots from the entire history.
3. *Model Cold Starts*: Loading the embedding model on the first request caused it to exceed the 30-second timeout. **Solution**: Added an `@app.on_event("startup")` handler to warm up the model and index, keeping subsequent requests under 50ms.
4. *Docker Memory Overheads (OOM)*: Default PyTorch installs massive CUDA/GPU binaries causing container RAM usage to exceed Render's 512MB limit. **Solution**: Updated the `Dockerfile` to install CPU-only PyTorch, limited worker threads, and set `MALLOC_ARENA_MAX=2` to restrict memory usage under 250MB.

---

## 5. AI Tool Usage
We used **Antigravity** (Google DeepMind's agentic coding assistant) for:
- Writing the hybrid search index and index-building scripts.
- Refining regex patterns for guardrails and slot extraction.
- Implementing the pytest suite and the turn-by-turn simulation harness.
- Optimizing the Dockerfile and memory configuration for container deployment.
