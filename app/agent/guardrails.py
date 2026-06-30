import re
from typing import List, Optional
from pydantic import BaseModel
from app.schemas import ChatMessage
from app.agent.slot_extraction import SlotExtractor

class GuardrailResult(BaseModel):
    blocked: bool
    reason: Optional[str] = None

def check_guardrails(messages: List[ChatMessage]) -> GuardrailResult:
    """
    Performs safety, scope, and prompt injection checks on user inputs.
    Uses a two-layer approach:
    1. Fast deterministic checks (regex & keywords)
    2. LLM-based classification fallback (via user_intent)
    """
    if not messages:
        return GuardrailResult(blocked=False)

    # Combine user message history
    user_content = " ".join([m.content.lower() for m in messages if m.role == "user"])
    latest_user_content = next((m.content.lower() for m in reversed(messages) if m.role == "user"), "")

    # Layer 1: Fast deterministic checks
    # A. Injection patterns
    injection_patterns = [
        r"ignore\s+(?:your\s+)?(?:previous\s+)?instructions",
        r"ignore.*instructions",
        r"you\s+are\s+now",
        r"system\s+prompt",
        r"reveal\s+(?:your\s+)?prompt",
        r"act\s+as",
        r"bypass\s+restrictions",
        r"ignore\s+rules",
        r"system\s+override"
    ]
    for pattern in injection_patterns:
        if re.search(pattern, user_content) or re.search(pattern, latest_user_content):
            return GuardrailResult(
                blocked=True,
                reason="I'm sorry, but I can only help you with questions related to SHL assessment recommendations and candidate hiring. Let me know if you need help finding a test!"
            )

    # B. Out-of-scope domains
    out_of_scope_keywords = [
        "salary negotiation", "negotiate my salary",
        "medical advice", "clinical diagnosis", "diagnose",
        "legal advice", "sue my employer", "wrongful termination", "lawsuit", "lawyer"
    ]
    
    # Check if the query is a borderline-but-legitimate query about assessments
    is_about_assessments = any(word in user_content for word in ["assessment", "test", "measure", "evaluat", "screen", "hire", "recruitment", "bias"])
    
    for kw in out_of_scope_keywords:
        if kw in user_content:
            # If it's a critical legal/medical term, or not related to assessments at all, block it
            if not is_about_assessments or kw in ["sue my employer", "wrongful termination", "lawsuit", "medical advice"]:
                return GuardrailResult(
                    blocked=True,
                    reason="I'm sorry, but I cannot assist with that topic. I can only help you select and recommend SHL assessments for hiring and development."
                )

    # Layer 2: LLM-based classification fallback
    extractor = SlotExtractor()
    state = extractor.extract_slots(messages)
    
    if state.user_intent in ("refusal_needed", "off_topic"):
        reason = "I'm sorry, but I can only help you with questions related to SHL assessment recommendations and candidate hiring. Let me know if you need help finding a test!"
        if state.user_intent == "off_topic":
            reason = "I am here to help you recommend SHL assessments for your hiring needs. Please let me know what roles or skills you are looking to test!"
        return GuardrailResult(
            blocked=True,
            reason=reason
        )

    return GuardrailResult(blocked=False)

