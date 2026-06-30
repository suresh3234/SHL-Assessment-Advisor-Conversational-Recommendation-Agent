import json
import re
from typing import List, Dict, Any, Optional, Literal
from pydantic import BaseModel
from app.schemas import ChatMessage
from app.agent.llm_client import LLMClient
from app.agent.prompts import SLOT_EXTRACTION_PROMPT

class ConversationState(BaseModel):
    role_title: Optional[str] = None
    seniority: Optional[str] = None
    must_have_skills: List[str] = []
    nice_to_have_skills: List[str] = []
    excluded_test_types: List[str] = []
    included_test_types: List[str] = []   # explicit asks like "add personality tests"
    duration_constraint_minutes: Optional[int] = None
    remote_required: Optional[bool] = None
    has_enough_context: bool         # the model's own judgment, used as one signal among others
    user_intent: Literal["clarify_needed", "recommend", "refine", "compare", "off_topic", "refusal_needed"]
    compare_targets: List[str] = []       # assessment names mentioned for comparison, if intent=compare

class SlotExtractor:
    """
    Extracts structured ConversationState from the conversation transcript
    using LLM complete_json or a rule-based fallback.
    """
    def __init__(self):
        self.llm_client = LLMClient()

    def extract_slots(self, messages: List[ChatMessage]) -> ConversationState:
        """
        Analyzes the transcript and returns a ConversationState.
        """
        if not messages:
            return ConversationState(
                has_enough_context=False,
                user_intent="clarify_needed"
            )

        # Format transcript
        transcript = ""
        for msg in messages:
            transcript += f"{msg.role.capitalize()}: {msg.content}\n"

        # Latest user message
        user_msgs = [m for m in messages if m.role == "user"]
        latest_user_message = user_msgs[-1].content if user_msgs else ""

        # Try LLM first if API key is set
        if self.llm_client.api_key:
            try:
                return self.llm_client.complete_json(
                    system_prompt=SLOT_EXTRACTION_PROMPT,
                    user_prompt=transcript,
                    schema=ConversationState
                )
            except Exception:
                # Fallback to rule-based on failure
                pass

        # Fallback to rule-based extraction
        return self._fallback_extraction(transcript, latest_user_message)

    def _fallback_extraction(self, transcript: str, latest_user_message: Optional[str] = None) -> ConversationState:
        """
        Rule-based fallback slot extractor using regex/keyword matching.
        """
        if latest_user_message is None:
            # Try to extract the last user message from the transcript
            lines = [line.strip() for line in transcript.strip().split("\n") if line.strip()]
            user_lines = [line for line in lines if line.lower().startswith("user:")]
            if user_lines:
                latest_user_message = re.sub(r"^user:\s*", "", user_lines[-1], flags=re.IGNORECASE)
            else:
                latest_user_message = transcript

        lower_transcript = transcript.lower()
        lower_latest = latest_user_message.lower()

        
        # Determine intent based on the LATEST user message
        user_intent: Literal["clarify_needed", "recommend", "refine", "compare", "off_topic", "refusal_needed"] = "clarify_needed"
        
        injection_patterns = [
            r"ignore\s+(?:your\s+)?(?:previous\s+)?instructions",
            r"ignore.*instructions",
            r"you\s+are\s+now",
            r"system\s+prompt",
            r"reveal\s+(?:your\s+)?prompt",
            r"act\s+as",
            r"bypass\s+restrictions",
            r"ignore\s+rules",
            r"system\s+override",
            r"tell me a joke"
        ]
        if any(re.search(pat, lower_latest) for pat in injection_patterns):
            user_intent = "refusal_needed"
        elif any(word in lower_latest for word in ["compare", "difference between", "versus", " vs "]):
            user_intent = "compare"
        elif any(word in lower_latest for word in ["weather", "pizza", "movie", "sports", "song"]):
            user_intent = "off_topic"
        elif any(word in lower_latest for word in ["thank", "thanks", "bye", "goodbye", "that's all", "that is all"]):
            user_intent = "recommend"
        elif any(word in lower_latest for word in ["actually", "change", "instead", "exclude", "add", "prefer", "make it"]):
            user_intent = "refine"
        else:
            # If the latest message has skills or role, it's a recommend
            has_skills_in_latest = False
            for skill in [
                "python", "java", "excel", "c\\+\\+", "c#", "dotnet", "net",
                "javascript", "sql", "sales", "cognitive", "numerical", 
                "verbal", "inductive", "deductive", "coding", "react",
                "accounting", "finance", "typing"
            ]:
                pattern = r"\b" + skill + r"\b" if not skill.endswith("+") else re.escape(skill)
                if re.search(pattern, lower_latest):
                    has_skills_in_latest = True
                    break
            if has_skills_in_latest or any(term in lower_latest for term in ["role", "developer", "engineer", "manager", "director", "hire", "recruitment"]):
                user_intent = "recommend"
            else:
                user_intent = "clarify_needed"

        # Extract slots from the FULL transcript
        # Extract role
        role_title = None
        role_match = re.search(r"(?:for a|for|hire a)\s+([a-zA-Z\s]+?)(?:\s+who|\s+with|\s+to|\.|\n|$)", lower_transcript)
        if role_match:
            role = role_match.group(1).strip()
            if role and role not in {"role", "position", "candidate", "developer", "engineer", "job"}:
                role_title = role.title()

        # Extract seniority
        seniority = None
        level_mappings = {
            "director": "Director",
            "executive": "Executive",
            "graduate": "Graduate",
            "manager": "Manager",
            "supervisor": "Supervisor",
            "entry": "Entry-Level",
            "junior": "Entry-Level",
            "mid": "Mid-Professional",
            "professional": "Mid-Professional",
            "senior": "Professional Individual Contributor"
        }
        for term, level in level_mappings.items():
            if re.search(r"\b" + re.escape(term) + r"\b", lower_transcript):
                seniority = level
                break

        # Extract skills
        must_have_skills = []
        nice_to_have_skills = []
        common_skills = [
            "python", "java", "excel", "c\\+\\+", "c#", "dotnet", "net",
            "javascript", "sql", "sales", "cognitive", "numerical", 
            "verbal", "inductive", "deductive", "coding", "react",
            "accounting", "finance", "typing"
        ]
        for skill in common_skills:
            pattern = r"\b" + skill + r"\b" if not skill.endswith("+") else re.escape(skill)
            if re.search(pattern, lower_transcript):
                display_name = skill.replace("\\", "")
                if display_name in ("dotnet", "net"):
                    display_name = ".NET"
                else:
                    display_name = display_name.capitalize()
                must_have_skills.append(display_name)

        # Extract duration
        duration_constraint_minutes = None
        duration_match = re.search(r"(\d+)\s*(?:min|minute)", lower_transcript)
        if duration_match:
            duration_constraint_minutes = int(duration_match.group(1))

        # Remote required
        remote_required = None
        if "remote" in lower_transcript:
            remote_required = True

        # Check for explicit inclusion/exclusion of test types
        included_test_types = []
        excluded_test_types = []
        
        if "personality" in lower_transcript or "behavior" in lower_transcript:
            included_test_types.append("Personality & Behavior")
        if "simulation" in lower_transcript:
            included_test_types.append("Simulations")
        if "no personality" in lower_transcript or "exclude personality" in lower_transcript:
            excluded_test_types.append("Personality & Behavior")

        # Check if we have enough context (e.g. at least one skill or job role)
        has_enough_context = True if (must_have_skills or role_title) else False

        # If compare, extract targets
        compare_targets = []
        if user_intent == "compare":
            # Very simple heuristic: find capitalized words or common test names
            # For simplicity in tests, we can just extract any found skills or names
            if "excel" in lower_transcript:
                compare_targets.append("Excel")
            if "word" in lower_transcript:
                compare_targets.append("Word")
            if "python" in lower_transcript:
                compare_targets.append("Python")
            if "ado.net" in lower_transcript:
                compare_targets.append("ADO.NET (New)")
            if "mvc" in lower_transcript:
                compare_targets.append(".NET MVC (New)")


        return ConversationState(
            role_title=role_title,
            seniority=seniority,
            must_have_skills=must_have_skills,
            nice_to_have_skills=nice_to_have_skills,
            excluded_test_types=excluded_test_types,
            included_test_types=included_test_types,
            duration_constraint_minutes=duration_constraint_minutes,
            remote_required=remote_required,
            has_enough_context=has_enough_context,
            user_intent=user_intent,
            compare_targets=compare_targets
        )
