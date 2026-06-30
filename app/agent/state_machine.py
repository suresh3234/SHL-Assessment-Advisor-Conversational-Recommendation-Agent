import os
import re
from enum import Enum
from typing import List, Tuple, Dict, Any, Optional
from app.schemas import ChatMessage, Recommendation
from app.agent.slot_extraction import SlotExtractor, ConversationState
from app.agent.llm_client import LLMClient
from app.agent.prompts import RECOMMEND_PROMPT
from app.retrieval.index import HybridIndex

class Action(str, Enum):
    CLARIFY = "CLARIFY"
    RECOMMEND = "RECOMMEND"
    REFINE = "REFINE"
    COMPARE = "COMPARE"
    REFUSE = "REFUSE"

def decide_action(state: ConversationState, turn_count: int, history: List[ChatMessage]) -> Action:
    """
    Pure deterministic controller that decides the next action based on the state,
    turn count, and history.
    """
    # 1. If guardrails blocked -> REFUSE
    if state.user_intent in ("refusal_needed", "off_topic"):
        return Action.REFUSE

    # Fast deterministic check on history for safety
    user_content = " ".join([m.content.lower() for m in history if m.role == "user"])
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
        if re.search(pattern, user_content):
            return Action.REFUSE

    out_of_scope_keywords = [
        "salary negotiation", "negotiate my salary",
        "medical advice", "clinical diagnosis", "diagnose",
        "legal advice", "sue my employer", "wrongful termination", "lawsuit", "lawyer"
    ]
    is_about_assessments = any(word in user_content for word in ["assessment", "test", "measure", "evaluat", "screen", "hire", "recruitment", "bias"])
    for kw in out_of_scope_keywords:
        if kw in user_content:
            if not is_about_assessments or kw in ["sue my employer", "wrongful termination", "lawsuit", "medical advice"]:
                return Action.REFUSE

    # 2. If user_intent == "compare" and compare_targets has >=2 resolvable entries -> COMPARE
    if state.user_intent == "compare":
        resolvable_count = 0
        try:
            from app.catalog.loader import load_catalog
            catalog = load_catalog()
            catalog_names = {item.name.lower() for item in catalog}
            for target in state.compare_targets:
                target_lower = target.lower()
                if any(target_lower in name or name in target_lower for name in catalog_names):
                    resolvable_count += 1
        except Exception:
            # Fallback for unit tests where catalog.json might not be present/loaded
            resolvable_count = len([t for t in state.compare_targets if t])
            
        if resolvable_count >= 2:
            return Action.COMPARE

    # Scan history to see if we have recommended earlier
    has_recommended_earlier = False
    for msg in history:
        if msg.role == "assistant":
            if "recommend" in msg.content.lower() or "- **" in msg.content or "shl" in msg.content.lower():
                has_recommended_earlier = True
                break

    # 3. Hard cap: if turn_count >= 7, force RECOMMEND
    has_minimum_signal = bool(state.role_title or state.must_have_skills)
    if turn_count >= 7:
        return Action.RECOMMEND

    # 4. If we have NOT recommended yet and state lacks a minimum signal -> CLARIFY
    if not has_recommended_earlier and not has_minimum_signal:
        return Action.CLARIFY

    # 5. If we HAVE recommended earlier and the latest user turn adds/removes/refines constraints -> REFINE
    if has_recommended_earlier:
        if state.user_intent == "refine" or state.excluded_test_types or state.included_test_types or state.duration_constraint_minutes is not None or state.remote_required is not None:
            return Action.REFINE

    # 6. Else if state has enough signal -> RECOMMEND
    if has_minimum_signal:
        return Action.RECOMMEND

    return Action.CLARIFY

class AgentStateMachine:
    """
    Manages the conversational state and decides the next action.
    """
    _index: Optional[HybridIndex] = None

    def __init__(self):
        self.slot_extractor = SlotExtractor()
        self.llm_client = LLMClient()

    @classmethod
    def get_index(cls) -> HybridIndex:
        if cls._index is None:
            cls._index = HybridIndex()
            current_dir = os.path.dirname(os.path.abspath(__file__))
            index_path = os.path.join(current_dir, "..", "catalog", "index")
            if os.path.exists(index_path + ".pkl"):
                cls._index.load(index_path)
            else:
                from app.catalog.loader import load_catalog
                cls._index.build(load_catalog())
        return cls._index

    def process_turn(self, messages: List[ChatMessage]) -> Tuple[str, List[Recommendation], bool]:
        """
        Processes a conversation history and determines:
        1. The assistant's reply string.
        2. A list of recommended assessments.
        3. Whether the conversation has ended.
        """
        if not messages:
            return "Hello! I am your SHL assessment recommendation assistant. How can I help you today?", [], False

        # 1. Extract slots
        slots = self.slot_extractor.extract_slots(messages)
        
        # 2. Determine turn count
        turn_count = len([m for m in messages if m.role == "user"])
        
        # 3. Decide action
        action = decide_action(slots, turn_count, messages)
        
        # 4. Handle REFUSE immediately
        if action == Action.REFUSE:
            reply = self._get_refusal_reply(slots)
            return reply, [], False
            
        # 5. Handle CLARIFY
        if action == Action.CLARIFY:
            reply = "To help me recommend the most suitable SHL assessments, could you please tell me what job role or specific skills you are looking to test?"
            if slots.role_title:
                reply = f"I see you are looking for assessments for a {slots.role_title} role. Could you tell me what specific skills (e.g. Python, Excel) you want to test?"
            return reply, [], False

        # 6. Handle COMPARE
        if action == Action.COMPARE:
            index = self.get_index()
            matching_items = []
            for target in slots.compare_targets:
                results = index.search(target, top_k=1)
                if results:
                    matching_items.extend(results)
            
            recommendations = [
                Recommendation(name=item.name, url=str(item.url), test_type=item.test_type)
                for item in matching_items
            ]
            reply = self._generate_compare_reply(slots.compare_targets, matching_items)
            return reply, recommendations, False

        # 7. Handle RECOMMEND / REFINE
        index = self.get_index()
        
        query_parts = []
        if slots.must_have_skills:
            query_parts.extend(slots.must_have_skills)
        if slots.role_title:
            query_parts.append(slots.role_title)
            
        search_query = " ".join(query_parts) if query_parts else "assessment"

        job_levels = [slots.seniority] if slots.seniority else None
        
        # Extract languages from transcript for filtering
        languages = []
        lower_transcript = " ".join([m.content.lower() for m in messages])
        known_languages = ["french", "spanish", "german", "italian", "dutch", "chinese", "english"]
        for lang in known_languages:
            if lang in lower_transcript:
                languages.append(lang.capitalize())
        if not languages:
            languages = None

        raw_results = index.search(
            query=search_query,
            top_k=5,
            job_levels=job_levels,
            languages=languages
        )

        # Apply duration constraint
        matching_items = []
        if slots.duration_constraint_minutes:
            max_dur = slots.duration_constraint_minutes
            for item in raw_results:
                dur_val = self._parse_duration_minutes(item.duration)
                if dur_val is None or dur_val <= max_dur:
                    matching_items.append(item)
        else:
            matching_items = raw_results

        # Apply excluded_test_types
        if slots.excluded_test_types:
            excluded = {t.lower() for t in slots.excluded_test_types}
            matching_items = [item for item in matching_items if item.test_type.lower() not in excluded]

        # Convert to Recommendations
        recommendations = [
            Recommendation(name=item.name, url=str(item.url), test_type=item.test_type)
            for item in matching_items
        ]

        # Generate reply
        reply = self._generate_recommendation_reply(messages, slots, matching_items, turn_count)

        end_of_conversation = self._determine_end_of_conversation(messages)

        return reply, recommendations, end_of_conversation

    def _parse_duration_minutes(self, duration_str: Optional[str]) -> Optional[int]:
        if not duration_str:
            return None
        match = re.search(r"(\d+)", duration_str)
        if match:
            return int(match.group(1))
        return None

    def _get_refusal_reply(self, slots: ConversationState) -> str:
        if slots.user_intent == "refusal_needed":
            return "I'm sorry, but I can only help you with questions related to SHL assessment recommendations and candidate hiring. Let me know if you need help finding a test!"
        return "I am here to help you recommend SHL assessments for your hiring needs. Please let me know what roles or skills you are looking to test!"

    def _generate_compare_reply(self, targets: List[str], items: List[Any]) -> str:
        reply = f"Here is a comparison of the assessments you requested: {', '.join(targets)}:\n\n"
        if not items:
            return "I couldn't find the requested assessments in the catalog to compare."
        for item in items:
            reply += f"- **{item.name}** ({item.test_type}): {item.description or 'No description available.'} (Duration: {item.duration or 'Varies'}, Languages: {', '.join(item.languages) or 'English'})\n"
        return reply

    def _generate_recommendation_reply(self, messages: List[ChatMessage], slots: ConversationState, matching_items: List[Any], turn_count: int) -> str:
        items_summary = ""
        for item in matching_items:
            items_summary += f"- {item.name} (Type: {item.test_type}, Languages: {', '.join(item.languages)}, Duration: {item.duration or 'N/A'})\n  Description: {item.description}\n"

        forced_annotation = ""
        if turn_count >= 7:
            forced_annotation = "\n\n*(Note: This is a best-effort shortlist based on the partial information provided.)*"

        if self.llm_client.api_key:
            transcript = ""
            for msg in messages:
                transcript += f"{msg.role.capitalize()}: {msg.content}\n"

            prompt = f"""Conversation History:
{transcript}

Extracted Requirements:
- Job Role: {slots.role_title}
- Experience Level: {slots.seniority}
- Must-Have Skills: {slots.must_have_skills}
- Nice-to-Have Skills: {slots.nice_to_have_skills}
- Excluded Test Types: {slots.excluded_test_types}
- Included Test Types: {slots.included_test_types}
- Max Duration: {slots.duration_constraint_minutes} minutes
- Remote Required: {slots.remote_required}

Matching Assessments from Catalog:
{items_summary if items_summary else "No matching assessments found."}

Please write your response to the user. Keep it natural, professional, and helpful. Do not repeat the internal JSON structure.
"""
            llm_reply = self.llm_client.generate_response(
                system_instruction=RECOMMEND_PROMPT,
                prompt=prompt,
                temperature=0.3
            )
            return llm_reply + forced_annotation
        
        # Fallback rule-based response
        if matching_items:
            reply = "Based on your requirements, I recommend the following SHL assessments:\n\n"
            for item in matching_items:
                reply += f"- **{item.name}** ({item.test_type}): {item.description or 'No description available.'} ({item.duration or 'Approx. completion time varies'})\n"
            
            missing = []
            if not slots.seniority:
                missing.append("job level (e.g., Graduate, Mid-Professional)")
            if missing and turn_count < 7:
                reply += f"\nTo help me narrow this down, could you also let me know the {' or '.join(missing)}?"
            return reply + forced_annotation
        else:
            return "I couldn't find any assessments matching your exact requirements. Could you please tell me more about the job role or the specific skills you want to test?" + forced_annotation

    def _determine_end_of_conversation(self, messages: List[ChatMessage]) -> bool:
        if not messages:
            return False
        user_msgs = [m for m in messages if m.role == "user"]
        if not user_msgs:
            return False
        last_content = user_msgs[-1].content.lower()
        end_triggers = ["thank you", "thanks", "bye", "goodbye", "that's all", "that is all", "no other questions", "no, thanks"]
        return any(trigger in last_content for trigger in end_triggers)



