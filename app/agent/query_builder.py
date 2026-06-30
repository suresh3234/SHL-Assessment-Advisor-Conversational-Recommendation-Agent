from typing import Tuple, List, Optional
from app.agent.slot_extraction import ConversationState

def build_query(state: ConversationState) -> Tuple[str, List[str], List[str]]:
    """
    Builds a search query string and test type filters from the ConversationState.
    """
    query_parts = []
    if state.must_have_skills:
        query_parts.extend(state.must_have_skills)
    if state.role_title:
        query_parts.append(state.role_title)
        
    query_string = " ".join(query_parts) if query_parts else "assessment"
    
    return query_string, state.included_test_types, state.excluded_test_types
