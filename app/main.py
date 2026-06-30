import time
import logging
import os
from typing import List, Any
from fastapi import FastAPI, status
from fastapi.responses import FileResponse
from pydantic import BaseModel
from app.schemas import ChatRequest, ChatResponse, Recommendation
from app.agent.guardrails import check_guardrails
from app.agent.slot_extraction import SlotExtractor, ConversationState
from app.agent.state_machine import decide_action, Action, AgentStateMachine
from app.agent.query_builder import build_query
from app.agent.prompts import REFUSAL_PROMPT, CLARIFY_PROMPT, RECOMMEND_PROMPT, COMPARE_PROMPT

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("agent")

app = FastAPI(
    title="SHL Assessment Recommendation Agent",
    description="FastAPI service for recommending SHL assessments based on candidate/role requirements.",
    version="0.1.0"
)

# Instantiate the state machine globally
state_machine = AgentStateMachine()

@app.on_event("startup")
def startup_event():
    """
    Load the catalog and warm up the search index on startup.
    This ensures cold starts are resolved before serving requests.
    """
    logger.info("Initializing search index and warming up models...")
    try:
        index = state_machine.get_index()
        # Trigger a dummy search to load the SentenceTransformer and FAISS index
        index.search("warmup", top_k=1)
        from app.catalog.loader import load_catalog
        catalog = load_catalog()
        logger.info(f"Startup complete: Loaded {len(catalog)} catalog entries and warmed up indices.")
    except Exception as e:
        logger.exception(f"Error during startup warm-up: {str(e)}")

@app.get("/health", status_code=status.HTTP_200_OK)
def health_check():
    """
    Health check endpoint. Must respond even on cold start with 200 OK.
    """
    return {"status": "ok"}

@app.get("/")
def read_index():
    """
    Serve the interactive web UI at the root path.
    """
    static_file_path = os.path.join(os.path.dirname(__file__), "static", "index.html")
    return FileResponse(static_file_path)



@app.post("/chat", response_model=ChatResponse, status_code=status.HTTP_200_OK)
def chat_endpoint(request: ChatRequest):
    """
    Stateless chat endpoint that validates the request, runs guardrails,
    extracts slots, decides action, and returns recommendations.
    """
    start_time = time.time()
    turn_count = len([m for m in request.messages if m.role == "user"])
    
    try:
        # 1. Run guardrails check
        guardrail_res = check_guardrails(request.messages)
        if guardrail_res.blocked:
            reply = guardrail_res.reason or "I'm sorry, but I can only assist with SHL assessment recommendations."
            if state_machine.llm_client.api_key:
                llm_reply = state_machine.llm_client.generate_response(
                    system_instruction=REFUSAL_PROMPT,
                    prompt="Generate a polite refusal based on the user's out-of-scope input.",
                    temperature=0.3
                )
                if "error" not in llm_reply.lower() and "api key" not in llm_reply.lower():
                    reply = llm_reply
            
            latency = time.time() - start_time
            logger.info(f"Action: REFUSE (Guardrails) | Turns: {turn_count} | Latency: {latency:.4f}s")
            
            return ChatResponse(
                reply=reply,
                recommendations=[],
                end_of_conversation=False
            )

        # 2. Run slot extraction to get ConversationState (skip if guardrails blocked)
        state = state_machine.slot_extractor.extract_slots(request.messages)
        
        # 3. Run state_machine.decide_action
        action = decide_action(state, turn_count, request.messages)
        
        reply = ""
        recommendations = []
        end_of_conversation = False

        # 4. Branch on decided action
        if action == Action.REFUSE:
            reply = state_machine._get_refusal_reply(state)
            if state_machine.llm_client.api_key:
                llm_reply = state_machine.llm_client.generate_response(
                    system_instruction=REFUSAL_PROMPT,
                    prompt="Generate a polite refusal.",
                    temperature=0.3
                )
                if "error" not in llm_reply.lower() and "api key" not in llm_reply.lower():
                    reply = llm_reply

        elif action == Action.CLARIFY:
            reply = "To help me recommend the most suitable SHL assessments, could you please tell me what job role or specific skills you are looking to test?"
            if state.role_title:
                reply = f"I see you are looking for assessments for a {state.role_title} role. Could you tell me what specific skills (e.g. Python, Excel) you want to test?"
            
            if state_machine.llm_client.api_key:
                transcript = "\n".join([f"{m.role}: {m.content}" for m in request.messages])
                llm_reply = state_machine.llm_client.generate_response(
                    system_instruction=CLARIFY_PROMPT,
                    prompt=f"Conversation history:\n{transcript}",
                    temperature=0.4
                )
                if "error" not in llm_reply.lower() and "api key" not in llm_reply.lower():
                    reply = llm_reply

        elif action == Action.COMPARE:
            index = state_machine.get_index()
            matching_items = []
            
            # Resolve targets (fuzzy-match names against catalog)
            from app.catalog.loader import load_catalog
            catalog = load_catalog()
            
            resolved_names = []
            compare_details = ""
            
            for target in state.compare_targets:
                target_lower = target.lower()
                matched_item = None
                for item in catalog:
                    if target_lower == item.name.lower() or target_lower in item.name.lower() or item.name.lower() in target_lower:
                        matched_item = item
                        break
                if matched_item:
                    matching_items.append(matched_item)
                    resolved_names.append(matched_item.name)
                    compare_details += f"- Name: {matched_item.name}\n  Type: {matched_item.test_type}\n  Description: {matched_item.description}\n  Duration: {matched_item.duration}\n\n"
            
            if matching_items:
                reply = f"Comparing {', '.join(resolved_names)}:\n\n" + \
                        "\n".join([f"- **{item.name}** ({item.test_type}): {item.description}" for item in matching_items])
                
                if state_machine.llm_client.api_key:
                    llm_reply = state_machine.llm_client.generate_response(
                        system_instruction=COMPARE_PROMPT.format(compare_details=compare_details),
                        prompt=f"Compare these assessments: {', '.join(resolved_names)}",
                        temperature=0.3
                    )
                    if "error" not in llm_reply.lower() and "api key" not in llm_reply.lower():
                        reply = llm_reply
            else:
                reply = "I couldn't find the requested assessments in the catalog to compare."

        elif action in (Action.RECOMMEND, Action.REFINE):
            # Query builder
            query_string, included_types, excluded_types = build_query(state)
            
            # Search index (retrieve ~20 candidates)
            index = state_machine.get_index()
            job_levels = [state.seniority] if state.seniority else None
            
            # Extract languages
            languages = []
            lower_transcript = " ".join([m.content.lower() for m in request.messages])
            known_languages = ["french", "spanish", "german", "italian", "dutch", "chinese", "english"]
            for lang in known_languages:
                if lang in lower_transcript:
                    languages.append(lang.capitalize())
            if not languages:
                languages = None
                
            candidates = index.search(
                query=query_string,
                top_k=20,
                job_levels=job_levels,
                languages=languages
            )
            
            # Apply duration and exclusion filters
            filtered_candidates = []
            for item in candidates:
                if excluded_types:
                    if item.test_type.lower() in {t.lower() for t in excluded_types}:
                        continue
                if state.duration_constraint_minutes:
                    dur_val = state_machine._parse_duration_minutes(item.duration)
                    if dur_val is not None and dur_val > state.duration_constraint_minutes:
                        continue
                filtered_candidates.append(item)
                
            if not filtered_candidates:
                filtered_candidates = candidates[:5]

            # LLM selects final 1-10 from ONLY those candidates (with Pydantic structure)
            class RecommendationSelection(BaseModel):
                selected_ids: List[str]
                reply: str

            selected_items = []
            reply = ""
            
            if state_machine.llm_client.api_key and filtered_candidates:
                candidates_text = ""
                candidate_map = {}
                for item in filtered_candidates:
                    candidates_text += f"- ID: {item.entity_id} | Name: {item.name} | Type: {item.test_type} | Description: {item.description}\n"
                    candidate_map[item.entity_id] = item
                
                try:
                    selection = state_machine.llm_client.complete_json(
                        system_prompt=RECOMMEND_PROMPT.format(candidates=candidates_text),
                        user_prompt="Select the best assessments for the user's requirements.",
                        schema=RecommendationSelection
                    )
                    
                    # Map selected ids back to full catalog records (anti-hallucination safety net)
                    for eid in selection.selected_ids:
                        if eid in candidate_map:
                            selected_items.append(candidate_map[eid])
                        else:
                            logger.warning(f"LLM hallucinated candidate ID: {eid}")
                    
                    reply = selection.reply
                except Exception as e:
                    logger.error(f"Structured recommendation selection failed: {str(e)}")
                    selected_items = filtered_candidates[:5]
            else:
                selected_items = filtered_candidates[:5]
                
            if not reply:
                reply = "Based on your requirements, I recommend the following SHL assessments:\n\n"
                for item in selected_items:
                    reply += f"- **{item.name}** ({item.test_type}): {item.description or 'No description available.'} ({item.duration or 'Approx. completion time varies'})\n"
                
                missing = []
                if not state.seniority:
                    missing.append("job level (e.g., Graduate, Mid-Professional)")
                if missing and turn_count < 7:
                    reply += f"\nTo help me narrow this down, could you also let me know the {' or '.join(missing)}?"

            # Map to Recommendation schemas
            recommendations = [
                Recommendation(name=item.name, url=str(item.url), test_type=item.test_type)
                for item in selected_items
            ]
            
            # Turn 7 annotation
            if turn_count >= 7:
                reply += "\n\n*(Note: This is a best-effort shortlist based on the partial information provided.)*"

            # Heuristic for end of conversation:
            # We set end_of_conversation to True after a successful recommend/refine response
            # unless the user's latest turn explicitly suggests they want to keep refining.
            latest_user_turn = next((m.content.lower() for m in reversed(request.messages) if m.role == "user"), "")
            refine_keywords = ["actually", "change", "instead", "exclude", "add", "prefer", "make it", "but", "however"]
            wants_refining = any(kw in latest_user_turn for kw in refine_keywords)
            
            if recommendations and not wants_refining:
                end_of_conversation = True

        latency = time.time() - start_time
        logger.info(f"Action: {action} | Turns: {turn_count} | Latency: {latency:.4f}s")
        
        return ChatResponse(
            reply=reply,
            recommendations=recommendations,
            end_of_conversation=end_of_conversation
        )
        
    except Exception as e:
        logger.exception(f"Internal error in chat endpoint: {str(e)}")
        # Safe fallback ChatResponse on any internal error (never 500)
        return ChatResponse(
            reply="I'm sorry, I encountered an unexpected error while processing your request. Please try again or ask me about SHL assessments.",
            recommendations=[],
            end_of_conversation=False
        )


