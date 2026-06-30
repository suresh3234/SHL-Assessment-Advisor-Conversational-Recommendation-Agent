import os
import json
from typing import Type, TypeVar
import httpx
from pydantic import BaseModel

T = TypeVar('T', bound=BaseModel)

class LLMClient:
    """
    A thin wrapper around the Groq and OpenRouter APIs.
    """
    def __init__(self):
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        self.openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
        
        if self.groq_api_key:
            self.api_key = self.groq_api_key
            self.base_url = "https://api.groq.com/openai/v1/chat/completions"
            self.model = "mixtral-8x7b-32768"
            self.headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
        elif self.openrouter_api_key:
            self.api_key = self.openrouter_api_key
            self.base_url = "https://openrouter.ai/api/v1/chat/completions"
            self.model = "meta-llama/llama-3.1-8b-instruct:free"
            self.headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/google/antigravity",
                "X-Title": "SHL Assessment Recommendation Agent"
            }
        else:
            self.api_key = None
            self.base_url = None
            self.model = None
            self.headers = {}

    def complete_json(self, system_prompt: str, user_prompt: str, schema: Type[T]) -> T:
        """
        Requests JSON-mode/structured output from the LLM and parses + validates it
        against the given Pydantic model. Retries once on parse/validation failure.
        """
        if not self.api_key:
            raise ValueError("No API key found. Please set GROQ_API_KEY or OPENROUTER_API_KEY in your environment.")

        # Inject schema instructions into system prompt
        full_system_prompt = (
            f"{system_prompt}\n\n"
            f"You MUST respond with a JSON object that strictly conforms to this JSON Schema:\n"
            f"{json.dumps(schema.model_json_schema())}\n\n"
            f"Do not include any explanation, markdown formatting outside of a valid JSON object, or extra text."
        )

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": full_system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.0
        }

        last_exception = None
        for attempt in range(2):  # 1 initial try + 1 retry = 2 attempts
            try:
                response = httpx.post(
                    self.base_url,
                    headers=self.headers,
                    json=payload,
                    timeout=30.0
                )
                response.raise_for_status()
                response_data = response.json()
                
                content = response_data["choices"][0]["message"]["content"]
                
                # Parse and validate
                parsed_json = json.loads(content)
                return schema.model_validate(parsed_json)
                
            except (httpx.HTTPStatusError, httpx.RequestError, json.JSONDecodeError, KeyError, Exception) as e:
                last_exception = e
                continue

        raise ValueError(f"Failed to obtain valid JSON matching schema after 2 attempts. Last error: {str(last_exception)}")

    def generate_response(self, system_instruction: str, prompt: str, temperature: float = 0.0) -> str:
        """
        Fallback simple text generation method.
        """
        if not self.api_key:
            return "Mock Response: API Key is not configured."
            
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": prompt}
            ],
            "temperature": temperature
        }
        
        try:
            response = httpx.post(
                self.base_url,
                headers=self.headers,
                json=payload,
                timeout=30.0
            )
            response.raise_for_status()
            response_data = response.json()
            return response_data["choices"][0]["message"]["content"]
        except Exception as e:
            return f"Error calling LLM: {str(e)}"


