"""
LLM Service - Ollama-only simplified interface.

This file intentionally restricts usage to Ollama (local) only.
If another provider is passed, the service will log a warning and
fallback to Ollama to avoid importing unnecessary provider SDKs.
"""
import os
import json
import logging
from typing import Dict, Optional, List, Any
from enum import Enum

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LLMProvider(Enum):
    """Supported LLM providers (only Ollama used)."""
    OLLAMA = "ollama"


class LLMService:
    """
    Simplified LLM service that uses Ollama (local) exclusively.
    """

    def __init__(
        self,
        provider: str = "ollama",
        model: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        provider_lower = (provider or "ollama").lower()
        if provider_lower != LLMProvider.OLLAMA.value:
            logger.warning("Only Ollama is supported. Falling back to Ollama (was: %s).", provider)
        self.provider = LLMProvider.OLLAMA
        self.model = model or "mistral"
        self.base_url = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.client = self._initialize_client()
        logger.info("LLM service initialized: %s / %s", self.provider.value, self.model)

    def _initialize_client(self):
        # Ollama client: lightweight requests session
        import requests
        s = requests.Session()
        return s

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        json_mode: bool = False
    ) -> Dict[str, Any]:
        """Generate response using Ollama only."""
        return self._generate_ollama(system_prompt, user_prompt, temperature, max_tokens, json_mode)

    def _generate_ollama(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_tokens: int,
        json_mode: bool
    ) -> Dict[str, Any]:
        url = f"{self.base_url}/api/generate"
        full_prompt = f"{system_prompt}\n\n{user_prompt}"
        if json_mode:
            full_prompt += "\n\nRespond ONLY with valid JSON. No other text."

        payload = {
            "model": self.model,
            "prompt": full_prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            }
        }

        try:
            resp = self.client.post(url, json=payload, timeout=60)
            resp.raise_for_status()
            result = resp.json()
            # Ollama returns a `response` or similar field depending on server version
            answer = result.get("response") or result.get("output") or ""
            if json_mode:
                try:
                    parsed = json.loads(answer)
                    return parsed
                except json.JSONDecodeError:
                    logger.warning("Ollama returned non-JSON despite json_mode; returning raw text")
                    return {"answer": answer}
            return {"answer": answer}
        except Exception as e:
            logger.error("Ollama generation failed: %s", e)
            return {"answer": "Error: Could not generate response", "error": str(e)}

    def generate_structured(
        self,
        system_prompt: str,
        user_prompt: str,
        output_schema: Dict[str, Any],
        temperature: float = 0.3
    ) -> Dict[str, Any]:
        """Request structured JSON output from Ollama by instructing the model."""
        schema_prompt = f"""
You must respond with valid JSON matching this schema:

{json.dumps(output_schema, indent=2)}

Ensure all required fields are present and types are correct.
"""
        enhanced_system = system_prompt + "\n\n" + schema_prompt
        return self.generate(
            system_prompt=enhanced_system,
            user_prompt=user_prompt,
            temperature=temperature,
            json_mode=True
        )


# Example usage
if __name__ == "__main__":
    # Test with Ollama (free, local)
    llm = LLMService(provider="ollama", model="mistral")
    
    response = llm.generate(
        system_prompt="You are a helpful financial analyst.",
        user_prompt="Explain what ROE means in simple terms.",
        temperature=0.5
    )
    
    print("Response:", response["answer"])
    
    # Test structured output
    classification = llm.classify_intent(
        user_query="Compare Apple and Microsoft's revenue growth in 2023",
        intent_classes=["comparison", "recommendation", "trend", "general"]
    )
    
    print("\nIntent classification:", classification)
    
    # Test entity extraction
    entities = llm.extract_entities(
        text="Apple's ROE was 147% in 2023, while Microsoft achieved 42%.",
        entity_types=["company", "metric", "year"]
    )
    
    print("\nExtracted entities:", entities)
