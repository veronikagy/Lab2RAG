import logging
from typing import List, Dict, Any, Optional, AsyncGenerator
import httpx
import json
import asyncio

logger = logging.getLogger(__name__)

class OpenRouterClient:
    """OpenRouter API client for LLM inference"""
    
    def __init__(self, api_key: str, base_url: str = "https://openrouter.ai/api/v1"):
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "HTTP-Referer": "http://localhost:8501",  # For OpenRouter
            "X-Title": "RAG System"
        }
        self._available_models = None
    
    async def get_available_models(self) -> List[Dict[str, Any]]:
        """Get list of available models from OpenRouter"""
        if self._available_models is not None:
            return self._available_models
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.base_url}/models",
                    headers=self.headers,
                    timeout=30.0
                )
                response.raise_for_status()
                
                data = response.json()
                models = data.get("data", [])
                
                # Filter for chat models and add useful metadata
                chat_models = []
                for model in models:
                    model_info = {
                        "id": model.get("id", ""),
                        "name": model.get("name", ""),
                        "description": model.get("description", ""),
                        "context_length": model.get("context_length", 4096),
                        "pricing": model.get("pricing", {}),
                        "top_provider": model.get("top_provider", {}),
                        "per_request_limits": model.get("per_request_limits")
                    }
                    chat_models.append(model_info)
                
                self._available_models = sorted(chat_models, key=lambda x: x["name"])
                return self._available_models
                
        except Exception as e:
            logger.error(f"Error fetching available models: {e}")
            return []
    
    async def generate_response(
        self,
        messages: List[Dict[str, str]],
        model: str,
        max_tokens: int = 1000,
        temperature: float = 0.7,
        stream: bool = False
    ) -> str:
        """Generate response from LLM"""
        try:
            payload = {
                "model": model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": stream
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers=self.headers,
                    json=payload,
                    timeout=60.0
                )
                response.raise_for_status()
                
                data = response.json()
                
                if "choices" in data and len(data["choices"]) > 0:
                    return data["choices"][0]["message"]["content"]
                else:
                    logger.error(f"No choices in response: {data}")
                    return "Извините, не удалось получить ответ от модели."
                    
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error {e.response.status_code}: {e.response.text}")
            return f"Ошибка API: {e.response.status_code}"
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"Ошибка генерации ответа: {str(e)}"
    
    async def generate_response_stream(
        self,
        messages: List[Dict[str, str]],
        model: str,
        max_tokens: int = 1000,
        temperature: float = 0.7
    ) -> AsyncGenerator[str, None]:
        """Generate streaming response from LLM"""
        try:
            payload = {
                "model": model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": True
            }
            
            async with httpx.AsyncClient() as client:
                async with client.stream(
                    "POST",
                    f"{self.base_url}/chat/completions",
                    headers=self.headers,
                    json=payload,
                    timeout=60.0
                ) as response:
                    response.raise_for_status()
                    
                    async for line in response.aiter_lines():
                        if line.startswith("data: "):
                            data_str = line[6:]  # Remove "data: " prefix
                            
                            if data_str.strip() == "[DONE]":
                                break
                            
                            try:
                                data = json.loads(data_str)
                                if "choices" in data and len(data["choices"]) > 0:
                                    delta = data["choices"][0].get("delta", {})
                                    content = delta.get("content", "")
                                    if content:
                                        yield content
                            except json.JSONDecodeError:
                                continue
                                
        except Exception as e:
            logger.error(f"Error in streaming response: {e}")
            yield f"Ошибка стриминга: {str(e)}"

class RAGPromptBuilder:
    """Build prompts for RAG system"""
    
    def __init__(self):
        self.system_prompt = """Вы - полезный ИИ-ассистент, специализирующийся на ответах на вопросы на основе предоставленных документов.

Инструкции:
1. Отвечайте ТОЛЬКО на основе предоставленного контекста
2. Если информации недостаточно, честно скажите об этом
3. Указывайте источники информации когда это возможно
4. Отвечайте на русском языке
5. Будьте точными и конкретными

Если контекст не содержит ответа на вопрос, скажите: "Извините, в предоставленных документах нет информации для ответа на этот вопрос."
"""
    
    def build_rag_prompt(
        self, 
        question: str, 
        context_chunks: List[Dict[str, Any]],
        max_context_length: int = 3000
    ) -> List[Dict[str, str]]:
        """Build RAG prompt with context"""
        
        # Build context from chunks
        context_parts = []
        current_length = 0
        
        for i, chunk in enumerate(context_chunks):
            chunk_text = chunk.get("text", "")
            source = chunk.get("source", "unknown")
            score = chunk.get("score", 0.0)
            
            chunk_header = f"\n--- Источник: {source} (релевантность: {score:.3f}) ---\n"
            full_chunk = chunk_header + chunk_text
            
            if current_length + len(full_chunk) > max_context_length:
                break
            
            context_parts.append(full_chunk)
            current_length += len(full_chunk)
        
        context = "\n".join(context_parts)
        
        # Build messages
        messages = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user", 
                "content": f"""Контекст из документов:
{context}

Вопрос: {question}

Пожалуйста, ответьте на вопрос на основе предоставленного контекста."""
            }
        ]
        
        return messages
    
    def build_simple_prompt(self, question: str) -> List[Dict[str, str]]:
        """Build simple prompt without RAG context"""
        return [
            {
                "role": "system", 
                "content": "Вы - полезный ИИ-ассистент. Отвечайте на русском языке."
            },
            {"role": "user", "content": question}
        ]
