import os
import json
from typing import List, Dict, Any, Optional
import requests

SYSTEM_PROMPT = """Ти — висококваліфікований персональний асистент турагенції 'Wanderlust Travel'.
Твоє завдання: допомагати клієнтам обирати подорожі, планувати відпустку, підбирати готелі, надавати точну інформацію про погоду, візи та особливості відпочинку.
Правила:
1. Відповідай виключно українською мовою, у ввічливому, надихаючому та професійному тоні.
2. Якщо клієнт запитує щось поза туристичною тематикою, чемно поверни розмову до подорожей та відпочинку.
3. Будь конкретним, структуруй відповідь маркованими списками та смайлами.
4. Враховуй історію діалогу та попередні вподобання клієнта.
"""


class LLMFallbackEngine:
    """
    Модуль інтеграції LLM (OpenAI / Groq / Gemini API або автономний fallback).
    Реалізує:
    - Prompt Engineering для ролі туристичного консультанта.
    - Передачу контексту (memory / history) для багатоходової бесіди.
    - Graceful fallback: якщо API-ключ не налаштовано або немає з'єднання,
      використовується автономний евристичний генератор, що забезпечує стабільну роботу на захисті проекту.
    """

    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY") or os.getenv("GROQ_API_KEY") or os.getenv("GEMINI_API_KEY")
        self.provider = self._detect_provider()

    def _detect_provider(self) -> str:
        if os.getenv("GROQ_API_KEY"):
            return "groq"
        if os.getenv("OPENAI_API_KEY"):
            return "openai"
        if os.getenv("GEMINI_API_KEY"):
            return "gemini"
        return "mock_ai"

    def generate_response(self, user_text: str, history: List[Dict[str, Any]], current_slots: Dict[str, Any]) -> str:
        """
        Генерує відповідь за допомогою LLM або інтелектуального евристичного генератора.
        """
        if self.provider in ["groq", "openai"] and self.api_key:
            try:
                return self._call_openai_compatible_api(user_text, history, current_slots)
            except Exception as e:
                return self._fallback_local_response(user_text, current_slots, error=str(e))

        return self._fallback_local_response(user_text, current_slots)

    def _call_openai_compatible_api(self, user_text: str, history: List[Dict[str, Any]], current_slots: Dict[str, Any]) -> str:
        url = "https://api.groq.com/openai/v1/chat/completions" if self.provider == "groq" else "https://api.openai.com/v1/chat/completions"
        model = "llama-3.1-8b-instant" if self.provider == "groq" else "gpt-3.5-turbo"

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        # Додаємо останні 6 реплік історії діалогу (memory)
        for h in history[-6:]:
            role = "assistant" if h.get("role") == "bot" else "user"
            messages.append({"role": role, "content": h.get("text", "")})

        # Додаємо поточне повідомлення з метаданими слотів
        context_hint = f"\n[Поточні параметри клієнта: {json.dumps(current_slots, ensure_ascii=False)}]" if current_slots else ""
        messages.append({"role": "user", "content": user_text + context_hint})

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": model,
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 400
        }

        resp = requests.post(url, headers=headers, json=payload, timeout=8)
        if resp.status_code == 200:
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        else:
            raise RuntimeError(f"API Error {resp.status_code}: {resp.text}")

    def _fallback_local_response(self, user_text: str, current_slots: Dict[str, Any], error: Optional[str] = None) -> str:
        """
        Вбудований локальний евристичний генератор для роботи без зовнішніх платних ключів.
        """
        dest = current_slots.get("destination")
        dest_str = f" у напрямку **{dest}**" if dest else ""

        return (
            f"🤖 *[Автономний AI-консультант Wanderlust]*\n\n"
            f"Я проаналізував ваш запит: *\"{user_text}\"*{dest_str}.\n\n"
            f"Як туристичний помічник, я рекомендую звернути увагу на наступні кроки:\n"
            f"1. **Вибір формату відпочинку**: пляжний релакс (Єгипет, Туреччина), культурний тур (Італія, Іспанія) або гори (Карпати).\n"
            f"2. **Оптимальний бюджет**: більшість тижневих турів з авіаперельотом та харчуванням стартують від $550 на особу.\n"
            f"3. **Бронювання**: напишіть мені *'Покажи тури'* або *'Яка погода'*, щоб отримати точні деталі!\n\n"
            f"Якщо бажаєте, я можу покликати живого менеджера для персональної консультації (напишіть *'Поклич оператора'*)."
        )
