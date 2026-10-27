import os
from typing import Dict, Any, Optional

from src.intent_classifier import IntentClassifier
from src.entity_extractor import EntityExtractor
from src.dialog_manager import DialogManager
from src.llm_fallback import LLMFallbackEngine


class TravelBot:
    """
    Головний клас бота-консультанта з туризму 'Wanderlust AI'.
    Об'єднує в єдиний пайплайн:
    1. Попередню обробку тексту та вилучення сутностей (NER).
    2. Класифікацію намірів користувача (ML Intent Classifier).
    3. Менеджер контексту та збереження пам'яті розмови (Dialog Manager & Slots).
    4. Інтелектуальний fallback / LLM engine.
    """

    def __init__(
        self,
        dataset_path: str = "dataset/intents.json",
        tours_path: str = "dataset/tours_db.json",
        model_path: str = "intent_model.joblib",
        confidence_threshold: float = 0.28
    ):
        self.confidence_threshold = confidence_threshold
        self.dataset_path = dataset_path

        # Ініціалізація компонентів
        self.classifier = IntentClassifier(model_path=model_path)
        self.extractor = EntityExtractor()
        self.dialog_manager = DialogManager(tours_db_path=tours_path)
        self.llm_engine = LLMFallbackEngine()

        # Завантаження або автоматичне навчання моделі
        if not self.classifier.load_model():
            print("🚀 Початкове навчання моделі класифікації намірів...")
            stats = self.classifier.train_from_file(self.dataset_path)
            print(f"✅ Модель успішно навчено! Точність CV: {stats['cv_mean_accuracy']:.2%}")

    def process_message(self, text: str) -> Dict[str, Any]:
        """
        Обробляє вхідне повідомлення від користувача та повертає повну структуру відповіді.
        """
        clean_input = text.strip()
        if not clean_input:
            return {
                "bot_text": "Будь ласка, введіть ваше запитання щодо турів чи подорожей!",
                "intent": "empty",
                "confidence": 0.0,
                "entities": {},
                "slots": self.dialog_manager.slots,
                "action_type": "empty"
            }

        # 1. Розпізнавання сутностей (Entities)
        extracted_entities = self.extractor.extract_entities(clean_input)

        # 2. Класифікація наміру (ML Intent Classification)
        intent, confidence, all_probs = self.classifier.predict(clean_input)

        # Якщо ведеться активний сценарій і впевненість класифікатора низька, але користувач надав сутність
        # (наприклад, лаконічна відповідь: "В Єгипет", "На двох", "до 800$"), продовжуємо поточний намір
        if confidence < self.confidence_threshold and self.dialog_manager.active_flow and extracted_entities:
            intent = self.dialog_manager.active_flow
            confidence = 0.92

        # 3. Перевірка порогу впевненості (Confidence Threshold)
        if confidence < self.confidence_threshold:
            # Виклик LLM / Generative fallback
            llm_text = self.llm_engine.generate_response(
                user_text=clean_input,
                history=self.dialog_manager.history,
                current_slots=self.dialog_manager.slots
            )
            self.dialog_manager.history.append({"role": "user", "text": clean_input, "intent": "fallback_llm"})
            self.dialog_manager.history.append({"role": "bot", "text": llm_text, "action": "llm_fallback"})

            return {
                "bot_text": llm_text,
                "intent": "low_confidence_fallback",
                "confidence": float(confidence),
                "all_probs": all_probs,
                "entities": extracted_entities,
                "slots": self.dialog_manager.slots,
                "action_type": "llm_fallback",
                "found_tours": []
            }

        # 4. Генерація відповіді через Dialogue Manager (з урахуванням пам'яті та слотів)
        response_data = self.dialog_manager.generate_response(
            intent=intent,
            confidence=confidence,
            entities=extracted_entities,
            raw_text=clean_input,
            intents_config=self.classifier.intents_data
        )

        return {
            "bot_text": response_data["text"],
            "intent": intent,
            "confidence": float(confidence),
            "all_probs": all_probs,
            "entities": extracted_entities,
            "slots": response_data["current_slots"],
            "action_type": response_data["action_type"],
            "found_tours": response_data["found_tours"]
        }

    def reset(self):
        """Очищення пам'яті розмови."""
        self.dialog_manager.reset_context()
