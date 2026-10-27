import json
import random
from typing import Dict, Any, List, Optional


class DialogManager:
    """
    Менеджер контексту та діалогу (Context Memory & State Machine).
    Відповідає за:
    - Збереження історії реплік (multi-turn memory).
    - Заповнення слотів (slot filling): напрямок, бюджет, кількість людей, тип відпочинку.
    - Управління станом поточної розмови (active flows: search, booking, inquiry).
    - Генерацію фінальних відповідей з урахуванням накопиченого контексту.
    """

    def __init__(self, tours_db_path: str = "dataset/tours_db.json"):
        self.tours_db_path = tours_db_path
        self.tours: List[Dict[str, Any]] = []
        self.country_info: Dict[str, Any] = {}
        self.load_tours_db()
        self.reset_context()

    def load_tours_db(self):
        try:
            with open(self.tours_db_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.tours = data.get("tours", [])
                self.country_info = data.get("country_info", {})
        except Exception:
            self.tours = []
            self.country_info = {}

    def reset_context(self):
        """Скидає контекст розмови до початкового стану."""
        self.slots = {
            "destination": None,
            "budget": None,
            "travelers_count": None,
            "tour_type": None,
            "duration": None
        }
        self.active_flow: Optional[str] = None
        self.selected_tour: Optional[Dict[str, Any]] = None
        self.history: List[Dict[str, Any]] = []

    def update_slots(self, new_entities: Dict[str, Any]):
        """Оновлює слоти виявленими сутностями."""
        for key, val in new_entities.items():
            if val is not None:
                self.slots[key] = val

    def get_state_summary(self) -> Dict[str, Any]:
        """Повертає поточний стан діалогу для відображення в UI чи логах."""
        return {
            "active_flow": self.active_flow,
            "slots": {k: v for k, v in self.slots.items() if v is not None},
            "selected_tour": self.selected_tour.get("title") if self.selected_tour else None,
            "history_turns": len(self.history)
        }

    def generate_response(
        self,
        intent: str,
        confidence: float,
        entities: Dict[str, Any],
        raw_text: str,
        intents_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Генерує відповідь бота на основі наміру, слотів та контексту.
        """
        self.update_slots(entities)
        self.history.append({"role": "user", "text": raw_text, "intent": intent, "entities": entities})

        response_text = ""
        action_type = "info"
        found_tours = []

        # Обробка сценаріїв відповідно до Intent
        if intent == "greeting":
            responses = intents_config.get("greeting", {}).get("responses", [
                "Вітаю! Я туристичний асистент Wanderlust. Чим можу допомогти?"
            ])
            response_text = random.choice(responses)
            action_type = "greeting"

        elif intent == "search_tours":
            self.active_flow = "search_tours"
            dest = self.slots.get("destination")
            tour_type = self.slots.get("tour_type")
            budget = self.slots.get("budget")

            # Фільтруємо тури за наявними критеріями
            matched = self._filter_tours(dest, tour_type, budget)

            if matched:
                self.selected_tour = matched[0]
                found_tours = matched[:3]
                response_text = f"🔎 За вашим запитом знайдено {len(matched)} чудових турів:\n\n"
                for i, t in enumerate(found_tours, 1):
                    response_text += (
                        f"**{i}. {t['title']}**\n"
                        f"📍 {t['destination']} ({t['city']}) | ⭐️ {t['hotel_rating']} | 🍽 {t['board']}\n"
                        f"⏱ {t['duration_days']} днів | 💰 **${t['price_usd']}** за особу\n"
                        f"_{t['description']}_\n\n"
                    )
                response_text += "Бажаєте дізнатися детальніше про один із варіантів або оформити бронювання?"
                action_type = "tours_list"
            else:
                if not dest and not tour_type:
                    response_text = (
                        "Які країни або формат відпочинку вас цікавлять? Наприклад:\n"
                        "🌴 Пляжний: *Єгипет, Туреччина, Греція*\n"
                        "🏛 Екскурсійний: *Італія, Іспанія, Норвегія*\n"
                        "🏔 Гірськолижний/SPA: *Карпати, Буковель*"
                    )
                else:
                    target = dest or tour_type
                    response_text = (
                        f"На жаль, точних турів за запитом '{target}' зараз немає в гарячих пропозиціях. "
                        f"Проте ми можемо підібрати індивідуальний тур або змінити параметри пошуку!"
                    )
                action_type = "clarification"

        elif intent == "book_tour":
            self.active_flow = "booking"
            dest = self.slots.get("destination")
            travelers = self.slots.get("travelers_count") or 2

            if self.selected_tour:
                tour = self.selected_tour
                total_cost = tour['price_usd'] * travelers
                response_text = (
                    f"🎉 **Заявку на бронювання майже сформовано!**\n\n"
                    f"✈️ **Тур**: {tour['title']}\n"
                    f"📍 **Напрямок**: {tour['destination']} ({tour['city']})\n"
                    f"👥 **Кількість подорожуючих**: {travelers} чол.\n"
                    f"💵 **Орієнтовна загальна сума**: **${total_cost}** ({tour['duration_days']} днів, {tour['board']})\n\n"
                    f"Будь ласка, вкажіть контактний телефон або електронну пошту для підтвердження броні!"
                )
                action_type = "booking_confirm"
            elif dest:
                response_text = (
                    f"Оформлюємо бронювання у напрямку **{dest}** на **{travelers}** осіб. "
                    f"Який готель або дати вильоту ви розглядаєте?"
                )
                action_type = "booking_pending"
            else:
                response_text = (
                    "Щоб оформити бронювання, спочатку оберіть тур або вкажіть напрямок, "
                    "куди ви бажаєте вирушити (наприклад: *'Бронюю тур в Єгипет на двох'*)."
                )
                action_type = "booking_no_target"

        elif intent == "check_weather":
            dest = self.slots.get("destination")
            dest_key = dest.lower() if dest else None
            
            info = self.country_info.get(dest_key) if dest_key else None
            if info and "weather" in info:
                response_text = (
                    f"☀️ **Кліматичний гід ({dest}):**\n\n"
                    f"{info['weather']}\n\n"
                    f"Бажаєте переглянути доступні тури на цей напрямок?"
                )
            else:
                response_text = (
                    "Про погоду в якому саме регіоні ви хочете дізнатися? "
                    "Ми маємо актуальні кліматичні дані для Єгипту, Туреччини, Італії, Іспанії, Греції, Карпат, Балі та Норвегії."
                )
            action_type = "weather_info"

        elif intent == "faq_visa_docs":
            dest = self.slots.get("destination")
            dest_key = dest.lower() if dest else None
            
            info = self.country_info.get(dest_key) if dest_key else None
            if info and "visa" in info:
                response_text = (
                    f"🛂 **Правила в'їзду та документи ({dest}):**\n\n"
                    f"{info['visa']}\n\n"
                    f"Медична страховка обов'язкова для всіх виїздів за кордон та входить у наші турпакети."
                )
            else:
                response_text = (
                    "📄 **Основні правила для українських туристів:**\n\n"
                    "1. До країн ЄС діє **безвізовий режим** за біометричним закордонним паспортом (до 90 днів).\n"
                    "2. Паспорт має бути дійсним мінімум 3-6 місяців після дати завершення поїздки.\n"
                    "3. У Єгипет віза ставиться по прибуттю ($25), або безкоштовний штамп Sinai Only (до 15 днів).\n"
                    "4. У Туреччину в'їзд вільний за біометричним паспортом або ID-карткою."
                )
            action_type = "docs_info"

        elif intent == "cancel_or_change":
            response_text = (
                "🔄 **Умови скасування та перенесення подорожі:**\n\n"
                "• Безкоштовне скасування можливе за 14+ днів до вильоту (згідно з тарифом туроператора).\n"
                "• Якщо оформлено страховку від невиїзду — повертається до 90% вартості туру навіть при форс-мажорі або хворобі.\n"
                "• Для перенесення дат або заміни туриста напишіть нам номер вашої броні, і наш оператор допоможе внести зміни без зайвих штрафів."
            )
            action_type = "cancel_policy"

        elif intent == "human_operator":
            response_text = (
                "👨‍💼 **Підключення тревел-консультанта:**\n\n"
                "Я передав запит живому менеджеру підтримки Wanderlust Travel.\n"
                "📞 Гаряча лінія: **+380 (44) 123-45-67** (безкоштовно по Україні)\n"
                "💬 Telegram оператора: **@wanderlust_support**\n"
                "Оператор відповість у цьому чаті протягом 2-3 хвилин!"
            )
            action_type = "operator"

        elif intent == "goodbye":
            responses = intents_config.get("goodbye", {}).get("responses", [
                "До побачення! Гарного дня та незабутніх подорожей!"
            ])
            response_text = random.choice(responses)
            action_type = "goodbye"

        else:
            response_text = (
                "Перепрошую, я не зовсім впевнений, що правильно вас зрозумів. "
                "Я вмію підбирати тури, бронювати поїздки, консультувати щодо погоди, віз та документів. Спробуйте уточнити запит!"
            )
            action_type = "fallback"

        self.history.append({"role": "bot", "text": response_text, "action": action_type})

        return {
            "text": response_text,
            "action_type": action_type,
            "found_tours": found_tours,
            "current_slots": {k: v for k, v in self.slots.items() if v is not None}
        }

    def _filter_tours(self, destination: Optional[str], tour_type: Optional[str], budget: Optional[Any]) -> List[Dict[str, Any]]:
        results = []
        for tour in self.tours:
            if destination:
                if tour["destination"].lower() != destination.lower() and destination.lower() not in tour["title"].lower():
                    continue
            if tour_type:
                if tour["type"] != tour_type:
                    continue
            if budget and isinstance(budget, dict) and "amount" in budget:
                if tour["price_usd"] > budget["amount"]:
                    continue
            results.append(tour)
        return results
