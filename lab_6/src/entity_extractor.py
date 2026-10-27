import re
from typing import Dict, Any, Optional

DESTINATIONS_MAP = {
    # Країни та їх відмінкові форми/аліаси
    "єгипет": "Єгипет",
    "єгипт": "Єгипет",
    "єгипті": "Єгипет",
    "єгипту": "Єгипет",
    "шарм": "Єгипет",
    "шарм-ель-шейх": "Єгипет",
    "хургад": "Єгипет",
    "хургада": "Єгипет",
    "хургаді": "Єгипет",
    
    "туреччин": "Туреччина",
    "туреччина": "Туреччина",
    "туреччині": "Туреччина",
    "туреччину": "Туреччина",
    "анталі": "Туреччина",
    "анталія": "Туреччина",
    "анталії": "Туреччина",
    "каппадокі": "Туреччина",
    "каппадокія": "Туреччина",
    "стамбул": "Туреччина",
    
    "італі": "Італія",
    "італія": "Італія",
    "італії": "Італія",
    "італію": "Італія",
    "рим": "Італія",
    "римі": "Італія",
    "риму": "Італія",
    "венеці": "Італія",
    "венеція": "Італія",
    "флоренці": "Італія",
    
    "іспані": "Іспанія",
    "іспанія": "Іспанія",
    "іспанії": "Іспанія",
    "іспанію": "Іспанія",
    "барселон": "Іспанія",
    "барселона": "Іспанія",
    "мадрид": "Іспанія",
    
    "греці": "Греція",
    "греція": "Греція",
    "греції": "Греція",
    "грецію": "Греція",
    "крит": "Греція",
    "санторіні": "Греція",
    "афін": "Греція",
    
    "карпат": "Карпати",
    "карпати": "Карпати",
    "карпатах": "Карпати",
    "карпати": "Карпати",
    "буковель": "Карпати",
    "буковелі": "Карпати",
    "яремч": "Карпати",
    
    "норвегі": "Норвегія",
    "норвегія": "Норвегія",
    "норвегії": "Норвегія",
    "норвегію": "Норвегія",
    "фіорд": "Норвегія",
    "фіорди": "Норвегія",
    
    "балі": "Балі",
    "мальдів": "Мальдіви",
    "мальдіви": "Мальдіви",
    "чорногорі": "Чорногорія",
    "чорногорія": "Чорногорія",
    "чорногорії": "Чорногорія",
    "дубай": "ОАЕ",
    "оае": "ОАЕ"
}

TOUR_TYPES_MAP = {
    "пляж": "пляжний",
    "пляжн": "пляжний",
    "мор": "пляжний",
    "море": "пляжний",
    "океан": "пляжний",
    "засмаг": "пляжний",
    
    "екскурс": "екскурсійний",
    "екскурсійн": "екскурсійний",
    "музе": "екскурсійний",
    "пам'ятк": "екскурсійний",
    "побачити": "екскурсійний",
    
    "лиж": "гірськолижний",
    "гірськолижн": "гірськолижний",
    "сноуборд": "гірськолижний",
    "гори": "гірськолижний",
    
    "спа": "спа",
    "spa": "спа",
    "релакс": "спа",
    "оздоровленн": "спа"
}


class EntityExtractor:
    """
    Модуль розпізнавання іменованих сутностей (NER / Slot Extractor) для туристичної сфери.
    Виділяє:
    - destination (напрямок подорожі)
    - budget (бюджет у валюті або числовому виразі)
    - travelers_count (кількість туристів)
    - tour_type (пляжний, екскурсійний, гірськолижний, спа)
    - duration (тривалість поїздки)
    """

    def extract_entities(self, text: str) -> Dict[str, Any]:
        text_lower = text.lower()
        entities = {
            "destination": self._extract_destination(text_lower),
            "budget": self._extract_budget(text_lower),
            "travelers_count": self._extract_travelers(text_lower),
            "tour_type": self._extract_tour_type(text_lower),
            "duration": self._extract_duration(text_lower)
        }
        return {k: v for k, v in entities.items() if v is not None}

    def _extract_destination(self, text: str) -> Optional[str]:
        # Спочатку перевірка багатослівних назв
        if "шарм" in text:
            return "Єгипет"
        if "буковель" in text:
            return "Карпати"
        if "шрі-ланка" in text or "шрі ланка" in text:
            return "Шрі-Ланка"

        # Пошук точних або довгих префіксних збігів
        words = re.findall(r"[\w\-]+", text)
        for word in words:
            if word in DESTINATIONS_MAP:
                return DESTINATIONS_MAP[word]

        for word in words:
            if len(word) >= 4:
                for key, val in DESTINATIONS_MAP.items():
                    if len(key) >= 4 and (word.startswith(key) or key.startswith(word)):
                        return val
        return None

    def _extract_budget(self, text: str) -> Optional[Dict[str, Any]]:
        # Патерни на кшталт: "$800", "800$", "1000 євро", "до 1200 usd", "25000 грн"
        patterns = [
            r"(\d+)\s*(доларів|дол|usd|\$)",
            r"(\$)\s*(\d+)",
            r"(\d+)\s*(євро|eur|\€)",
            r"(\€)\s*(\d+)",
            r"(\d+)\s*(гривень|грн|\₴)",
            r"(до|бюджет|близько)\s*(\d+)\s*(доларів|дол|usd|\$|євро|eur|\€|грн)?"
        ]
        
        # 1. Пошук прямої суми з валютою
        usd_match = re.search(r"(?:до\s*)?(\d+)\s*(?:\$|usd|дол|долар)", text)
        if usd_match:
            return {"amount": int(usd_match.group(1)), "currency": "USD"}
            
        eur_match = re.search(r"(?:до\s*)?(\d+)\s*(?:€|eur|євро)", text)
        if eur_match:
            return {"amount": int(eur_match.group(1)), "currency": "EUR"}
            
        uah_match = re.search(r"(?:до\s*)?(\d+)\s*(?:₴|грн|грив)", text)
        if uah_match:
            return {"amount": int(uah_match.group(1)), "currency": "UAH"}

        # Просто число поруч зі словами бюджет/до/вартість
        num_match = re.search(r"(?:до|бюджет|в межах|до)\s*(\d{3,5})", text)
        if num_match:
            return {"amount": int(num_match.group(1)), "currency": "USD"}

        if "бюджетн" in text or "дешев" in text or "економ" in text:
            return {"tier": "economy"}
        if "люкс" in text or "преміум" in text or "vip" in text:
            return {"tier": "premium"}
            
        return None

    def _extract_travelers(self, text: str) -> Optional[int]:
        # Специфічні українські конструкції
        if re.search(r"\b(на одного|для одного|один|1 людин|1 доросл)\b", text):
            return 1
        if re.search(r"\b(на двох|для двох|двоє|двох|2 люди|2 доросл|парою)\b", text):
            return 2
        if re.search(r"\b(на трьох|для трьох|троє|3 люди|3 доросл)\b", text):
            return 3
        if re.search(r"\b(на чотирьох|для чотирьох|четверо|4 люди|4 доросл)\b", text):
            return 4
        if re.search(r"\b(з дитиною|сім'я|сімейний|родиною)\b", text):
            return 3  # Сім'я типово 3 особи
        
        # Загальний патерн "N людей/осіб/туристів"
        match = re.search(r"(\d+)\s*(?:людин|осіб|турист|дорослих)", text)
        if match:
            return int(match.group(1))
        return None

    def _extract_tour_type(self, text: str) -> Optional[str]:
        words = re.findall(r"[\w\-]+", text)
        for word in words:
            for key, val in TOUR_TYPES_MAP.items():
                if word.startswith(key):
                    return val
        return None

    def _extract_duration(self, text: str) -> Optional[str]:
        # "на 7 днів", "на тиждень", "на вихідні", "10 ночей"
        if "вихідн" in text or "вікенд" in text:
            return "3-4 дні (вихідні)"
        if "тиждень" in text:
            return "7 днів (тиждень)"
        
        match = re.search(r"(\d+)\s*(?:днів|дні|ночей|доби)", text)
        if match:
            return f"{match.group(1)} днів"
        return None
