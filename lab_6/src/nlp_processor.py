import re
from typing import List

# Спрощений український стемер для нормалізації словоформ (відмінки, числа)
UKR_VOWELS = set("аеєиіїоуюя")

# Регулярні вирази для типових українських закінчень
PERFECTIVE_GROUND = re.compile(r"(ив|ивши|ившись|явшись|ва|вав|вали)$")
REFLEXIVE = re.compile(r"(ся|сь)$")
ADJECTIVE = re.compile(r"(ими|их|им|ого|ому|ою|ій|іх|их|им|ими|ій|ий|ої|е|є|а|я|і|и)$")
PARTICIPLE = re.compile(r"(ий|ого|ому|им|ім|ою|ій|і|их|ими|у|ю|а|я|е|є)$")
VERB = re.compile(r"(ла|на|ли|ло|ти|ли|лося|лися|мо|те|ють|ять|уть|ать)$")
NOUN = re.compile(r"(ами|ями|ою|ею|єю|ів|ев|єв|ей|ам|ям|ах|ях|ом|ем|єм|ові|еві|єві|а|я|о|е|є|і|и|у|ю)$")


def ukrainian_stem(word: str) -> str:
    """
    Легкий алгоритм стемінгу для української мови (Porter-like heuristic).
    Нормалізує відмінкові форми іменників та прикметників для стабільної роботи TF-IDF та пошуку.
    """
    word = word.lower().strip()
    if len(word) <= 3:
        return word

    # Видаляємо зворотні частки (-ся, -сь)
    word = REFLEXIVE.sub("", word)

    # Видаляємо дієслівні закінчення або закінчення іменників/прикметників
    if NOUN.search(word):
        word = NOUN.sub("", word)
    elif ADJECTIVE.search(word):
        word = ADJECTIVE.sub("", word)
    elif VERB.search(word):
        word = VERB.sub("", word)

    return word if len(word) >= 2 else word


def clean_text(text: str) -> str:
    """
    Базова очистка тексту: переведення в нижній регістр, видалення зайвих символів.
    """
    text = text.lower()
    # Залишаємо українські літери, цифри, латиницю, пробіли та символи валют
    text = re.sub(r"[^\w\s\$\€\₴\-]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize_and_stem(text: str) -> List[str]:
    """
    Токенізація та стемінг тексту українською мовою.
    """
    cleaned = clean_text(text)
    tokens = cleaned.split()
    return [ukrainian_stem(token) for token in tokens if len(token) > 0]


def preprocess_for_tfidf(text: str) -> str:
    """
    Попередня підготовка рядка для передачі в TfidfVectorizer.
    """
    stemmed_tokens = tokenize_and_stem(text)
    return " ".join(stemmed_tokens)
