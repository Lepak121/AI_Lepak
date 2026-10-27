import sys
import json

# Забезпечення коректного виведення UTF-8 у Windows консолі
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from src.intent_classifier import IntentClassifier
from src.nlp_processor import preprocess_for_tfidf


def run_evaluation():
    print("=" * 60)
    print(" [*] ВАЛІДАЦІЯ ТА ОЦІНКА ЯКОСТІ КЛАСИФІКАЦІЇ НАМІРІВ (INTENTS)")
    print("=" * 60)

    with open("dataset/intents.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    texts = []
    labels = []
    for item in data["intents"]:
        intent = item["intent"]
        for ex in item["examples"]:
            texts.append(preprocess_for_tfidf(ex))
            labels.append(intent)

    print(f"Всього прикладів у датасеті: {len(texts)}")
    print(f"Кількість класів намірів: {len(set(labels))}")

    # Розбиття на train / test (75% / 25%)
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.25, random_state=42, stratify=labels
    )

    classifier = IntentClassifier()
    stats = classifier.train_from_file("dataset/intents.json")
    print(f"\nКрос-валідація (5-Fold Stratified):")
    print(f"Середня точність (Accuracy): {stats['cv_mean_accuracy']:.2%}")
    print(f"Оцінки фолдів: {[f'{s:.2%}' for s in stats['cv_scores']]}")

    # Оцінка на тестовій вибірці
    pipeline = classifier.pipeline
    y_pred = pipeline.predict(X_test)

    print("\n" + "=" * 60)
    print(" [*] CLASSIFICATION REPORT (PRECISION, RECALL, F1-SCORE):")
    print("=" * 60)
    report = classification_report(y_test, y_pred, digits=4)
    print(report)

    print("=" * 60)
    print(" [*] ТЕСТУВАННЯ НА РЕАЛЬНИХ НЕБАЧЕНИХ ФРАЗАХ:")
    print("=" * 60)
    unseen_phrases = [
        "Салют, друже!",
        "Хочу на море в Анталію з дружиною на 7 днів",
        "Бронюю тур в Рим на двох прямо зараз",
        "Чи не буде дощу в Греції в липні?",
        "Скільки ще має діяти мій біометричний паспорт?",
        "Хочу скасувати поїздку через сімейні обставини",
        "З'єднайте з живим оператором терміново",
        "Дякую, гарного дня!",
        "Яка вартість квитка на Марс?"  # Out of domain query
    ]

    for phrase in unseen_phrases:
        intent, conf, _ = classifier.predict(phrase)
        print(f"Запит: '{phrase}'")
        print(f" -> Передбачений намір: [{intent}] (впевненість: {conf:.1%})")
        print("-" * 50)


if __name__ == "__main__":
    run_evaluation()
