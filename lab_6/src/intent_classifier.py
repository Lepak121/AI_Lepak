import os
import json
from typing import Tuple, Dict, Any, List
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score

from src.nlp_processor import preprocess_for_tfidf


class IntentClassifier:
    """
    Класифікатор намірів (Intent Classifier) на основі TF-IDF та Logistic Regression.
    Забезпечує:
    - Високу точність класифікації для української мови завдяки кастомному стемінгу.
    - Оцінку впевненості (confidence score) для кожного передбаченого наміру.
    - Автоматичне збереження та завантаження натренованої моделі через joblib.
    """

    def __init__(self, model_path: str = "intent_model.joblib"):
        self.model_path = model_path
        self.pipeline: Pipeline = None
        self.classes: List[str] = []
        self.intents_data: Dict[str, Any] = {}

    def train_from_file(self, dataset_path: str) -> Dict[str, Any]:
        """
        Завантажує датасет з JSON та тренує ML пайплайн.
        """
        with open(dataset_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.intents_data = {item["intent"]: item for item in data["intents"]}

        texts = []
        labels = []

        for item in data["intents"]:
            intent_name = item["intent"]
            for example in item["examples"]:
                processed = preprocess_for_tfidf(example)
                texts.append(processed)
                labels.append(intent_name)

        # Побудова пайплайну: TF-IDF (1-2 грами) + Logistic Regression
        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)),
            ("clf", LogisticRegression(C=3.0, max_iter=1000, random_state=42))
        ])

        # Крос-валідація для оцінки якості
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(self.pipeline, texts, labels, cv=cv, scoring="accuracy")

        # Фінальне навчання на всіх даних
        self.pipeline.fit(texts, labels)
        self.classes = list(self.pipeline.classes_)

        # Збереження моделі
        joblib.dump({
            "pipeline": self.pipeline,
            "classes": self.classes,
            "intents_data": self.intents_data
        }, self.model_path)

        return {
            "num_samples": len(texts),
            "num_classes": len(self.classes),
            "cv_mean_accuracy": float(cv_scores.mean()),
            "cv_scores": [float(s) for s in cv_scores]
        }

    def load_model(self) -> bool:
        """
        Завантажує збережену модель з диску, якщо файл існує.
        """
        if os.path.exists(self.model_path):
            data = joblib.load(self.model_path)
            self.pipeline = data["pipeline"]
            self.classes = data["classes"]
            self.intents_data = data.get("intents_data", {})
            return True
        return False

    def predict(self, text: str) -> Tuple[str, float, Dict[str, float]]:
        """
        Передбачає намір для введеного тексту.
        Повертає: (найкращий_намір, впевненість, словник_ймовірностей_для_всіх_класів)
        """
        if self.pipeline is None:
            raise RuntimeError("Модель ще не навчена або не завантажена!")

        processed_text = preprocess_for_tfidf(text)
        probs = self.pipeline.predict_proba([processed_text])[0]
        
        prob_dict = {cls_name: float(p) for cls_name, p in zip(self.classes, probs)}
        best_intent = max(prob_dict, key=prob_dict.get)
        confidence = prob_dict[best_intent]

        return best_intent, confidence, prob_dict
