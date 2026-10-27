"""
Лабораторна робота: Розпізнавання образів за кількісними та якісними характеристиками.
Теоретична основа:
1. Кількісні характеристики (нормалізуються через Min-Max scaling до [0, 1]).
2. Якісні характеристики:
   - Бінарні (0 або 1).
   - Порядкові / рангові (відображення у діапазон [0, 1] відповідно до ступеня прояву).
   - Номінальні / категоріальні (відстань неспівпадіння: 0 при збігу, 1 при відмінності).
3. Метрика розпізнавання:
   - Узагальнена зважена міра Евкліда (L2)
   - Мангеттенська відстань (L1)
4. Алгоритми класифікації:
   - Метод мінімальної відстані до еталону класу (центроїда)
   - Метод k-найближчих сусідів (k-NN)
"""

import math
from typing import Dict, List, Any, Tuple, Optional


class Feature:
    """Опис ознаки (характеристики) образу."""
    def __init__(self, name: str, ftype: str, weight: float = 1.0, 
                 order_levels: Optional[List[str]] = None, unit: str = ""):
        self.name = name
        self.ftype = ftype  # 'quantitative', 'binary', 'ordinal', 'nominal'
        self.weight = weight
        self.order_levels = order_levels or []
        self.unit = unit
        self.min_val = float('inf')
        self.max_val = float('-inf')


class PatternRecognizer:
    def __init__(self, name: str = "Система розпізнавання образів"):
        self.name = name
        self.features: Dict[str, Feature] = {}
        self.training_data: List[Dict[str, Any]] = []
        self.classes: List[str] = []
        self.class_centroids: Dict[str, Dict[str, float]] = {}

    def add_feature(self, feature: Feature):
        self.features[feature.name] = feature

    def fit(self, dataset: List[Dict[str, Any]]):
        """Навчання моделі: оновлення діапазонів ознак та розрахунок еталонів класів."""
        self.training_data = dataset
        self.classes = sorted(list(set(item['class'] for item in dataset)))

        # 1. Пошук Min та Max для кількісних ознак (Min-Max нормалізація)
        for fname, feat in self.features.items():
            if feat.ftype == 'quantitative':
                values = [float(item[fname]) for item in dataset if fname in item]
                feat.min_val = min(values)
                feat.max_val = max(values)
                if feat.min_val == feat.max_val:
                    feat.max_val += 1e-6

        # 2. Розрахунок центроїдів (еталонів класів)
        self.class_centroids = {}
        for c in self.classes:
            c_samples = [item for item in dataset if item['class'] == c]
            norm_vectors = [self.encode_sample(item) for item in c_samples]
            
            centroid = {}
            for key in norm_vectors[0].keys():
                centroid[key] = sum(v[key] for v in norm_vectors) / len(norm_vectors)
            self.class_centroids[c] = centroid

    def encode_sample(self, sample: Dict[str, Any]) -> Dict[str, float]:
        """Приведення кількісних та якісних ознак до простору [0, 1]."""
        encoded = {}
        for fname, feat in self.features.items():
            val = sample[fname]
            if feat.ftype == 'quantitative':
                val_float = float(val)
                norm_val = (val_float - feat.min_val) / (feat.max_val - feat.min_val)
                encoded[fname] = max(0.0, min(1.0, norm_val))
            elif feat.ftype == 'binary':
                v_str = str(val).strip().lower()
                encoded[fname] = 1.0 if v_str in ('так', 'yes', 'true', '1', '+') else 0.0
            elif feat.ftype == 'ordinal':
                v_str = str(val).strip()
                if v_str in feat.order_levels:
                    idx = feat.order_levels.index(v_str)
                    n_levels = len(feat.order_levels)
                    norm_val = idx / (n_levels - 1) if n_levels > 1 else 0.0
                else:
                    norm_val = 0.5
                encoded[fname] = norm_val
        return encoded

    def feature_distance(self, fname: str, val1: Any, val2: Any) -> float:
        """Обчислення різниці між значеннями ознаки у межах [0, 1]."""
        feat = self.features[fname]
        if feat.ftype == 'quantitative':
            v1_norm = (float(val1) - feat.min_val) / (feat.max_val - feat.min_val)
            v2_norm = (float(val2) - feat.min_val) / (feat.max_val - feat.min_val)
            return abs(v1_norm - v2_norm)
        elif feat.ftype == 'binary':
            b1 = 1 if str(val1).strip().lower() in ('так', 'yes', 'true', '1') else 0
            b2 = 1 if str(val2).strip().lower() in ('так', 'yes', 'true', '1') else 0
            return float(abs(b1 - b2))
        elif feat.ftype == 'ordinal':
            v1_str, v2_str = str(val1).strip(), str(val2).strip()
            idx1 = feat.order_levels.index(v1_str) if v1_str in feat.order_levels else 0
            idx2 = feat.order_levels.index(v2_str) if v2_str in feat.order_levels else 0
            span = max(1, len(feat.order_levels) - 1)
            return abs(idx1 - idx2) / span
        elif feat.ftype == 'nominal':
            return 0.0 if str(val1).strip().lower() == str(val2).strip().lower() else 1.0
        return 0.0

    def euclidean_distance(self, sample1: Dict[str, Any], sample2: Dict[str, Any]) -> Tuple[float, Dict[str, float]]:
        """Узагальнена зважена міра Евкліда (L2)."""
        weighted_sq_sum = 0.0
        sum_weights = 0.0
        feature_diffs = {}

        for fname, feat in self.features.items():
            if fname in sample1 and fname in sample2:
                d_i = self.feature_distance(fname, sample1[fname], sample2[fname])
                w_i = feat.weight
                feature_diffs[fname] = d_i
                weighted_sq_sum += w_i * (d_i ** 2)
                sum_weights += w_i

        if sum_weights == 0:
            return 0.0, feature_diffs

        total_dist = math.sqrt(weighted_sq_sum / sum_weights)
        return total_dist, feature_diffs

    def manhattan_distance(self, sample1: Dict[str, Any], sample2: Dict[str, Any]) -> Tuple[float, Dict[str, float]]:
        """Мангеттенська відстань (L1)."""
        weighted_sum = 0.0
        sum_weights = 0.0
        feature_diffs = {}

        for fname, feat in self.features.items():
            if fname in sample1 and fname in sample2:
                d_i = self.feature_distance(fname, sample1[fname], sample2[fname])
                w_i = feat.weight
                feature_diffs[fname] = d_i
                weighted_sum += w_i * d_i
                sum_weights += w_i

        total_dist = weighted_sum / sum_weights if sum_weights > 0 else 0.0
        return total_dist, feature_diffs

    def classify_by_centroid(self, sample: Dict[str, Any], metric: str = 'euclidean') -> Dict[str, Any]:
        """Класифікація за методом мінімальної відстані до еталону класу."""
        encoded_sample = self.encode_sample(sample)
        distances = {}
        details = {}

        for c, centroid in self.class_centroids.items():
            weighted_acc = 0.0
            sum_weights = 0.0
            feat_breakdown = {}

            for fname, feat in self.features.items():
                w_i = feat.weight
                if feat.ftype in ('quantitative', 'binary', 'ordinal'):
                    val_norm = encoded_sample[fname]
                    cent_norm = centroid[fname]
                    d_i = abs(val_norm - cent_norm)
                else:  # nominal
                    c_values = [s[fname] for s in self.training_data if s['class'] == c]
                    most_common = max(set(c_values), key=c_values.count) if c_values else ""
                    d_i = 0.0 if str(sample[fname]).strip().lower() == str(most_common).strip().lower() else 1.0

                feat_breakdown[fname] = d_i
                if metric == 'manhattan':
                    weighted_acc += w_i * d_i
                else:
                    weighted_acc += w_i * (d_i ** 2)
                sum_weights += w_i

            if metric == 'manhattan':
                total_d = weighted_acc / sum_weights if sum_weights > 0 else 0.0
            else:
                total_d = math.sqrt(weighted_acc / sum_weights) if sum_weights > 0 else 0.0

            distances[c] = total_d
            details[c] = feat_breakdown

        best_class = min(distances, key=distances.get)
        inv_dists = {c: 1.0 / (d + 1e-4) for c, d in distances.items()}
        sum_inv = sum(inv_dists.values())
        confidence = {c: (inv / sum_inv) * 100.0 for c, inv in inv_dists.items()}

        return {
            'predicted_class': best_class,
            'distances': distances,
            'confidence': confidence,
            'details': details,
            'metric': metric
        }

    def classify_knn(self, sample: Dict[str, Any], k: int = 3, metric: str = 'euclidean') -> Dict[str, Any]:
        """Класифікація методом k-найближчих сусідів (k-NN)."""
        distances = []
        for i, train_item in enumerate(self.training_data):
            if metric == 'manhattan':
                d, _ = self.manhattan_distance(sample, train_item)
            else:
                d, _ = self.euclidean_distance(sample, train_item)

            distances.append({
                'index': i,
                'class': train_item['class'],
                'name': train_item.get('name', f"Зразок #{i+1}"),
                'distance': d
            })

        distances.sort(key=lambda x: x['distance'])
        k_nearest = distances[:k]

        votes = {c: 0.0 for c in self.classes}
        for neighbor in k_nearest:
            weight = 1.0 / (neighbor['distance'] + 1e-4)
            votes[neighbor['class']] += weight

        best_class = max(votes, key=votes.get)
        total_votes = sum(votes.values())
        confidence = {c: (votes[c] / total_votes) * 100.0 for c in self.classes}

        return {
            'predicted_class': best_class,
            'k_nearest': k_nearest,
            'confidence': confidence,
            'votes': votes,
            'metric': metric
        }
