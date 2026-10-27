"""
Набори даних для лабораторної роботи з розпізнавання образів смартфонів.
Містять комбінацію якісних (бінарних, порядкових, номінальних) та кількісних ознак.
"""

from pattern_recognition import Feature, PatternRecognizer

def get_smartphones_model() -> PatternRecognizer:
    """
    Створює та навчає модель розпізнавання образів смартфонів.
    Класи:
    - Бюджетний (Budget)
    - Середній клас (Mid-range)
    - Флагман (Flagship)
    - Геймерський (Gaming)
    """
    model = PatternRecognizer(name="Класифікація смартфонів за класами")

    # 1. Кількісні характеристики (Лінійна Min-Max нормалізація)
    model.add_feature(Feature("price_usd", "quantitative", weight=1.2, unit="$"))
    model.add_feature(Feature("ram_gb", "quantitative", weight=1.1, unit="ГБ"))
    model.add_feature(Feature("battery_mah", "quantitative", weight=1.0, unit="мА·год"))
    model.add_feature(Feature("camera_mp", "quantitative", weight=0.9, unit="Мп"))
    model.add_feature(Feature("screen_hz", "quantitative", weight=0.8, unit="Гц"))
    model.add_feature(Feature("weight_g", "quantitative", weight=0.6, unit="г"))

    # 2. Якісні характеристики (Шкалювання та кодування)
    model.add_feature(Feature(
        "water_resistance", "ordinal", weight=1.0,
        order_levels=["Немає", "IP54 (бризки)", "IP68 (занурення)"]
    ))
    model.add_feature(Feature(
        "display_matrix", "ordinal", weight=0.9,
        order_levels=["TFT/IPS", "AMOLED", "LTPO OLED"]
    ))
    model.add_feature(Feature("wireless_charging", "binary", weight=1.0))
    model.add_feature(Feature("active_cooling", "binary", weight=1.3))
    model.add_feature(Feature("body_material", "nominal", weight=0.8))

    training_data = [
        # --- Бюджетні смартфони ---
        {
            "name": "Redmi 13C", "class": "Бюджетний",
            "price_usd": 130, "ram_gb": 4, "battery_mah": 5000, "camera_mp": 50, "screen_hz": 90, "weight_g": 192,
            "water_resistance": "Немає", "display_matrix": "TFT/IPS",
            "wireless_charging": "Ні", "active_cooling": "Ні", "body_material": "Пластик"
        },
        {
            "name": "Samsung Galaxy A05", "class": "Бюджетний",
            "price_usd": 120, "ram_gb": 4, "battery_mah": 5000, "camera_mp": 50, "screen_hz": 60, "weight_g": 195,
            "water_resistance": "Немає", "display_matrix": "TFT/IPS",
            "wireless_charging": "Ні", "active_cooling": "Ні", "body_material": "Пластик"
        },
        {
            "name": "Motorola Moto G24", "class": "Бюджетний",
            "price_usd": 140, "ram_gb": 4, "battery_mah": 5000, "camera_mp": 50, "screen_hz": 90, "weight_g": 181,
            "water_resistance": "IP54 (бризки)", "display_matrix": "TFT/IPS",
            "wireless_charging": "Ні", "active_cooling": "Ні", "body_material": "Пластик"
        },
        {
            "name": "Realme C53", "class": "Бюджетний",
            "price_usd": 150, "ram_gb": 6, "battery_mah": 5000, "camera_mp": 50, "screen_hz": 90, "weight_g": 182,
            "water_resistance": "Немає", "display_matrix": "TFT/IPS",
            "wireless_charging": "Ні", "active_cooling": "Ні", "body_material": "Пластик"
        },

        # --- Середній клас (Mid-range) ---
        {
            "name": "Redmi Note 13 Pro", "class": "Середній клас",
            "price_usd": 320, "ram_gb": 8, "battery_mah": 5000, "camera_mp": 200, "screen_hz": 120, "weight_g": 188,
            "water_resistance": "IP54 (бризки)", "display_matrix": "AMOLED",
            "wireless_charging": "Ні", "active_cooling": "Ні", "body_material": "Скло"
        },
        {
            "name": "Samsung Galaxy A55", "class": "Середній клас",
            "price_usd": 400, "ram_gb": 8, "battery_mah": 5000, "camera_mp": 50, "screen_hz": 120, "weight_g": 213,
            "water_resistance": "IP68 (занурення)", "display_matrix": "AMOLED",
            "wireless_charging": "Ні", "active_cooling": "Ні", "body_material": "Алюміній"
        },
        {
            "name": "POCO X6 Pro", "class": "Середній клас",
            "price_usd": 350, "ram_gb": 12, "battery_mah": 5000, "camera_mp": 64, "screen_hz": 120, "weight_g": 186,
            "water_resistance": "IP54 (бризки)", "display_matrix": "AMOLED",
            "wireless_charging": "Ні", "active_cooling": "Ні", "body_material": "Пластик"
        },
        {
            "name": "Nothing Phone (2a)", "class": "Середній клас",
            "price_usd": 370, "ram_gb": 8, "battery_mah": 5000, "camera_mp": 50, "screen_hz": 120, "weight_g": 190,
            "water_resistance": "IP54 (бризки)", "display_matrix": "AMOLED",
            "wireless_charging": "Ні", "active_cooling": "Ні", "body_material": "Пластик"
        },

        # --- Флагмани ---
        {
            "name": "iPhone 15 Pro", "class": "Флагман",
            "price_usd": 1150, "ram_gb": 8, "battery_mah": 3274, "camera_mp": 48, "screen_hz": 120, "weight_g": 187,
            "water_resistance": "IP68 (занурення)", "display_matrix": "LTPO OLED",
            "wireless_charging": "Так", "active_cooling": "Ні", "body_material": "Титан/Скло"
        },
        {
            "name": "Samsung Galaxy S24 Ultra", "class": "Флагман",
            "price_usd": 1300, "ram_gb": 12, "battery_mah": 5000, "camera_mp": 200, "screen_hz": 120, "weight_g": 232,
            "water_resistance": "IP68 (занурення)", "display_matrix": "LTPO OLED",
            "wireless_charging": "Так", "active_cooling": "Ні", "body_material": "Титан/Скло"
        },
        {
            "name": "Google Pixel 8 Pro", "class": "Флагман",
            "price_usd": 999, "ram_gb": 12, "battery_mah": 5050, "camera_mp": 50, "screen_hz": 120, "weight_g": 213,
            "water_resistance": "IP68 (занурення)", "display_matrix": "LTPO OLED",
            "wireless_charging": "Так", "active_cooling": "Ні", "body_material": "Алюміній"
        },
        {
            "name": "Xiaomi 14 Ultra", "class": "Флагман",
            "price_usd": 1250, "ram_gb": 16, "battery_mah": 5000, "camera_mp": 50, "screen_hz": 120, "weight_g": 220,
            "water_resistance": "IP68 (занурення)", "display_matrix": "LTPO OLED",
            "wireless_charging": "Так", "active_cooling": "Ні", "body_material": "Екошкіра/Титан"
        },

        # --- Геймерські смартфони ---
        {
            "name": "ASUS ROG Phone 8 Pro", "class": "Геймерський",
            "price_usd": 1200, "ram_gb": 16, "battery_mah": 5500, "camera_mp": 50, "screen_hz": 165, "weight_g": 225,
            "water_resistance": "IP68 (занурення)", "display_matrix": "LTPO OLED",
            "wireless_charging": "Так", "active_cooling": "Так", "body_material": "Алюміній"
        },
        {
            "name": "Nubia RedMagic 9 Pro", "class": "Геймерський",
            "price_usd": 750, "ram_gb": 16, "battery_mah": 6500, "camera_mp": 50, "screen_hz": 120, "weight_g": 229,
            "water_resistance": "Немає", "display_matrix": "AMOLED",
            "wireless_charging": "Ні", "active_cooling": "Так", "body_material": "Алюміній"
        },
        {
            "name": "Black Shark 5 Pro", "class": "Геймерський",
            "price_usd": 700, "ram_gb": 12, "battery_mah": 4650, "camera_mp": 108, "screen_hz": 144, "weight_g": 220,
            "water_resistance": "Немає", "display_matrix": "AMOLED",
            "wireless_charging": "Ні", "active_cooling": "Так", "body_material": "Скло"
        }
    ]

    model.fit(training_data)
    return model


def get_test_samples() -> list:
    """Тестові контрольні зразки для перевірки."""
    return [
        {
            "name": "Невідомий зразок A (Бюджетний сегмент)",
            "expected": "Бюджетний",
            "sample": {
                "price_usd": 135, "ram_gb": 4, "battery_mah": 5000, "camera_mp": 50, "screen_hz": 90, "weight_g": 190,
                "water_resistance": "Немає", "display_matrix": "TFT/IPS",
                "wireless_charging": "Ні", "active_cooling": "Ні", "body_material": "Пластик"
            }
        },
        {
            "name": "Невідомий зразок B (Преміальний флагман)",
            "expected": "Флагман",
            "sample": {
                "price_usd": 1200, "ram_gb": 12, "battery_mah": 4800, "camera_mp": 50, "screen_hz": 120, "weight_g": 205,
                "water_resistance": "IP68 (занурення)", "display_matrix": "LTPO OLED",
                "wireless_charging": "Так", "active_cooling": "Ні", "body_material": "Титан/Скло"
            }
        },
        {
            "name": "Невідомий зразок C (Потужний геймерський з кулером)",
            "expected": "Геймерський",
            "sample": {
                "price_usd": 850, "ram_gb": 16, "battery_mah": 6000, "camera_mp": 50, "screen_hz": 144, "weight_g": 230,
                "water_resistance": "Немає", "display_matrix": "AMOLED",
                "wireless_charging": "Ні", "active_cooling": "Так", "body_material": "Алюміній"
            }
        },
        {
            "name": "Невідомий зразок D (Збалансований середній клас)",
            "expected": "Середній клас",
            "sample": {
                "price_usd": 360, "ram_gb": 8, "battery_mah": 5000, "camera_mp": 64, "screen_hz": 120, "weight_g": 185,
                "water_resistance": "IP54 (бризки)", "display_matrix": "AMOLED",
                "wireless_charging": "Ні", "active_cooling": "Ні", "body_material": "Скло"
            }
        }
    ]
