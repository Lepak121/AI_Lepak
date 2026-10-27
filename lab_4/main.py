#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Лабораторна робота: Розпізнавання образів за кількісними та якісними характеристиками
Предметна область: Класифікація категорій смартфонів / мобільних девайсів
Метрики: Евклідова (L2) та Мангеттенська (L1) відстані
"""

import sys

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

from dataset import get_smartphones_model, get_test_samples


def print_header(title: str):
    line = "=" * 78
    print(f"\n{line}")
    print(f" {title.center(76)} ")
    print(f"{line}\n")


def print_features_info(model):
    print("[i] СПИСОК ОЗНАК (ХАРАКТЕРИСТИК) ОБРАЗІВ:")
    print("-" * 78)
    print(f"{'No':<3} {'Назва ознаки':<20} {'Тип ознаки':<22} {'Вага':<6} {'Опис / Шкала'}")
    print("-" * 78)

    type_names = {
        'quantitative': 'Кількісна (Min-Max)',
        'binary': 'Якісна (бінарна)',
        'ordinal': 'Якісна (порядкова)',
        'nominal': 'Якісна (номінальна)'
    }

    for idx, (fname, feat) in enumerate(model.features.items(), start=1):
        ftype_str = type_names.get(feat.ftype, feat.ftype)
        if feat.ftype == 'quantitative':
            desc = f"min={feat.min_val:.0f}, max={feat.max_val:.0f} {feat.unit}"
        elif feat.ftype == 'ordinal':
            desc = " -> ".join(feat.order_levels)
        elif feat.ftype == 'binary':
            desc = "Ні (0) / Так (1)"
        else:
            desc = "Категоріальні значення"

        if len(desc) > 28:
            desc = desc[:25] + "..."
        print(f"{idx:<3} {fname:<20} {ftype_str:<22} {feat.weight:<6.1f} {desc}")
    print("-" * 78)


def display_classification_result(sample_name: str, sample_data: dict, model, k: int = 3):
    print(f"\n[>] АНАЛІЗ ЗРАЗКА: {sample_name}")
    print("Вхідні характеристики:")
    for k_feat, val in sample_data.items():
        feat = model.features.get(k_feat)
        unit = f" {feat.unit}" if feat and feat.unit else ""
        print(f"  * {k_feat}: {val}{unit}")

    # Розрахунок обох метрик
    res_euc = model.classify_by_centroid(sample_data, metric='euclidean')
    res_euc_knn = model.classify_knn(sample_data, k=k, metric='euclidean')

    res_man = model.classify_by_centroid(sample_data, metric='manhattan')
    res_man_knn = model.classify_knn(sample_data, k=k, metric='manhattan')

    print("\n[+] ПОРІВНЯННЯ МЕТРИК ВІДСТАНІ ДО КЛАСІВ (ЕВКЛІД vs МАНГЕТТЕН):")
    print("-" * 78)
    print(f"{'Клас':<16} {'Евклід (L2)':<14} {'Впевн. (L2)':<12} {'Мангеттен (L1)':<16} {'Впевн. (L1)':<12}")
    print("-" * 78)

    for c in model.classes:
        e_d = res_euc['distances'][c]
        e_c = res_euc['confidence'][c]
        m_d = res_man['distances'][c]
        m_c = res_man['confidence'][c]

        e_mark = "*" if c == res_euc['predicted_class'] else " "
        m_mark = "*" if c == res_man['predicted_class'] else " "
        print(f"{c:<16} {e_d:<14.4f} {e_c:>6.1f}% {e_mark}    {m_d:<16.4f} {m_c:>6.1f}% {m_mark}")
    print("-" * 78)

    print(f"\n[ПІДСУМОК РОЗПІЗНАВАННЯ]")
    print(f"  >> Евклідова міра (L2):      Центроїди -> [{res_euc['predicted_class']}],  k-NN -> [{res_euc_knn['predicted_class']}]")
    print(f"  >> Мангеттенська міра (L1):  Центроїди -> [{res_man['predicted_class']}],  k-NN -> [{res_man_knn['predicted_class']}]")

    print("\n[k-NN] Найближчі сусіди (за мірою Евкліда):")
    for i, neighbor in enumerate(res_euc_knn['k_nearest'], start=1):
        print(f"     {i}. {neighbor['name']:<24} (Клас: {neighbor['class']:<15}, d = {neighbor['distance']:.4f})")


def run_automated_tests(model):
    print_header("АВТОМАТИЧНЕ ТЕСТУВАННЯ СИСТЕМИ НА КОНТРОЛЬНИХ ЗРАЗКАХ СМАРТФОНІВ")
    tests = get_test_samples()

    correct_euc = 0
    correct_man = 0
    total = len(tests)

    for test in tests:
        name = test['name']
        expected = test['expected']
        data = test['sample']

        res_euc = model.classify_by_centroid(data, metric='euclidean')
        res_man = model.classify_by_centroid(data, metric='manhattan')

        is_e_ok = (res_euc['predicted_class'] == expected)
        is_m_ok = (res_man['predicted_class'] == expected)

        if is_e_ok:
            correct_euc += 1
        if is_m_ok:
            correct_man += 1

        display_classification_result(name, data, model, k=3)
        print(f"Очікуваний клас: [{expected}] | Евклід L2: {'[OK]' if is_e_ok else '[FAIL]'} | Мангеттен L1: {'[OK]' if is_m_ok else '[FAIL]'}")
        print("=" * 78)

    print("\n[ПІДСУМКОВА ТОЧНІСТЬ ACCURACY]")
    print(f"  * Евклідова метрика (L2):     {correct_euc}/{total} ({correct_euc/total*100:.1f}%)")
    print(f"  * Мангеттенська метрика (L1): {correct_man}/{total} ({correct_man/total*100:.1f}%)")


def run_interactive_mode(model):
    print_header("ІНТЕРАКТИВНЕ РОЗПІЗНАВАННЯ ВЛАСНОГО СМАРТФОНА")
    print("Введіть характеристики смартфона або натисніть Enter для значення за замовчуванням:")

    sample = {}
    for fname, feat in model.features.items():
        if feat.ftype == 'quantitative':
            default_val = (feat.min_val + feat.max_val) / 2
            prompt = f"  Введіть {fname} ({feat.min_val:.0f} - {feat.max_val:.0f} {feat.unit}) [{default_val:.0f}]: "
            user_in = input(prompt).strip()
            sample[fname] = float(user_in) if user_in else default_val
        elif feat.ftype == 'ordinal':
            opts = ", ".join(feat.order_levels)
            prompt = f"  Виберіть {fname} ({opts}) ['{feat.order_levels[0]}']: "
            user_in = input(prompt).strip()
            sample[fname] = user_in if user_in in feat.order_levels else feat.order_levels[0]
        elif feat.ftype == 'binary':
            prompt = f"  Вкажіть {fname} (Так / Ні) ['Ні']: "
            user_in = input(prompt).strip()
            sample[fname] = user_in if user_in else "Ні"
        elif feat.ftype == 'nominal':
            prompt = f"  Введіть {fname} (Пластик, Алюміній, Скло, Титан/Скло) ['Пластик']: "
            user_in = input(prompt).strip()
            sample[fname] = user_in if user_in else "Пластик"

    display_classification_result("Введений користувачем смартфон", sample, model)


def main():
    model = get_smartphones_model()

    print_header("СИСТЕМА РОЗПІЗНАВАННЯ ОБРАЗІВ СМАРТФОНІВ (КІЛЬКІСНІ ТА ЯКІСНІ ОЗНАКИ)")
    print(f"Модель: {model.name}")
    print(f"Кількість навчальних еталонів у базі: {len(model.training_data)}")
    print(f"Класи образів: {', '.join(model.classes)}")
    print_features_info(model)

    while True:
        print("\nОБЕРІТЬ РЕЖИМ РОБОТИ:")
        print("  1. Запустити автоматичне тестування на контрольних зразках")
        print("  2. Ввести власний смартфон вручну для розпізнавання")
        print("  3. Показати навчальну вибірку (еталони)")
        print("  4. Запустити графічний інтерфейс (GUI на Tkinter)")
        print("  0. Вихід")

        try:
            choice = input("\nВаш вибір (0-4): ").strip()
        except EOFError:
            break

        if choice == '1':
            run_automated_tests(model)
        elif choice == '2':
            run_interactive_mode(model)
        elif choice == '3':
            print("\n[+] НАВЧАЛЬНА ВИБІРКА СМАРТФОНІВ (БАЗА ЗНАНЬ):")
            print("-" * 78)
            for idx, item in enumerate(model.training_data, start=1):
                print(f"{idx:2d}. [{item['class']:<15}] {item.get('name', 'Зразок'):<22} | "
                      f"{item['price_usd']}$ | RAM: {item['ram_gb']}GB | Батарея: {item['battery_mah']}mAh | "
                      f"Вологозахист: {item['water_resistance']} | Матеріал: {item['body_material']}")
            print("-" * 78)
        elif choice == '4':
            try:
                import gui_app
                print("Запуск графічного інтерфейсу...")
                gui_app.launch_gui()
            except Exception as e:
                print(f"Помилка запуску GUI: {e}")
        elif choice == '0':
            print("\nРоботу завершено. Дякуємо за використання програми!")
            break
        else:
            print("Невірний вибір, спробуйте ще раз.")


if __name__ == "__main__":
    main()
