"""
main.py - Головна точка входу проєкту комп'ютерного зору (OpenCV Vision Studio)
Підтримує як сучасний графічний інтерфейс (GUI), так і повноцінний консольний режим (CLI).

Використання:
    python main.py                     # Запуск графічного інтерфейсу (GUI)
    python main.py --console           # Інтерактивне консольне меню
    python main.py --mode face         # Автоматичний тест детекції облич
    python main.py --mode qr           # Автоматичний тест розпізнавання QR-коду
    python main.py --mode shapes       # Автоматичний тест розпізнавання фігур
    python main.py --report            # Генерація офіційного Word-звіту (.docx)
"""

import sys
import os
import argparse
from vision_engine import VisionEngine
from generate_docx_report import create_lab_report


def run_cli_interactive():
    """Інтерактивне меню для роботи в терміналі."""
    engine = VisionEngine()
    samples_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "samples")
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    os.makedirs(output_dir, exist_ok=True)

    while True:
        print("\n" + "=" * 55)
        print("    СИСТЕМА МАШИННОГО ЗОРУ OPENCV (Лабораторна №5)")
        print("=" * 55)
        print("1. 👤 Детекція облич та очей (тестове фото)")
        print("2. 📱 Розпізнавання та декодування QR-коду")
        print("3. 🔷 Контурний аналіз: геометричні фігури та кольори")
        print("4. ⚡ Виділення меж алгоритмом Canny")
        print("5. 🖥️ Запустити графічний інтерфейс (GUI)")
        print("6. 📄 Згенерувати офіційний Word-звіт (.docx)")
        print("0. 🚪 Вихід")
        print("-" * 55)

        choice = input("Оберіть дію [0-6]: ").strip()

        if choice == "1":
            path = os.path.join(samples_dir, "face_sample.jpg")
            img = VisionEngine.read_image_file(path)
            if img is not None:
                res, stats = engine.detect_faces_and_eyes(img)
                out_path = os.path.join(output_dir, "cli_detected_face.png")
                VisionEngine.save_image_file(out_path, res)
                print(f"\n[+] {stats['message']}")
                print(f"[+] Результат збережено у: {out_path}")
            else:
                print(f"[!] Не знайдено {path}")

        elif choice == "2":
            path = os.path.join(samples_dir, "qr_sample.png")
            img = VisionEngine.read_image_file(path)
            if img is not None:
                res, stats = engine.detect_and_decode_qr(img)
                out_path = os.path.join(output_dir, "cli_detected_qr.png")
                VisionEngine.save_image_file(out_path, res)
                print(f"\n[+] {stats['message']}")
                for t in stats.get('qr_texts', []):
                    print(f"    Декодований вміст: {t}")
                print(f"[+] Результат збережено у: {out_path}")
            else:
                print(f"[!] Не знайдено {path}")

        elif choice == "3":
            path = os.path.join(samples_dir, "shapes_sample.png")
            img = VisionEngine.read_image_file(path)
            if img is not None:
                res, stats = engine.detect_shapes_and_colors(img)
                out_path = os.path.join(output_dir, "cli_detected_shapes.png")
                VisionEngine.save_image_file(out_path, res)
                print(f"\n[+] {stats['message']}:")
                for s in stats.get('shapes', []):
                    print(f"    • {s['shape']:<15} Колір: {s['color']:<12} Площа: {s['area']} px")
                print(f"[+] Результат збережено у: {out_path}")
            else:
                print(f"[!] Не знайдено {path}")

        elif choice == "4":
            path = os.path.join(samples_dir, "shapes_sample.png")
            img = VisionEngine.read_image_file(path)
            if img is not None:
                res, stats = engine.detect_edges_canny(img)
                out_path = os.path.join(output_dir, "cli_detected_canny.png")
                VisionEngine.save_image_file(out_path, res)
                print(f"\n[+] {stats['message']}")
                print(f"[+] Результат збережено у: {out_path}")
            else:
                print(f"[!] Не знайдено {path}")

        elif choice == "5":
            print("\n[+] Запуск графічного інтерфейсу...")
            from gui_app import run_gui
            run_gui()

        elif choice == "6":
            print("\n[+] Генерація звіту...")
            report_file = create_lab_report()
            print(f"[+] Звіт успішно згенеровано: {report_file}")

        elif choice == "0":
            print("\nРоботу завершено. До побачення!")
            break
        else:
            print("[!] Невірний вибір. Спробуйте ще раз.")


def main():
    parser = argparse.ArgumentParser(description="OpenCV Vision Studio (Лабораторна робота 5)")
    parser.add_argument("--console", action="store_true", help="Запустити інтерактивне консольне меню")
    parser.add_argument("--mode", choices=["face", "qr", "shapes", "canny"], help="Режим автономної обробки")
    parser.add_argument("--source", type=str, help="Шлях до вхідного зображення")
    parser.add_argument("--report", action="store_true", help="Згенерувати звіт Word (.docx)")

    args = parser.parse_args()

    if args.report:
        create_lab_report()
        return

    if args.console:
        run_cli_interactive()
        return

    if args.mode:
        engine = VisionEngine()
        src_path = args.source
        if not src_path:
            # Дефолтний вибір зразка
            defaults = {
                "face": "samples/face_sample.jpg",
                "qr": "samples/qr_sample.png",
                "shapes": "samples/shapes_sample.png",
                "canny": "samples/shapes_sample.png"
            }
            src_path = defaults.get(args.mode)

        img = VisionEngine.read_image_file(src_path)
        if img is None:
            print(f"[!] Помилка відкриття файлу: {src_path}")
            sys.exit(1)

        out_name = f"cli_{args.mode}_result.png"
        out_path = os.path.join("output", out_name)

        if args.mode == "face":
            res, stats = engine.detect_faces_and_eyes(img)
        elif args.mode == "qr":
            res, stats = engine.detect_and_decode_qr(img)
        elif args.mode == "shapes":
            res, stats = engine.detect_shapes_and_colors(img)
        elif args.mode == "canny":
            res, stats = engine.detect_edges_canny(img)

        VisionEngine.save_image_file(out_path, res)
        print(f"[+] Результат обробки ({args.mode}): {stats.get('message')}")
        print(f"[+] Збережено у: {out_path}")
        return

    # За замовчуванням запускаємо графічний інтерфейс (GUI)
    from gui_app import run_gui
    run_gui()


if __name__ == "__main__":
    main()
