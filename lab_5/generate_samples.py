"""
generate_samples.py - Скрипт для генерації та підготовки тестових зразків
Створює папку samples/ з тестовими зображеннями:
 - shapes_sample.png (геометричні фігури та кольори)
 - qr_sample.png (QR-код з посиланням та текстом)
 - face_sample.jpg (тестове фото для детекції обличчя)
"""

import os
import urllib.request
import cv2
import numpy as np
import qrcode


def safe_imwrite(filepath: str, img: np.ndarray, ext: str = ".png"):
    """Записує зображення у файл із підтримкою кириличних шляхів Windows."""
    success, encoded = cv2.imencode(ext, img)
    if success:
        with open(filepath, "wb") as f:
            encoded.tofile(f)
        return True
    return False


def create_shapes_sample(filepath: str):
    """Створює тестове зображення з кольоровими фігурами."""
    width, height = 800, 600
    # Світло-сіре тло
    img = np.full((height, width, 3), 245, dtype=np.uint8)

    # 1. Червоне коло (Red Circle)
    cv2.circle(img, (180, 160), 75, (40, 40, 225), -1)
    cv2.circle(img, (180, 160), 75, (20, 20, 150), 2)

    # 2. Синій прямокутник (Blue Rectangle)
    cv2.rectangle(img, (350, 90), (550, 220), (220, 120, 20), -1)
    cv2.rectangle(img, (350, 90), (550, 220), (160, 80, 10), 2)

    # 3. Зелений трикутник (Green Triangle)
    triangle_pts = np.array([[680, 230], [600, 100], [760, 100]], np.int32)
    cv2.fillPoly(img, [triangle_pts], (50, 180, 40))
    cv2.polylines(img, [triangle_pts], True, (30, 120, 25), 2)

    # 4. Фіолетовий квадрат (Purple Square)
    cv2.rectangle(img, (120, 360), (260, 500), (180, 40, 160), -1)
    cv2.rectangle(img, (120, 360), (260, 500), (120, 20, 110), 2)

    # 5. Оранжевий п'ятикутник (Orange Pentagon)
    pentagon_pts = np.array([
        [450, 330],
        [540, 395],
        [505, 500],
        [395, 500],
        [360, 395]
    ], np.int32)
    cv2.fillPoly(img, [pentagon_pts], (20, 140, 240))
    cv2.polylines(img, [pentagon_pts], True, (10, 90, 180), 2)

    # 6. Жовте коло (Yellow Circle)
    cv2.circle(img, (670, 420), 65, (30, 210, 240), -1)
    cv2.circle(img, (670, 420), 65, (20, 160, 190), 2)

    # Заголовок
    cv2.putText(img, "Computer Vision Test Sample: Geometric Shapes & Colors",
                (30, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 50, 60), 2, cv2.LINE_AA)

    safe_imwrite(filepath, img, ".png")
    print(f"[+] Створено зразок фігур: {filepath}")


def create_qr_sample(filepath: str):
    """Створює валідний тестовий QR-код."""
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_H,
        box_size=12,
        border=4,
    )
    qr.add_data("https://github.com/opencv/opencv | Computer Vision Lab 5 - Done!")
    qr.make(fit=True)

    img = qr.make_image(fill_color="black", back_color="white")
    img.save(filepath)
    print(f"[+] Створено зразок QR-коду: {filepath}")


def download_or_create_face_sample(filepath: str):
    """Завантажує класичне фото або створює резервне фото обличчя."""
    if os.path.exists(filepath) and os.path.getsize(filepath) > 1000:
        print(f"[+] Фото обличчя вже присутнє: {filepath}")
        return

    url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/lena.jpg"
    try:
        urllib.request.urlretrieve(url, filepath)
        print(f"[+] Завантажено еталонне тестове фото обличчя: {filepath}")
    except Exception as e:
        print(f"[!] Не вдалося завантажити фото з мережі ({e}), створюємо синтетичний зразок...")
        img = np.full((512, 512, 3), 230, dtype=np.uint8)
        cv2.circle(img, (256, 256), 140, (180, 200, 230), -1)
        cv2.circle(img, (256, 256), 140, (100, 100, 120), 3)
        cv2.circle(img, (210, 220), 22, (255, 255, 255), -1)
        cv2.circle(img, (210, 220), 10, (150, 70, 30), -1)
        cv2.circle(img, (302, 220), 22, (255, 255, 255), -1)
        cv2.circle(img, (302, 220), 10, (150, 70, 30), -1)
        cv2.line(img, (256, 235), (256, 280), (120, 120, 140), 3)
        cv2.ellipse(img, (256, 310), (55, 30), 0, 0, 180, (40, 40, 200), 3)
        safe_imwrite(filepath, img, ".jpg")
        print(f"[+] Створено синтетичний зразок обличчя: {filepath}")


def main():
    samples_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "samples")
    os.makedirs(samples_dir, exist_ok=True)

    shapes_path = os.path.join(samples_dir, "shapes_sample.png")
    qr_path = os.path.join(samples_dir, "qr_sample.png")
    face_path = os.path.join(samples_dir, "face_sample.jpg")

    create_shapes_sample(shapes_path)
    create_qr_sample(qr_path)
    download_or_create_face_sample(face_path)

    print("\nУсі зразки успішно згенеровано в папці:", samples_dir)


if __name__ == "__main__":
    main()
