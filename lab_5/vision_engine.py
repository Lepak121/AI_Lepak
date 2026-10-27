"""
vision_engine.py - Модуль комп'ютерного зору на базі OpenCV
Реалізує:
 1. Детекція облич та очей (Haar Cascades)
 2. Розпізнавання та декодування QR-кодів (QRCodeDetector)
 3. Розпізнавання геометричних фігур та кольорів (Contours & HSV analysis)
 4. Детекція руху (Frame Differencing & Motion Contours)
 5. Виділення меж та контурів (Canny Edge Detection)
"""

import os
import tempfile
import cv2
import numpy as np
from typing import Dict, Any, Tuple, List, Optional


def load_cascade_safe(cascade_name: str) -> cv2.CascadeClassifier:
    """
    Безпечно завантажує каскад Хаара.
    Якщо шлях містить не-ASCII символи (наприклад, кирилицю 'Робочий стіл'),
    копіює XML-файл у системний temp-каталог для коректного завантаження C++ ядром OpenCV.
    """
    cascade_src = os.path.join(cv2.data.haarcascades, cascade_name)
    has_non_ascii = any(ord(c) > 127 for c in cascade_src)

    if not has_non_ascii:
        classifier = cv2.CascadeClassifier(cascade_src)
        if not classifier.empty():
            return classifier

    try:
        temp_dir = tempfile.gettempdir()
        temp_cascade_path = os.path.join(temp_dir, cascade_name)
        if not os.path.exists(temp_cascade_path) or os.path.getsize(temp_cascade_path) == 0:
            with open(cascade_src, 'rb') as f_in, open(temp_cascade_path, 'wb') as f_out:
                f_out.write(f_in.read())
        return cv2.CascadeClassifier(temp_cascade_path)
    except Exception as e:
        print(f"[!] Помилка завантаження каскаду {cascade_name}: {e}")
        return cv2.CascadeClassifier()


class VisionEngine:
    def __init__(self):
        # Завантаження каскадів облич та очей
        self.face_cascade = load_cascade_safe('haarcascade_frontalface_default.xml')
        self.eye_cascade = load_cascade_safe('haarcascade_eye.xml')

        # Детектор та декодер QR-кодів
        self.qr_detector = cv2.QRCodeDetector()

        # Попередній кадр для детекції руху
        self.prev_motion_gray = None

        # Палітра кольорів у форматі BGR для візуалізації
        self.COLORS = {
            'emerald': (80, 200, 120),
            'cyan': (255, 220, 0),
            'purple': (230, 80, 180),
            'amber': (30, 180, 255),
            'red': (60, 60, 240),
            'white': (255, 255, 255),
            'dark_bg': (24, 24, 28)
        }

    # -------------------------------------------------------------------------
    # Робота з файлами зображень (Unicode Safe)
    # -------------------------------------------------------------------------
    @staticmethod
    def read_image_file(filepath: str) -> Optional[np.ndarray]:
        """Безпечне зчитування зображення навіть із кириличного шляху."""
        try:
            with open(filepath, 'rb') as f:
                data = bytearray(f.read())
            np_arr = np.asarray(data, dtype=np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            return img
        except Exception as e:
            print(f"[!] Помилка зчитування файлу {filepath}: {e}")
            return None

    @staticmethod
    def save_image_file(filepath: str, img: np.ndarray, ext: str = ".png") -> bool:
        """Безпечне збереження зображення у файл навіть із кириличним шляхом."""
        try:
            os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
            success, encoded = cv2.imencode(ext, img)
            if success:
                with open(filepath, 'wb') as f:
                    encoded.tofile(f)
                return True
        except Exception as e:
            print(f"[!] Помилка збереження файлу {filepath}: {e}")
        return False

    # -------------------------------------------------------------------------
    # 1. Детекція облич та очей
    # -------------------------------------------------------------------------
    def detect_faces_and_eyes(self, frame: np.ndarray, scale_factor: float = 1.15, min_neighbors: int = 5) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Знаходить обличчя та очі за допомогою класифікаторів Хаара.
        Повертає оброблений кадр та статистику виявлення.
        """
        output = frame.copy()
        if self.face_cascade.empty():
            return output, {
                'mode': 'Face & Eye Detection',
                'faces_count': 0,
                'eyes_count': 0,
                'message': 'Помилка: каскад облич не завантажено'
            }

        gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=max(1.05, float(scale_factor)),
            minNeighbors=int(min_neighbors),
            minSize=(30, 30)
        )

        detected_eyes_total = 0
        details = []

        for idx, (x, y, w, h) in enumerate(faces):
            # Рамка для обличчя
            cv2.rectangle(output, (x, y), (x + w, y + h), self.COLORS['emerald'], 2)
            
            # Підпис
            label = f"Face #{idx + 1} ({w}x{h}px)"
            self._draw_label(output, label, x, y - 10, self.COLORS['emerald'])

            # Пошук очей у верхній зоні обличчя
            roi_gray = gray[y:y + int(h * 0.7), x:x + w]
            roi_color = output[y:y + int(h * 0.7), x:x + w]
            
            eyes = []
            if not self.eye_cascade.empty():
                eyes = self.eye_cascade.detectMultiScale(
                    roi_gray,
                    scaleFactor=1.1,
                    minNeighbors=3,
                    minSize=(15, 15),
                    maxSize=(int(w * 0.35), int(h * 0.35))
                )
            detected_eyes_total += len(eyes)

            for (ex, ey, ew, eh) in eyes:
                center = (ex + ew // 2, ey + eh // 2)
                radius = int(round((ew + eh) * 0.25))
                cv2.circle(roi_color, center, radius, self.COLORS['cyan'], 2)
                cv2.circle(roi_color, center, 2, (0, 0, 255), -1)

            details.append({
                'id': idx + 1,
                'bbox': (int(x), int(y), int(w), int(h)),
                'eyes_count': len(eyes)
            })

        stats = {
            'mode': 'Face & Eye Detection',
            'faces_count': len(faces),
            'eyes_count': detected_eyes_total,
            'details': details,
            'message': f"Виявлено облич: {len(faces)}, очей: {detected_eyes_total}"
        }
        return output, stats

    # -------------------------------------------------------------------------
    # 2. Розпізнавання та декодування QR-кодів
    # -------------------------------------------------------------------------
    def detect_and_decode_qr(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Знаходить та розшифровує QR-коди в кадрі.
        Повертає оброблений кадр та текстовий вміст коду.
        """
        output = frame.copy()
        qr_results = []

        try:
            retval, decoded_info, points, _ = self.qr_detector.detectAndDecodeMulti(output)
            if retval and points is not None and len(points) > 0:
                for i, pts in enumerate(points):
                    pts = pts.astype(int)
                    for j in range(4):
                        pt1 = tuple(pts[j])
                        pt2 = tuple(pts[(j + 1) % 4])
                        cv2.line(output, pt1, pt2, self.COLORS['purple'], 3)
                    for pt in pts:
                        cv2.circle(output, tuple(pt), 4, self.COLORS['amber'], -1)

                    text = decoded_info[i] if i < len(decoded_info) else ""
                    if text:
                        qr_results.append(text)
                        label = f"QR: {text[:28]}..." if len(text) > 28 else f"QR: {text}"
                        self._draw_label(output, label, pts[0][0], max(20, pts[0][1] - 10), self.COLORS['purple'])
        except Exception:
            pass

        # Одиночний детектор як fallback
        if not qr_results:
            try:
                text, points_single, _ = self.qr_detector.detectAndDecode(output)
                if points_single is not None and len(points_single) > 0:
                    pts = points_single[0].astype(int)
                    for j in range(4):
                        pt1 = tuple(pts[j])
                        pt2 = tuple(pts[(j + 1) % 4])
                        cv2.line(output, pt1, pt2, self.COLORS['purple'], 3)
                    for pt in pts:
                        cv2.circle(output, tuple(pt), 4, self.COLORS['amber'], -1)
                    if text:
                        qr_results.append(text)
                        label = f"QR: {text[:28]}..." if len(text) > 28 else f"QR: {text}"
                        self._draw_label(output, label, pts[0][0], max(20, pts[0][1] - 10), self.COLORS['purple'])
            except Exception:
                pass

        stats = {
            'mode': 'QR Code Recognition',
            'qr_count': len(qr_results),
            'qr_texts': qr_results,
            'message': f"Знайдено QR-кодів: {len(qr_results)}" + (f" -> '{qr_results[0]}'" if qr_results else "")
        }
        return output, stats

    # -------------------------------------------------------------------------
    # 3. Розпізнавання геометричних фігур та кольорів
    # -------------------------------------------------------------------------
    def detect_shapes_and_colors(self, frame: np.ndarray, min_area: float = 1200.0) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Знаходить геометричні фігури (трикутник, чотирикутник, круг, полігон)
        та визначає їхній колір за колірним простором HSV.
        """
        output = frame.copy()
        blurred = cv2.GaussianBlur(output, (5, 5), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)

        edges = cv2.Canny(gray, 40, 150)
        dilated = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)

        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        raw_shapes = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue

            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.035 * peri, True)
            num_vertices = len(approx)

            shape_name = "Невідомо"
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = float(w) / h if h != 0 else 0

            if num_vertices == 3:
                shape_name = "Трикутник"
            elif num_vertices == 4:
                if 0.88 <= aspect_ratio <= 1.12:
                    shape_name = "Квадрат"
                else:
                    shape_name = "Прямокутник"
            elif num_vertices == 5:
                shape_name = "П'ятикутник"
            elif num_vertices == 6:
                shape_name = "Шестикутник"
            else:
                circularity = (4 * np.pi * area) / (peri * peri) if peri > 0 else 0
                if circularity > 0.70:
                    shape_name = "Коло"
                else:
                    shape_name = f"Полігон ({num_vertices})"

            # Центр мас
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
            else:
                cX, cY = x + w // 2, y + h // 2

            cY_safe = max(0, min(output.shape[0] - 1, cY))
            cX_safe = max(0, min(output.shape[1] - 1, cX))
            h_val = int(hsv[cY_safe, cX_safe][0])
            s_val = int(hsv[cY_safe, cX_safe][1])
            v_val = int(hsv[cY_safe, cX_safe][2])
            color_name = self._classify_color(h_val, s_val, v_val)

            raw_shapes.append({
                'shape': shape_name,
                'color': color_name,
                'center': (cX, cY),
                'area': round(area, 1),
                'vertices': num_vertices,
                'contour': cnt
            })

        # Фільтрація дублікатів (якщо центри розташовані занадто близько)
        found_shapes = []
        for s in sorted(raw_shapes, key=lambda x: x['area'], reverse=True):
            cX, cY = s['center']
            is_dup = False
            for existing in found_shapes:
                eX, eY = existing['center']
                dist = np.hypot(cX - eX, cY - eY)
                if dist < 40:
                    is_dup = True
                    break
            if not is_dup:
                found_shapes.append(s)
                # Малюємо контур та мітку
                cv2.drawContours(output, [s['contour']], -1, self.COLORS['emerald'], 2)
                cv2.circle(output, (cX, cY), 4, self.COLORS['amber'], -1)
                tag = f"{s['shape']} | {s['color']}"
                self._draw_label(output, tag, cX - 35, cY - 10, self.COLORS['cyan'])

        # Прибираємо сам об'єкт контуру перед серіалізацією статистики
        for s in found_shapes:
            s.pop('contour', None)

        stats = {
            'mode': 'Shape & Color Recognition',
            'shapes_count': len(found_shapes),
            'shapes': found_shapes,
            'message': f"Знайдено фігур: {len(found_shapes)}"
        }
        return output, stats

    # -------------------------------------------------------------------------
    # 4. Детекція руху (Motion Detection)
    # -------------------------------------------------------------------------
    def detect_motion(self, frame: np.ndarray, threshold: int = 25, min_area: int = 500) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Знаходить ділянки руху за допомогою покадрової різниці (Frame Differencing).
        """
        output = frame.copy()
        gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        if self.prev_motion_gray is None or self.prev_motion_gray.shape != gray.shape:
            self.prev_motion_gray = gray
            stats = {
                'mode': 'Motion Detection',
                'motion_detected': False,
                'motion_zones': 0,
                'message': 'Ініціалізація базового кадру руху...'
            }
            return output, stats

        frame_diff = cv2.absdiff(self.prev_motion_gray, gray)
        thresh = cv2.threshold(frame_diff, threshold, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        motion_zones = 0
        total_motion_area = 0

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue

            motion_zones += 1
            total_motion_area += area
            x, y, w, h = cv2.boundingRect(cnt)
            
            cv2.rectangle(output, (x, y), (x + w, y + h), self.COLORS['red'], 2)
            cv2.putText(output, "MOTION", (x, max(15, y - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, self.COLORS['red'], 1, cv2.LINE_AA)

        cv2.accumulateWeighted(gray, self.prev_motion_gray.astype(np.float32), 0.2)
        self.prev_motion_gray = cv2.convertScaleAbs(self.prev_motion_gray)

        is_moving = motion_zones > 0
        status_text = f"Рух виявлено! Зон: {motion_zones} (Площа: {int(total_motion_area)})" if is_moving else "Руху не зафіксовано"

        stats = {
            'mode': 'Motion Detection',
            'motion_detected': is_moving,
            'motion_zones': motion_zones,
            'total_area': total_motion_area,
            'message': status_text
        }
        return output, stats

    def reset_motion(self):
        """Скидає попередній кадр руху."""
        self.prev_motion_gray = None

    # -------------------------------------------------------------------------
    # 5. Виділення меж Canny
    # -------------------------------------------------------------------------
    def detect_edges_canny(self, frame: np.ndarray, thresh1: int = 50, thresh2: int = 150) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Застосовує детектор меж Canny.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, thresh1, thresh2)

        colored_edges = np.zeros_like(frame)
        colored_edges[edges != 0] = self.COLORS['cyan']

        edge_pixels = int(np.count_nonzero(edges))
        density = round((edge_pixels / edges.size) * 100, 2)

        stats = {
            'mode': 'Canny Edge Detection',
            'edge_pixels': edge_pixels,
            'density_percent': density,
            'thresh1': thresh1,
            'thresh2': thresh2,
            'message': f"Межі Канні: {edge_pixels} точок ({density}% кадру)"
        }
        return colored_edges, stats

    # -------------------------------------------------------------------------
    # Допоміжні методи
    # -------------------------------------------------------------------------
    def _classify_color(self, h: int, s: int, v: int) -> str:
        """Класифікує назву кольору за значеннями Hue, Saturation, Value."""
        if v < 40:
            return "Чорний"
        if s < 35 and v > 180:
            return "Білий"
        if s < 45:
            return "Сірий"

        if (0 <= h <= 10) or (165 <= h <= 179):
            return "Червоний"
        elif 11 <= h <= 25:
            return "Оранжевий"
        elif 26 <= h <= 35:
            return "Жовтий"
        elif 36 <= h <= 85:
            return "Зелений"
        elif 86 <= h <= 130:
            return "Синій"
        elif 131 <= h <= 164:
            return "Фіолетовий"
        return "Кольоровий"

    def _draw_label(self, img: np.ndarray, text: str, x: int, y: int, color_bgr: Tuple[int, int, int]):
        """Малює елегантну підкладку під текст для чіткої читабельності."""
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.48
        thickness = 1
        (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
        
        y_top = max(0, y - th - 6)
        x_left = max(0, x - 2)
        y_bot = min(img.shape[0] - 1, y + baseline + 2)
        x_right = min(img.shape[1] - 1, x + tw + 6)

        sub_img = img[y_top:y_bot, x_left:x_right]
        if sub_img.shape[0] > 0 and sub_img.shape[1] > 0:
            black_rect = np.full_like(sub_img, self.COLORS['dark_bg'])
            blended = cv2.addWeighted(sub_img, 0.25, black_rect, 0.75, 0)
            img[y_top:y_bot, x_left:x_right] = blended

        cv2.putText(img, text, (x_left + 4, y_top + th + 2), font, scale, color_bgr, thickness, cv2.LINE_AA)
