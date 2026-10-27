"""
gui_app.py - Графічний інтерфейс додатку машинного зору (OpenCV Vision Studio)
Побудований на базі Tkinter та Pillow у сучасній темній стилістиці.
"""

import os
import sys
import time
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk

from vision_engine import VisionEngine


class VisionApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("OpenCV Vision Studio - Лабораторна робота №5")
        self.root.geometry("1160x780")
        self.root.minsize(980, 680)
        self.root.configure(bg="#18181b")

        # Ініціалізація ядра машинного зору
        self.engine = VisionEngine()

        # Змінні стану
        self.current_frame = None          # Оригінальний кадр (BGR)
        self.processed_frame = None        # Оброблений кадр (BGR)
        self.cap = None                    # VideoCapture об'єкт
        self.is_video_running = False      # Прапорець роботи відеопотоку
        self.is_paused = False             # Прапорець паузи
        self.source_type = "none"          # "webcam", "image", "video", "sample"
        self.video_thread = None

        # Метрики продуктивності
        self.fps = 0.0
        self.frame_count = 0
        self.last_time = time.time()

        # Режим детекції: "face", "qr", "shapes", "motion", "canny", "none"
        self.selected_mode = tk.StringVar(value="face")

        # Налаштування стилів
        self._setup_styles()

        # Побудова інтерфейсу
        self._build_ui()

        # Завантажуємо стартовий тестовий зразок
        self._load_default_sample()

        # Обробка закриття вікна
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def _setup_styles(self):
        style = ttk.Style()
        style.theme_use("clam")

        # Колірна палітра
        self.c_bg = "#18181b"          # Основний темний фон
        self.c_panel = "#27272a"       # Панелі та картки
        self.c_card = "#3f3f46"        # Виділені контейнери
        self.c_accent = "#10b981"      # Смарагдовий акцент
        self.c_cyan = "#06b6d4"        # Блакитний акцент
        self.c_purple = "#8b5cf6"      # Фіолетовий акцент
        self.c_text = "#f4f4f5"        # Основний світлий текст
        self.c_text_dim = "#a1a1aa"    # Приглушений текст

        style.configure(".", background=self.c_bg, foreground=self.c_text, font=("Segoe UI", 10))
        style.configure("TFrame", background=self.c_bg)
        style.configure("Card.TFrame", background=self.c_panel, relief="flat")
        style.configure("TLabelframe", background=self.c_panel, foreground=self.c_text, bordercolor=self.c_card)
        style.configure("TLabelframe.Label", background=self.c_panel, foreground=self.c_cyan, font=("Segoe UI", 10, "bold"))

        style.configure("TLabel", background=self.c_panel, foreground=self.c_text)
        style.configure("Dim.TLabel", background=self.c_panel, foreground=self.c_text_dim, font=("Segoe UI", 9))
        style.configure("Title.TLabel", background=self.c_bg, foreground="#ffffff", font=("Segoe UI", 14, "bold"))
        style.configure("Status.TLabel", background="#09090b", foreground=self.c_accent, font=("Consolas", 10))

        style.configure("TRadiobutton", background=self.c_panel, foreground=self.c_text, font=("Segoe UI", 10))
        style.map("TRadiobutton",
                  background=[("active", self.c_panel)],
                  foreground=[("active", self.c_accent)])

        style.configure("TButton", background="#3b82f6", foreground="#ffffff", borderwidth=0, font=("Segoe UI", 10, "bold"), padding=6)
        style.map("TButton",
                  background=[("active", "#2563eb"), ("disabled", "#52525b")])

        style.configure("Green.TButton", background="#10b981", foreground="#ffffff", font=("Segoe UI", 10, "bold"), padding=6)
        style.map("Green.TButton", background=[("active", "#059669")])

        style.configure("Purple.TButton", background="#8b5cf6", foreground="#ffffff", font=("Segoe UI", 10, "bold"), padding=6)
        style.map("Purple.TButton", background=[("active", "#7c3aed")])

    def _build_ui(self):
        # Головний контейнер
        main_box = ttk.Frame(self.root, padding=12)
        main_box.pack(fill=tk.BOTH, expand=True)

        # -------------------------------------------------------------
        # Ліва колонка: Екран перегляду (Viewport) + Статус-бар
        # -------------------------------------------------------------
        left_col = ttk.Frame(main_box)
        left_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        # Заголовок та метадані потоку
        header_bar = ttk.Frame(left_col)
        header_bar.pack(fill=tk.X, pady=(0, 8))

        self.lbl_title = ttk.Label(header_bar, text="👁️ Live Vision Canvas", style="Title.TLabel")
        self.lbl_title.pack(side=tk.LEFT)

        self.lbl_fps = tk.Label(header_bar, text="FPS: --", font=("Consolas", 10, "bold"),
                                bg="#27272a", fg="#10b981", padx=8, pady=3, relief="flat")
        self.lbl_fps.pack(side=tk.RIGHT, padx=4)

        self.lbl_resolution = tk.Label(header_bar, text="0 x 0", font=("Consolas", 10),
                                       bg="#27272a", fg="#a1a1aa", padx=8, pady=3, relief="flat")
        self.lbl_resolution.pack(side=tk.RIGHT)

        # Полотно відображення (Canvas)
        self.canvas_frame = tk.Frame(left_col, bg="#09090b", bd=1, relief="solid")
        self.canvas_frame.pack(fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(self.canvas_frame, bg="#09090b", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Нижня стрічка повідомлень
        self.status_bar = tk.Label(left_col, text="Готово до роботи. Оберіть джерело та режим детекції.",
                                   bg="#09090b", fg="#e4e4e7", font=("Segoe UI", 10),
                                   anchor="w", padx=10, pady=6)
        self.status_bar.pack(fill=tk.X, pady=(8, 0))

        # -------------------------------------------------------------
        # Права колонка: Панель налаштувань та результатів
        # -------------------------------------------------------------
        right_col = ttk.Frame(main_box, width=380)
        right_col.pack(side=tk.RIGHT, fill=tk.Y)
        right_col.pack_propagate(False)

        # Блок 1: Джерело вхідних даних (Source Selection)
        src_box = ttk.LabelFrame(right_col, text=" 📥 Джерело вхідних даних ", padding=10)
        src_box.pack(fill=tk.X, pady=(0, 10))

        btn_row1 = ttk.Frame(src_box)
        btn_row1.pack(fill=tk.X, pady=(0, 6))

        self.btn_webcam = ttk.Button(btn_row1, text="📹 Веб-камера", style="Green.TButton", command=self.toggle_webcam)
        self.btn_webcam.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))

        self.btn_open_img = ttk.Button(btn_row1, text="📂 Відкрити фото", command=self.load_image_file)
        self.btn_open_img.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(4, 0))

        # Швидкі тестові зразки (Samples)
        lbl_samples = ttk.Label(src_box, text="Швидкі зразки (samples):", style="Dim.TLabel")
        lbl_samples.pack(anchor="w", pady=(4, 2))

        samples_row = ttk.Frame(src_box)
        samples_row.pack(fill=tk.X)

        ttk.Button(samples_row, text="👤 Обличчя", command=lambda: self.load_sample("face_sample.jpg")).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        ttk.Button(samples_row, text="📱 QR-код", command=lambda: self.load_sample("qr_sample.png")).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        ttk.Button(samples_row, text="🔷 Фігури", command=lambda: self.load_sample("shapes_sample.png")).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)

        # Блок 2: Режими розпізнавання (Vision Tasks)
        mode_box = ttk.LabelFrame(right_col, text=" 🎯 Режим розпізнавання ", padding=10)
        mode_box.pack(fill=tk.X, pady=(0, 10))

        modes = [
            ("👤 Детекція облич та очей", "face"),
            ("📱 Розпізнавання QR-кодів", "qr"),
            ("🔷 Фігури та колірний аналіз", "shapes"),
            ("🏃 Детекція руху (Motion)", "motion"),
            ("⚡ Виділення меж (Canny Edges)", "canny"),
            ("⚪ Початкове відео (Original)", "none")
        ]

        for text, mode_val in modes:
            rb = ttk.Radiobutton(mode_box, text=text, value=mode_val, variable=self.selected_mode, command=self.on_mode_change)
            rb.pack(anchor="w", pady=2)

        # Блок 3: Параметри та повзунки (Parameters)
        param_box = ttk.LabelFrame(right_col, text=" ⚙️ Налаштування параметрів ", padding=10)
        param_box.pack(fill=tk.X, pady=(0, 10))

        # Слайдер 1: Чутливість / Поріг 1
        self.lbl_slider1 = ttk.Label(param_box, text="Чутливість / Поріг: 1.15", style="Dim.TLabel")
        self.lbl_slider1.pack(anchor="w")

        self.slider1 = tk.Scale(param_box, from_=1.05, to=1.5, resolution=0.05, orient=tk.HORIZONTAL,
                                bg=self.c_panel, fg=self.c_text, highlightthickness=0,
                                troughcolor=self.c_card, activebackground=self.c_cyan,
                                command=self.on_slider1_change)
        self.slider1.set(1.15)
        self.slider1.pack(fill=tk.X, pady=(0, 6))

        # Слайдер 2: Мінімальна площа / Поріг 2
        self.lbl_slider2 = ttk.Label(param_box, text="Мін. площа / Поріг Canny: 1200", style="Dim.TLabel")
        self.lbl_slider2.pack(anchor="w")

        self.slider2 = tk.Scale(param_box, from_=100, to=3000, resolution=50, orient=tk.HORIZONTAL,
                                bg=self.c_panel, fg=self.c_text, highlightthickness=0,
                                troughcolor=self.c_card, activebackground=self.c_cyan,
                                command=self.on_slider2_change)
        self.slider2.set(1200)
        self.slider2.pack(fill=tk.X)

        # Блок 4: Інформація та результати (Live Telemetry)
        info_box = ttk.LabelFrame(right_col, text=" 📊 Результати розпізнавання ", padding=10)
        info_box.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        # Текстовий блок для виводу розпізнаних даних
        self.txt_results = tk.Text(info_box, height=8, bg="#111113", fg="#e4e4e7",
                                   insertbackground="#ffffff", font=("Consolas", 9),
                                   relief="flat", wrap=tk.WORD)
        self.txt_results.pack(fill=tk.BOTH, expand=True)

        # Блок 5: Кнопки дій (Actions)
        act_box = ttk.Frame(right_col)
        act_box.pack(fill=tk.X)

        self.btn_snapshot = ttk.Button(act_box, text="📸 Зберегти знімок", style="Purple.TButton", command=self.save_snapshot)
        self.btn_snapshot.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))

        self.btn_report = ttk.Button(act_box, text="📄 Звіт Word (.docx)", command=self.open_generate_report)
        self.btn_report.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(4, 0))

    # -------------------------------------------------------------------------
    # Завантаження зразків та зображень
    # -------------------------------------------------------------------------
    def _load_default_sample(self):
        sample_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "samples", "face_sample.jpg")
        if os.path.exists(sample_path):
            self.load_image_from_path(sample_path)

    def load_sample(self, filename: str):
        self.stop_video_stream()
        sample_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "samples", filename)
        if not os.path.exists(sample_path):
            messagebox.showwarning("Файл не знайдено", f"Зразок {filename} не знайдено в папці samples/")
            return

        # Автоматичне перемикання режиму під відповідний зразок
        if "face" in filename:
            self.selected_mode.set("face")
        elif "qr" in filename:
            self.selected_mode.set("qr")
        elif "shapes" in filename:
            self.selected_mode.set("shapes")

        self.load_image_from_path(sample_path)

    def load_image_file(self):
        self.stop_video_stream()
        file_path = filedialog.askopenfilename(
            title="Оберіть зображення",
            filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.bmp;*.webp"), ("All files", "*.*")]
        )
        if file_path:
            self.load_image_from_path(file_path)

    def load_image_from_path(self, path: str):
        img = VisionEngine.read_image_file(path)
        if img is None:
            messagebox.showerror("Помилка", f"Не вдалося прочитати зображення:\n{path}")
            return

        self.source_type = "image"
        self.current_frame = img
        self.status_bar.config(text=f"Завантажено зображення: {os.path.basename(path)}")
        self.process_and_render_frame()

    # -------------------------------------------------------------------------
    # Робота з веб-камерою
    # -------------------------------------------------------------------------
    def toggle_webcam(self):
        if self.is_video_running:
            self.stop_video_stream()
        else:
            self.start_webcam()

    def start_webcam(self):
        self.stop_video_stream()
        # Спроба відкриття камери 0 або DSHOW на Windows
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(0)

        if not self.cap.isOpened():
            messagebox.showerror("Камера не знайдена", "Не вдалося отримати доступ до веб-камери.\nПеревірте підключення або використовуйте тестові фото/зразки.")
            return

        # Налаштування розміру кадру
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        self.is_video_running = True
        self.source_type = "webcam"
        self.engine.reset_motion()
        self.btn_webcam.config(text="⏹️ Зупинити камеру", style="TButton")
        self.status_bar.config(text="Веб-камера підключена. Обробка відеопотоку...")

        self.last_time = time.time()
        self.frame_count = 0

        # Запуск фонового циклу захоплення кадру
        self.video_thread = threading.Thread(target=self._video_loop, daemon=True)
        self.video_thread.start()

    def stop_video_stream(self):
        self.is_video_running = False
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None

        self.btn_webcam.config(text="📹 Веб-камера", style="Green.TButton")
        self.lbl_fps.config(text="FPS: --")

    def _video_loop(self):
        while self.is_video_running and self.cap is not None and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret or frame is None:
                time.sleep(0.01)
                continue

            # Горизонтальне віддзеркалення для природного сприйняття камери
            self.current_frame = cv2.flip(frame, 1)

            # Підрахунок FPS
            self.frame_count += 1
            now = time.time()
            dt = now - self.last_time
            if dt >= 0.5:
                self.fps = round(self.frame_count / dt, 1)
                self.frame_count = 0
                self.last_time = now
                self.root.after(0, lambda: self.lbl_fps.config(text=f"FPS: {self.fps}"))

            # Обробка та оновлення інтерфейсу в головному потоці
            self.root.after(0, self.process_and_render_frame)
            time.sleep(0.02)  # обмеження ~30-40 FPS для збереження ресурсів

    # -------------------------------------------------------------------------
    # Обробка кадру ядром машинного зору
    # -------------------------------------------------------------------------
    def process_and_render_frame(self):
        if self.current_frame is None:
            return

        frame = self.current_frame.copy()
        mode = self.selected_mode.get()
        stats = {}

        if mode == "face":
            scale = float(self.slider1.get())
            min_neighbors = int(max(2, self.slider2.get() / 300))
            self.processed_frame, stats = self.engine.detect_faces_and_eyes(frame, scale_factor=scale, min_neighbors=min_neighbors)

        elif mode == "qr":
            self.processed_frame, stats = self.engine.detect_and_decode_qr(frame)

        elif mode == "shapes":
            min_area = float(self.slider2.get())
            self.processed_frame, stats = self.engine.detect_shapes_and_colors(frame, min_area=min_area)

        elif mode == "motion":
            thresh = int(self.slider1.get() * 20)
            min_area = int(self.slider2.get() / 2)
            self.processed_frame, stats = self.engine.detect_motion(frame, threshold=thresh, min_area=min_area)

        elif mode == "canny":
            t1 = int(self.slider1.get() * 40)
            t2 = int(self.slider2.get() / 10)
            self.processed_frame, stats = self.engine.detect_edges_canny(frame, thresh1=t1, thresh2=t2)

        else:
            # Режим "none" (Оригінал)
            self.processed_frame = frame
            stats = {'mode': 'Original (No Filter)', 'message': 'Оригінальний кадр без фільтрів'}

        # Оновлення статусу та результатів
        if 'message' in stats and self.source_type != "webcam":
            self.status_bar.config(text=stats['message'])

        self._update_telemetry(stats)
        self._render_canvas()

    def _update_telemetry(self, stats: dict):
        """Оновлює текстове поле результатів розпізнавання."""
        lines = []
        mode = stats.get('mode', '')
        lines.append(f"РЕЖИМ: {mode}")
        lines.append("-" * 34)

        if mode == 'Face & Eye Detection':
            faces = stats.get('faces_count', 0)
            eyes = stats.get('eyes_count', 0)
            lines.append(f"👤 Знайдено облич: {faces}")
            lines.append(f"👀 Знайдено очей:  {eyes}")
            for d in stats.get('details', []):
                x, y, w, h = d['bbox']
                lines.append(f" • Обличчя #{d['id']}: [{w}x{h}] у ({x},{y}), очей: {d['eyes_count']}")

        elif mode == 'QR Code Recognition':
            count = stats.get('qr_count', 0)
            lines.append(f"📱 Виявлено QR-кодів: {count}")
            for idx, text in enumerate(stats.get('qr_texts', [])):
                lines.append(f"Код #{idx + 1}:\n{text}")

        elif mode == 'Shape & Color Recognition':
            shapes = stats.get('shapes', [])
            lines.append(f"🔷 Всього фігур: {len(shapes)}")
            for idx, s in enumerate(shapes):
                lines.append(f" #{idx + 1}: {s['shape']} ({s['color']}) S={int(s['area'])}")

        elif mode == 'Motion Detection':
            detected = stats.get('motion_detected', False)
            status_str = "ТАК (Активність)" if detected else "НІ (Спокій)"
            lines.append(f"🏃 Рух: {status_str}")
            lines.append(f"Зон руху: {stats.get('motion_zones', 0)}")
            lines.append(f"Площа руху: {int(stats.get('total_area', 0))} px")

        elif mode == 'Canny Edge Detection':
            lines.append(f"Точок меж: {stats.get('edge_pixels', 0)}")
            lines.append(f"Щільність меж: {stats.get('density_percent', 0)}%")

        text_content = "\n".join(lines)
        self.txt_results.delete("1.0", tk.END)
        self.txt_results.insert(tk.END, text_content)

    def _render_canvas(self):
        if self.processed_frame is None:
            return

        # Розміри полотна
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        if cw < 50 or ch < 50:
            cw, ch = 640, 480

        h, w = self.processed_frame.shape[:2]
        self.lbl_resolution.config(text=f"{w} x {h}")

        # Масштабування із збереженням пропорцій (Fit Aspect Ratio)
        scale = min(cw / w, ch / h)
        nw, nh = int(w * scale), int(h * scale)
        if nw <= 0 or nh <= 0:
            return

        resized = cv2.resize(self.processed_frame, (nw, nh), interpolation=cv2.INTER_AREA)
        # OpenCV BGR -> PIL RGB
        rgb_frame = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_frame)
        self.photo_img = ImageTk.PhotoImage(image=pil_img)

        # Центрування на Canvas
        self.canvas.delete("all")
        self.canvas.create_image(cw // 2, ch // 2, anchor=tk.CENTER, image=self.photo_img)

    # -------------------------------------------------------------------------
    # Події елементів управління
    # -------------------------------------------------------------------------
    def on_mode_change(self):
        mode = self.selected_mode.get()
        if mode == "face":
            self.lbl_slider1.config(text="Масштаб каскаду (scaleFactor):")
            self.lbl_slider2.config(text="Чутливість (minNeighbors):")
            self.slider1.config(from_=1.05, to=1.4, resolution=0.05)
            self.slider2.config(from_=300, to=2400, resolution=100)
            self.slider1.set(1.15)
            self.slider2.set(1200)

        elif mode == "shapes":
            self.lbl_slider1.config(text="Чутливість розмиття:")
            self.lbl_slider2.config(text="Мінімальна площа фігури (px):")
            self.slider1.config(from_=1.0, to=3.0, resolution=0.5)
            self.slider2.config(from_=300, to=3000, resolution=50)
            self.slider1.set(1.0)
            self.slider2.set(1200)

        elif mode == "canny":
            self.lbl_slider1.config(text="Поріг Canny нижній:")
            self.lbl_slider2.config(text="Поріг Canny верхній:")
            self.slider1.config(from_=1.0, to=4.0, resolution=0.5)
            self.slider2.config(from_=500, to=2500, resolution=100)
            self.slider1.set(1.2)
            self.slider2.set(1500)

        elif mode == "motion":
            self.lbl_slider1.config(text="Поріг чутливості руху:")
            self.lbl_slider2.config(text="Мінімальна зона руху (px):")
            self.slider1.config(from_=0.8, to=3.0, resolution=0.2)
            self.slider2.config(from_=200, to=2000, resolution=100)
            self.slider1.set(1.2)
            self.slider2.set(1000)
            self.engine.reset_motion()

        self.process_and_render_frame()

    def on_slider1_change(self, val):
        if not self.is_video_running:
            self.process_and_render_frame()

    def on_slider2_change(self, val):
        if not self.is_video_running:
            self.process_and_render_frame()

    # -------------------------------------------------------------------------
    # Збереження знімка
    # -------------------------------------------------------------------------
    def save_snapshot(self):
        if self.processed_frame is None:
            messagebox.showwarning("Попередження", "Немає кадру для збереження.")
            return

        out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
        os.makedirs(out_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        mode = self.selected_mode.get()
        filename = f"snapshot_{mode}_{timestamp}.png"
        filepath = os.path.join(out_dir, filename)

        if VisionEngine.save_image_file(filepath, self.processed_frame, ".png"):
            self.status_bar.config(text=f"Знімок збережено: output/{filename}")
            messagebox.showinfo("Збережено", f"Знімок успішно збережено у:\noutput/{filename}")
        else:
            messagebox.showerror("Помилка", "Не вдалося зберегти знімок.")

    # -------------------------------------------------------------------------
    # Генерація звіту docx
    # -------------------------------------------------------------------------
    def open_generate_report(self):
        try:
            from generate_docx_report import create_lab_report
            report_path = create_lab_report()
            messagebox.showinfo("Звіт готовий", f"Офіційний звіт лабораторної роботи згенеровано:\n{report_path}")
        except Exception as e:
            messagebox.showerror("Помилка генерації звіту", f"Виникла помилка: {e}")

    def on_close(self):
        self.stop_video_stream()
        self.root.destroy()


def run_gui():
    root = tk.Tk()
    app = VisionApp(root)
    root.mainloop()


if __name__ == "__main__":
    run_gui()
