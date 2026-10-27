#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Графічний інтерфейс користувача (GUI на Tkinter)
для системи розпізнавання образів смартфонів за кількісними та якісними характеристиками.
Підтримує як Евклідову (L2), так і Мангеттенську (L1) метрики!
"""

import tkinter as tk
from tkinter import ttk
from dataset import get_smartphones_model, get_test_samples


class PatternRecognitionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Розпізнавання образів смартфонів (Кількісні та Якісні ознаки)")
        self.root.geometry("1060x780")
        self.root.minsize(940, 680)

        self.model = get_smartphones_model()
        self.test_samples = get_test_samples()

        self.vars = {}
        self.metric_var = tk.StringVar(value="euclidean")

        self.setup_ui()
        self.load_preset(0)
        self.classify_current()

    def setup_ui(self):
        style = ttk.Style()
        style.theme_use('clam')

        header_frame = tk.Frame(self.root, bg="#1E293B", pady=12, padx=16)
        header_frame.pack(fill=tk.X)

        tk.Label(
            header_frame, 
            text="СИСТЕМА РОЗПІЗНАВАННЯ ОБРАЗІВ СМАРТФОНІВ",
            font=("Segoe UI", 16, "bold"), 
            fg="#F8FAFC", 
            bg="#1E293B"
        ).pack(anchor="w")

        tk.Label(
            header_frame, 
            text="Класи: Бюджетний | Середній клас | Флагман | Геймерський | Алгоритми: Min-Max, Ранжування, Евклід (L2), Мангеттен (L1)",
            font=("Segoe UI", 9), 
            fg="#94A3B8", 
            bg="#1E293B"
        ).pack(anchor="w")

        main_pane = tk.PanedWindow(self.root, orient=tk.HORIZONTAL, bg="#CBD5E1", sashwidth=4)
        main_pane.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        left_frame = ttk.LabelFrame(main_pane, text=" Характеристики смартфона для розпізнавання ", padding=12)
        main_pane.add(left_frame, minsize=480)

        right_frame = ttk.LabelFrame(main_pane, text=" Результати розпізнавання та метричний аналіз ", padding=12)
        main_pane.add(right_frame, minsize=480)

        # Пресети
        preset_frame = ttk.Frame(left_frame)
        preset_frame.pack(fill=tk.X, pady=(0, 6))

        ttk.Label(preset_frame, text="Швидкі приклади:", font=("Segoe UI", 9, "bold")).pack(side=tk.LEFT, padx=(0, 4))
        for i, test in enumerate(self.test_samples):
            btn = ttk.Button(
                preset_frame, 
                text=test['expected'], 
                command=lambda idx=i: self.load_preset(idx),
                width=11
            )
            btn.pack(side=tk.LEFT, padx=2)

        # Вибір метрики
        metric_frame = ttk.LabelFrame(left_frame, text=" Вибір метричного критерію відстані ", padding=6)
        metric_frame.pack(fill=tk.X, pady=(0, 8))

        rb1 = ttk.Radiobutton(metric_frame, text="Евклідова відстань (L2-норма)", variable=self.metric_var, 
                                value="euclidean", command=self.classify_current)
        rb1.pack(side=tk.LEFT, padx=10)

        rb2 = ttk.Radiobutton(metric_frame, text="Мангеттенська відстань (L1-норма)", variable=self.metric_var, 
                                value="manhattan", command=self.classify_current)
        rb2.pack(side=tk.LEFT, padx=10)

        # Скрол для параметрів
        canvas = tk.Canvas(left_frame, highlightthickness=0)
        scrollbar = ttk.Scrollbar(left_frame, orient="vertical", command=canvas.yview)
        scroll_content = ttk.Frame(canvas)

        scroll_content.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scroll_content, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        ttk.Label(scroll_content, text="-- Кількісні характеристики (Лінійна Min-Max нормалізація) --", 
                  font=("Segoe UI", 9, "bold", "italic")).pack(anchor="w", pady=(0, 4))

        quant_features = [
            ("price_usd", "Ціна смартфона ($):", 120, 1400, 10),
            ("ram_gb", "Обсяг ОЗП (RAM, ГБ):", 4, 24, 2),
            ("battery_mah", "Ємність батареї (мА·год):", 3000, 7000, 100),
            ("camera_mp", "Камера основна (Мп):", 12, 200, 2),
            ("screen_hz", "Частота оновлення (Гц):", 60, 165, 15),
            ("weight_g", "Вага смартфона (грами):", 150, 260, 5)
        ]

        for fname, flabel, min_v, max_v, step in quant_features:
            row = ttk.Frame(scroll_content)
            row.pack(fill=tk.X, pady=2)
            ttk.Label(row, text=flabel, width=26, anchor="w").pack(side=tk.LEFT)
            var = tk.DoubleVar()
            self.vars[fname] = var

            val_lbl = ttk.Label(row, text=str(min_v), width=6, anchor="e")
            scale = ttk.Scale(
                row, from_=min_v, to=max_v, variable=var,
                command=lambda val, l=val_lbl: l.config(text=f"{float(val):.0f}")
            )
            scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)
            val_lbl.pack(side=tk.RIGHT)

        ttk.Label(scroll_content, text="-- Якісні характеристики (Шкалювання та кодування) --", 
                  font=("Segoe UI", 9, "bold", "italic")).pack(anchor="w", pady=(10, 4))

        self.create_combo_row(scroll_content, "water_resistance", "Вологозахист (порядкова):", 
                              ["Немає", "IP54 (бризки)", "IP68 (занурення)"])
        self.create_combo_row(scroll_content, "display_matrix", "Матриця екрана (порядкова):", 
                              ["TFT/IPS", "AMOLED", "LTPO OLED"])
        self.create_combo_row(scroll_content, "wireless_charging", "Бездротова зарядка (бінарна):", 
                              ["Ні", "Так"])
        self.create_combo_row(scroll_content, "active_cooling", "Активний кулер (бінарна):", 
                              ["Ні", "Так"])
        self.create_combo_row(scroll_content, "body_material", "Матеріал корпусу (номінальна):", 
                              ["Пластик", "Алюміній", "Скло", "Титан/Скло", "Екошкіра/Титан"])

        btn_run = tk.Button(
            left_frame, text="⚡ РОЗПІЗНАТИ ОБРАЗ СМАРТФОНА",
            font=("Segoe UI", 11, "bold"), bg="#2563EB", fg="white",
            relief=tk.FLAT, pady=7, cursor="hand2", command=self.classify_current
        )
        btn_run.pack(fill=tk.X, pady=(8, 0))

        # Банер
        self.banner_frame = tk.Frame(right_frame, bg="#10B981", pady=10, padx=12)
        self.banner_frame.pack(fill=tk.X, pady=(0, 8))

        self.lbl_winner_title = tk.Label(
            self.banner_frame, text="ВИЗНАЧЕНИЙ КЛАС ОБРАЗУ:", 
            font=("Segoe UI", 9, "bold"), fg="#ECFDF5", bg="#10B981"
        )
        self.lbl_winner_title.pack()

        self.lbl_winner_class = tk.Label(
            self.banner_frame, text="Флагман", 
            font=("Segoe UI", 18, "bold"), fg="white", bg="#10B981"
        )
        self.lbl_winner_class.pack(pady=(2, 0))

        ttk.Label(right_frame, text="Відстані до класів за обраною метрикою:", font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(4, 2))

        self.progress_container = ttk.Frame(right_frame)
        self.progress_container.pack(fill=tk.X, pady=(0, 8))
        self.class_widgets = {}

        for c in self.model.classes:
            row = ttk.Frame(self.progress_container)
            row.pack(fill=tk.X, pady=2)
            ttk.Label(row, text=f"{c}:", width=16, anchor="w", font=("Segoe UI", 9)).pack(side=tk.LEFT)
            pb = ttk.Progressbar(row, orient="horizontal", mode="determinate", length=140)
            pb.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)
            info_lbl = ttk.Label(row, text="d=0.0000 | 0.0%", width=18, anchor="e", font=("Consolas", 9))
            info_lbl.pack(side=tk.RIGHT)
            self.class_widgets[c] = (pb, info_lbl)

        ttk.Label(right_frame, text="k-Найближчі сусіди з навчальної вибірки (k=3):", font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(6, 2))
        self.knn_text = tk.Text(right_frame, height=4, font=("Consolas", 9), bg="#F8FAFC", relief=tk.SOLID, bd=1)
        self.knn_text.pack(fill=tk.X, pady=(0, 6))

        ttk.Label(right_frame, text="Покроковий внесок ознак у відстань до переможця:", font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(4, 2))
        columns = ("feature", "type", "sample_val", "dist_i")
        self.tree = ttk.Treeview(right_frame, columns=columns, show="headings", height=7)
        self.tree.heading("feature", text="Ознака")
        self.tree.heading("type", text="Тип")
        self.tree.heading("sample_val", text="Значення зразка")
        self.tree.heading("dist_i", text="Різниця d_i [0..1]")

        self.tree.column("feature", width=120)
        self.tree.column("type", width=90)
        self.tree.column("sample_val", width=120)
        self.tree.column("dist_i", width=90, anchor="e")

        tree_scroll = ttk.Scrollbar(right_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=tree_scroll.set)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tree_scroll.pack(side=tk.RIGHT, fill=tk.Y)

    def create_combo_row(self, parent, fname, label_text, values):
        row = ttk.Frame(parent)
        row.pack(fill=tk.X, pady=2)
        ttk.Label(row, text=label_text, width=26, anchor="w").pack(side=tk.LEFT)
        var = tk.StringVar(value=values[0])
        self.vars[fname] = var
        cb = ttk.Combobox(row, textvariable=var, values=values, state="readonly", width=18)
        cb.pack(side=tk.RIGHT, fill=tk.X, expand=True)

    def load_preset(self, idx: int):
        test = self.test_samples[idx]
        sample = test['sample']
        for k, v in sample.items():
            if k in self.vars:
                self.vars[k].set(v)
        self.classify_current()

    def get_current_sample(self):
        return {fname: var.get() for fname, var in self.vars.items()}

    def classify_current(self):
        sample = self.get_current_sample()
        metric = self.metric_var.get()

        res_cent = self.model.classify_by_centroid(sample, metric=metric)
        res_knn = self.model.classify_knn(sample, k=3, metric=metric)

        best_class = res_cent['predicted_class']
        knn_class = res_knn['predicted_class']

        colors = {
            "Бюджетний": "#0284C7",
            "Середній клас": "#10B981",
            "Флагман": "#8B5CF6",
            "Геймерський": "#F43F5E"
        }
        accent = colors.get(best_class, "#2563EB")
        self.banner_frame.config(bg=accent)
        m_label = "Евклід (L2)" if metric == "euclidean" else "Мангеттен (L1)"
        self.lbl_winner_title.config(bg=accent, text=f"МЕТРИКА: {m_label} | k-NN: {knn_class}")
        self.lbl_winner_class.config(bg=accent, text=f"{best_class} ({res_cent['confidence'][best_class]:.1f}%)")

        for c, (pb, info_lbl) in self.class_widgets.items():
            dist = res_cent['distances'][c]
            conf = res_cent['confidence'][c]
            pb['value'] = conf
            info_lbl.config(text=f"d={dist:.4f} | {conf:4.1f}%")

        self.knn_text.delete("1.0", tk.END)
        for i, neighbor in enumerate(res_knn['k_nearest'], start=1):
            line = f"{i}. {neighbor['name']:<22} | Клас: {neighbor['class']:<14} | d = {neighbor['distance']:.4f}\n"
            self.knn_text.insert(tk.END, line)

        for item in self.tree.get_children():
            self.tree.delete(item)

        best_details = res_cent['details'][best_class]
        for fname, feat in self.model.features.items():
            diff = best_details.get(fname, 0.0)
            val = sample.get(fname, "")
            unit = feat.unit if feat.ftype == 'quantitative' else ""
            display_val = f"{val} {unit}".strip()
            self.tree.insert("", tk.END, values=(fname, feat.ftype, display_val, f"{diff:.4f}"))


def launch_gui():
    root = tk.Tk()
    app = PatternRecognitionGUI(root)
    root.mainloop()


if __name__ == "__main__":
    launch_gui()
