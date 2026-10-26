import tkinter as tk
from tkinter import messagebox

class ExpertSystemApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Експертна система: IT-Профорієнтація")
        self.root.geometry("650x580")
        self.root.config(bg="#f4f6f9")
        self.root.minsize(600, 520)

        # База знань (бали для професійних напрямків)
        self.scores = {
            "Data Science & AI": 0,
            "Backend-розробка": 0,
            "UI/UX Дизайн": 0,
            "QA (Тестування та якість)": 0,
            "Frontend-розробка": 0,
            "DevOps / Хмарні технології": 0
        }

        self.current_question = 0

        # Розширена база знань з 5 питань
        self.questions = [
            {
                "question": "1. Який склад мислення та підхід до задач вам найближчий?",
                "options": [
                    ("Аналітичний (робота з великими масивами даних, логіка, статистика)", {"Data Science & AI": 40, "Backend-розробка": 20}),
                    ("Творчий / Візуальний (естетика, дизайн інтерфейсів, робота з кольором)", {"UI/UX Дизайн": 45, "Frontend-розробка": 30}),
                    ("Уважний до деталей (пошук слабких місць, аналіз структури та помилок)", {"QA (Тестування та якість)": 40, "Backend-розробка": 10})
                ]
            },
            {
                "question": "2. Що вам більше подобається створювати чи налаштовувати?",
                "options": [
                    ("Складну архітектуру та бізнес-логіку програми, яка працює «під капотом»", {"Backend-розробка": 40, "Data Science & AI": 20}),
                    ("Інтерфейси та додатки, якими щодня користуються люди на екранах", {"Frontend-розробка": 40, "UI/UX Дизайн": 35}),
                    ("Налаштування серверів, автоматизацію процесів доставки коду та хмари", {"DevOps / Хмарні технології": 45, "Backend-розробка": 15})
                ]
            },
            {
                "question": "3. Як ви ставитеся до рутинних та чітко структурованих завдань?",
                "options": [
                    ("Спокійно, ціную чіткі інструкції, протоколи та стандарти безпеки", {"QA (Тестування та якість)": 30, "DevOps / Хмарні технології": 30}),
                    ("Мені більше до душі постійний пошук нових експериментальних рішень", {"UI/UX Дизайн": 30, "Data Science & AI": 30}),
                    ("Бажаю балансу між творчістю при написанні коду та стабільністю структури", {"Frontend-розробка": 25, "Backend-розробка": 25})
                ]
            },
            {
                "question": "4. Яка кінцева мета в проєкті приносить вам найбільше задоволення?",
                "options": [
                    ("Бачити, як моя модель робить точні прогнози чи знаходить закономірності", {"Data Science & AI": 35, "Backend-розробка": 15}),
                    ("Бачити бездоганно працюючий та швидкий інтерфейс без жодних багів", {"Frontend-розробка": 35, "QA (Тестування та якість)": 25}),
                    ("Розуміти, що створена мною система витримує мільйонні навантаження 24/7", {"DevOps / Хмарні технології": 40, "Backend-розробка": 30})
                ]
            },
            {
                "question": "5. У якому середовищі вам комфортніше працювати комунікаційно?",
                "options": [
                    ("Здебільшого самостійно за глибоким зануренням в алгоритми та код", {"Backend-розробка": 25, "Data Science & AI": 25}),
                    ("У постійній взаємодії з дизайнерами, бізнесом та користувачами продукту", {"UI/UX Дизайн": 30, "Frontend-розробка": 30}),
                    ("На стику команд розробки та системного адміністрування (допомога всім)", {"DevOps / Хмарні технології": 35, "QA (Тестування та якість)": 25})
                ]
            }
        ]

        # Змінні для збереження посилань на віджети результату
        self.result_label = None
        self.restart_btn = None

        # Верхній заголовок
        self.title_label = tk.Label(root, text="Віртуальний кар'єрний консультант", font=("Segoe UI", 16, "bold"), bg="#f4f6f9", fg="#2c3e50")
        self.title_label.pack(pady=15)

        # Блок запитання
        self.question_label = tk.Label(root, text="", font=("Segoe UI", 13, "bold"), bg="#f4f6f9", fg="#34495e", wraplength=580, justify="left")
        self.question_label.pack(pady=10, padx=25)

        # Фрейм для кнопок варіантів
        self.btn_frame = tk.Frame(root, bg="#f4f6f9")
        self.btn_frame.pack(pady=10, fill="x", padx=30)

        self.option_buttons = []
        for i in range(3):
            btn = tk.Button(self.btn_frame, text="", font=("Segoe UI", 11),
                            bg="#ffffff", fg="#2c3e50", relief="groove", 
                            wraplength=520, justify="left", anchor="w",
                            padx=12, pady=10,
                            command=lambda idx=i: self.select_option(idx))
            btn.pack(pady=6, fill="x")
            self.option_buttons.append(btn)

        self.next_question()

    def next_question(self):
        if self.current_question < len(self.questions):
            q_data = self.questions[self.current_question]
            self.question_label.config(text=f"Запитання {self.current_question + 1} із {len(self.questions)}:\n{q_data['question']}")
            
            for i, (text, _) in enumerate(q_data["options"]):
                self.option_buttons[i].config(text=text, state="normal")
                self.option_buttons[i].pack(pady=6, fill="x")
            
            for i in range(len(q_data["options"]), 3):
                self.option_buttons[i].config(text="")
                self.option_buttons[i].pack_forget()
        else:
            self.show_result()

    def select_option(self, option_index):
        q_data = self.questions[self.current_question]
        _, weights = q_data["options"][option_index]
        
        for category, points in weights.items():
            if category in self.scores:
                self.scores[category] += points

        self.current_question += 1
        self.next_question()

    def show_result(self):
        self.question_label.pack_forget()
        self.btn_frame.pack_forget()
        self.title_label.config(text="Результат експертного аналізу")

        best_match = max(self.scores, key=self.scores.get)
        max_score = self.scores[best_match]

        result_text = f"💡 На основі ваших відповідей експерт рекомендує звернути увагу на:\n\n📌 {best_match}\n\n"
        
        if max_score > 90:
            result_text += "Аналіз експерта: Ваші відповіді демонструють дуже чіткий вектор і високу схильність саме до цього профілю. Рекомендуємо почати профільне навчання в цій сфері."
        else:
            result_text += "Аналіз експерта: Ваші інтереси збалансовані, тому ви можете успішно реалізувати себе як у цьому напрямку, так і в суміжних галузях IT."

        self.result_label = tk.Label(self.root, text=result_text, font=("Segoe UI", 12), bg="#f4f6f9", fg="#2c3e50", wraplength=580, justify="left")
        self.result_label.pack(pady=30, padx=25)

        self.restart_btn = tk.Button(self.root, text="Пройти консультацію знову", font=("Segoe UI", 11, "bold"), bg="#3498db", fg="white",
                                    command=self.restart_app, relief="flat", padx=20, pady=10)
        self.restart_btn.pack(pady=10)

    def restart_app(self):
        # Видаляємо попередні віджети результату, якщо вони існують
        if self.result_label:
            self.result_label.destroy()
        if self.restart_btn:
            self.restart_btn.destroy()

        self.scores = {k: 0 for k in self.scores}
        self.current_question = 0
        
        self.btn_frame.pack(pady=10, fill="x", padx=30)
        self.question_label.pack(pady=10, padx=25)
        self.title_label.config(text="Віртуальний кар'єрний консультант")
        self.next_question()

if __name__ == "__main__":
    root = tk.Tk()
    app = ExpertSystemApp(root)
    root.mainloop()