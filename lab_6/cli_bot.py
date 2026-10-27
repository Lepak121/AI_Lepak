import sys
from src.bot_core import TravelBot

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def main():
    print("=" * 65)
    print(" 🌍 ТУРИСТИЧНИЙ АСИСТЕНТ 'WANDERLUST AI' (CLI-РЕЖИМ)")
    print(" Введіть повідомлення для спілкування з ботом.")
    print(" Команди: 'reset' - очистити пам'ять діалогу, 'exit' - вийти.")
    print("=" * 65)

    bot = TravelBot()

    while True:
        try:
            user_input = input("\n👤 Ви: ").strip()
            if not user_input:
                continue

            if user_input.lower() in ["exit", "quit", "вихід"]:
                print("\n🤖 Бот: Дякуємо за звернення до Wanderlust Travel! До зустрічі!")
                break

            if user_input.lower() in ["reset", "скинути", "clear"]:
                bot.reset()
                print("\n🔄 [Пам'ять діалогу та слоти успішно очищено]")
                continue

            # Обробка репліки
            response = bot.process_message(user_input)

            # Виведення інформації про AI діагностику
            intent = response["intent"]
            conf = response["confidence"]
            entities = response["entities"]
            slots = response["slots"]

            print("-" * 65)
            print(f"📊 [AI Діагностика] Намір: {intent} ({conf:.1%}) | Сутності: {entities}")
            if slots:
                print(f"🧠 [Пам'ять слотів]: {slots}")
            print("-" * 65)

            # Відповідь бота
            print(f"🤖 Wanderlust Bot:\n{response['bot_text']}")

        except (KeyboardInterrupt, EOFError):
            print("\nСесію завершено.")
            break


if __name__ == "__main__":
    main()
