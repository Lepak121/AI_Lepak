import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from flask import Flask, render_template, request, jsonify
from src.bot_core import TravelBot

app = Flask(__name__)
bot = TravelBot()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json() or {}
    user_message = data.get("message", "")
    response = bot.process_message(user_message)
    return jsonify(response)


@app.route("/api/reset", methods=["POST"])
def reset():
    bot.reset()
    return jsonify({"status": "ok", "message": "Context reset successfully"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"🚀 Веб-додаток Wanderlust AI запущено на http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
