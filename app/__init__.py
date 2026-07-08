"""Smart Farming AI Agent — Flask application factory."""
from flask import Flask
from flask_cors import CORS
from dotenv import load_dotenv
import os

load_dotenv()


def create_app() -> Flask:
    app = Flask(__name__, template_folder="../templates", static_folder="../static")
    app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev-secret-key-change-in-production")
    CORS(app)

    # Register blueprints
    from app.routes.chat import chat_bp
    from app.routes.weather import weather_bp
    from app.routes.crops import crops_bp
    from app.routes.market import market_bp
    from app.routes.advisory import advisory_bp
    from app.routes.main import main_bp

    app.register_blueprint(main_bp)
    app.register_blueprint(chat_bp, url_prefix="/api/chat")
    app.register_blueprint(weather_bp, url_prefix="/api/weather")
    app.register_blueprint(crops_bp, url_prefix="/api/crops")
    app.register_blueprint(market_bp, url_prefix="/api/market")
    app.register_blueprint(advisory_bp, url_prefix="/api/advisory")

    return app
