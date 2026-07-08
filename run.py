"""Smart Farming AI Agent — WSGI entry point."""
from app import create_app

app = create_app()

if __name__ == "__main__":
    import os
    port = int(os.getenv("FLASK_PORT", 5000))
    debug = os.getenv("FLASK_DEBUG", "False").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug)
