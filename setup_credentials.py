"""
setup_credentials.py — Interactive credential setup for KisanMitra.

Run:  python setup_credentials.py
"""
import os
import sys
import re


ENV_FILE = os.path.join(os.path.dirname(__file__), ".env")

BANNER = """
╔══════════════════════════════════════════════════════╗
║   KisanMitra — IBM watsonx.ai Credential Setup      ║
╚══════════════════════════════════════════════════════╝
"""

HELP = """
Where to get your credentials
──────────────────────────────
IBM Cloud API Key:
  1. Go to https://cloud.ibm.com/iam/apikeys
  2. Click "Create an IBM Cloud API key"
  3. Give it a name (e.g. "KisanMitra"), click Create
  4. COPY the key NOW — it is only shown once

watsonx.ai Project ID:
  1. Go to https://dataplatform.cloud.ibm.com/wx/home
  2. Open (or create) your watsonx.ai project
  3. Click the "Manage" tab → copy the Project ID (UUID)

watsonx.ai URL (leave default unless your region differs):
  • Dallas (us-south): https://us-south.ml.cloud.ibm.com  ← default
  • Frankfurt (eu-de): https://eu-de.ml.cloud.ibm.com
  • Tokyo   (jp-tok):  https://jp-tok.ml.cloud.ibm.com
  • London  (eu-gb):   https://eu-gb.ml.cloud.ibm.com
"""


def prompt(label: str, current: str, secret: bool = False) -> str:
    display = "[hidden]" if (secret and current and "placeholder" not in current) else current
    val = input(f"  {label} [{display}]: ").strip()
    return val if val else current


def is_placeholder(val: str) -> bool:
    return not val or "your_" in val or "placeholder" in val or "change_in_production" in val


def read_env() -> dict:
    env: dict = {}
    if os.path.exists(ENV_FILE):
        with open(ENV_FILE, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, _, v = line.partition("=")
                    env[k.strip()] = v.strip()
    return env


def write_env(env: dict) -> None:
    lines = [
        "# IBM watsonx.ai Credentials",
        f"WATSONX_API_KEY={env.get('WATSONX_API_KEY', '')}",
        f"WATSONX_PROJECT_ID={env.get('WATSONX_PROJECT_ID', '')}",
        f"WATSONX_URL={env.get('WATSONX_URL', 'https://us-south.ml.cloud.ibm.com')}",
        "",
        "# Weather API (Open-Meteo is free, no key needed)",
        f"OPENWEATHER_API_KEY={env.get('OPENWEATHER_API_KEY', '')}",
        "",
        "# Flask Configuration",
        f"FLASK_SECRET_KEY={env.get('FLASK_SECRET_KEY', _random_secret())}",
        "FLASK_DEBUG=False",
        f"FLASK_PORT={env.get('FLASK_PORT', '5000')}",
    ]
    with open(ENV_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def _random_secret() -> str:
    import secrets
    return secrets.token_hex(32)


def validate_credentials(api_key: str, project_id: str, url: str) -> bool:
    """Try a lightweight watsonx.ai API call to verify credentials."""
    print("\n  🔄 Validating credentials against IBM watsonx.ai…", flush=True)
    try:
        from ibm_watsonx_ai import Credentials
        from ibm_watsonx_ai.foundation_models import ModelInference
        from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams

        creds = Credentials(url=url, api_key=api_key)
        model = ModelInference(
            model_id="ibm/granite-4-h-small",
            credentials=creds,
            project_id=project_id,
            params={GenParams.MAX_NEW_TOKENS: 5, GenParams.DECODING_METHOD: "greedy"},
        )
        # Minimal generate_text call (compatible with ibm-watsonx-ai 1.1.x)
        answer = model.generate_text(
            prompt="<|user|>\nReply with the single word: OK\n<|assistant|>"
        )
        print(f"  ✅ Connection successful! Model replied: '{answer.strip()}'")
        return True
    except ImportError:
        print("  ⚠️  ibm-watsonx-ai not installed. Run: pip install -r requirements.txt")
        return False
    except Exception as e:
        msg = str(e)
        print(f"\n  ❌ Validation failed: {msg}")
        if "api_key" in msg.lower() or "iam" in msg.lower() or "401" in msg:
            print("     → Check your WATSONX_API_KEY (copy it exactly; it's shown only once).")
        elif "project" in msg.lower() or "404" in msg:
            print("     → Check your WATSONX_PROJECT_ID (should be a UUID like xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx).")
        elif "url" in msg.lower() or "connection" in msg.lower():
            print(f"     → Check your WATSONX_URL. Current: {url}")
        return False


def main() -> None:
    print(BANNER)

    env = read_env()

    # Show help if any key is still a placeholder
    if is_placeholder(env.get("WATSONX_API_KEY", "")) or \
       is_placeholder(env.get("WATSONX_PROJECT_ID", "")):
        print(HELP)

    print("Enter your credentials (press Enter to keep the existing value):\n")

    api_key    = prompt("IBM Cloud API Key   ", env.get("WATSONX_API_KEY", ""), secret=True)
    project_id = prompt("watsonx Project ID  ", env.get("WATSONX_PROJECT_ID", ""))
    url        = prompt("watsonx URL         ", env.get("WATSONX_URL", "https://us-south.ml.cloud.ibm.com"))

    # Basic format checks
    if is_placeholder(api_key):
        print("\n  ❌ API key looks like a placeholder. Please enter the real key.")
        sys.exit(1)

    uuid_re = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.I)
    if not uuid_re.match(project_id):
        print(f"\n  ⚠️  Project ID '{project_id}' doesn't look like a UUID.")
        print("     Expected format: xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx")
        cont = input("     Continue anyway? [y/N]: ").strip().lower()
        if cont != "y":
            sys.exit(1)

    # Update env dict and write
    env["WATSONX_API_KEY"]    = api_key
    env["WATSONX_PROJECT_ID"] = project_id
    env["WATSONX_URL"]        = url
    if is_placeholder(env.get("FLASK_SECRET_KEY", "")):
        env["FLASK_SECRET_KEY"] = _random_secret()

    write_env(env)
    print(f"\n  💾 Saved credentials to {ENV_FILE}")

    # Validate
    ok = validate_credentials(api_key, project_id, url)

    print()
    if ok:
        print("  🚀 Ready! Start the server with:  python run.py")
    else:
        print("  📝 Credentials saved but validation failed.")
        print("     Fix the values above and re-run:  python setup_credentials.py")
        sys.exit(1)


if __name__ == "__main__":
    main()
