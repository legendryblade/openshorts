import importlib.util
import os
import sys

# Make the repo root importable so tests can import the app modules directly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Tests always run the app in self-host (BYOK) mode. app.py freezes
# BILLING_ENABLED at import time and load_dotenv never overrides an existing
# variable, so this must be set here — before any test module imports app — or
# the suite's behavior would depend on the developer's personal .env.
os.environ["BILLING_ENABLED"] = "0"

# Same principle as BILLING_ENABLED, for the pipeline knobs a developer's
# personal .env is most likely to set. With a real LLM_BASE_URL in the
# environment, main._run_gemini_stage takes the local-LLM branch instead of the
# Gemini branch the retry tests fake, and llm_backend signs requests with the
# developer's key (which then leaks into assertion diffs). Empty strings are
# used rather than deletes because load_dotenv() — run at import by main.py,
# app.py and gemini_worker.py — never overrides a variable that is already set.
# CAPTION_MARGIN_V gets the shipped default instead of "": int("") would crash
# subtitles at import.
os.environ["LLM_BASE_URL"] = ""
os.environ["LLM_API_KEY"] = ""
os.environ["CAPTION_MARGIN_V"] = "43"

# The billing/cloud tests import sqlalchemy and alembic, which only come from
# requirements-billing.txt (see its header). A self-host install never gets
# them, and an unimportable module at collection time interrupts the WHOLE
# suite — so skip those files when the optional deps are absent.
collect_ignore = []
if not all(importlib.util.find_spec(m) for m in ("sqlalchemy", "alembic")):
    collect_ignore += [
        "test_account_erasure.py",
        "test_api_keys.py",
        "test_billing_states.py",
        "test_mcp_oauth.py",
        "test_metering_free_plan.py",
        "test_proxy_ledger.py",
    ]
