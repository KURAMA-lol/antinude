import importlib.util
import os
import secrets
import sys
from pathlib import Path

from flask import jsonify, request

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "safe-guard-moderator.py"

spec = importlib.util.spec_from_file_location("antinude_source", SOURCE)
if spec is None or spec.loader is None:
    raise RuntimeError("Could not load the moderation application")
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
app = module.app


@app.before_request
def require_api_key_for_image_checks():
    if request.path != "/check":
        return None
    expected = os.environ.get("API_KEY")
    if not expected:
        return jsonify({"status": "error", "message": "API authentication is not configured"}), 503
    supplied = request.headers.get("X-API-Key", "")
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        supplied = auth[7:].strip()
    if not secrets.compare_digest(supplied, expected):
        return jsonify({"status": "error", "message": "Unauthorized"}), 401
    return None
