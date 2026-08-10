"""
PosterIQ — Flask Backend
Endpoints:
  POST /api/score        → score + issues + features from uploaded image
  POST /api/enhance      → placeholder for your enhancement model
  GET  /api/health       → health check
"""

import os
import uuid
import json
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename

from scorer import evaluate_new_poster, is_model_loaded, load_model

# ── Config ──────────────────────────────────────────────────────────────────
UPLOAD_FOLDER   = "uploads"
ALLOWED_EXT     = {"png", "jpg", "jpeg", "webp", "bmp"}
STATE_PATH      = os.environ.get("STATE_PATH", "poster_scorer_state.pkl")
MAX_CONTENT_MB  = 20

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"]    = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_MB * 1024 * 1024

CORS(app, resources={r"/api/*": {"origins": "*"}})   # tighten in production
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("posteriq")

# Pre-load model at startup so first request isn't slow
load_model(STATE_PATH)


# ── Helpers ──────────────────────────────────────────────────────────────────
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT


def build_recommendations(issues: list[str], features: dict) -> list[dict]:
    """
    Turn raw issue strings + feature values into structured recommendation
    objects that match what the frontend expects.
    Each rec: { id, priority, category, text }
    """
    priority_map = {
        "Low contrast":     ("critical", "Contrast"),
        "Overly harsh":     ("critical", "Contrast"),
        "Too cluttered":    ("critical", "Layout"),
        "Too many edges":   ("critical", "Layout"),
        "Not enough white": ("improve",  "Whitespace"),
        "Too simple":       ("improve",  "Visual"),
        "Dull colors":      ("improve",  "Color"),
        "Too many colors":  ("improve",  "Color"),
        "Too much text":    ("improve",  "Typography"),
        "Too little text":  ("improve",  "Typography"),
        "Layout imbalance": ("improve",  "Layout"),
    }

    recs = []
    for i, issue in enumerate(issues, 1):
        priority, category = "improve", "Design"
        for key, (pri, cat) in priority_map.items():
            if key.lower() in issue.lower():
                priority, category = pri, cat
                break
        recs.append({"id": i, "priority": priority, "category": category, "text": issue})

    # If no issues found, add a positive note
    if not recs:
        recs.append({
            "id": 1, "priority": "good", "category": "Overall",
            "text": "No major design issues detected. Your poster scores well across all dimensions."
        })

    # Append feature-level insights (always shown)
    br = features.get("brightness", 128)
    if br < 80:
        recs.append({"id": len(recs)+1, "priority": "improve", "category": "Brightness",
                     "text": f"Image is quite dark (brightness={br:.0f}/255). Consider increasing overall luminosity."})
    elif br > 200:
        recs.append({"id": len(recs)+1, "priority": "improve", "category": "Brightness",
                     "text": f"Image is very bright (brightness={br:.0f}/255). High brightness can wash out text."})

    ws = features.get("whitespace", 0)
    if ws < 0.1:
        recs.append({"id": len(recs)+1, "priority": "improve", "category": "Whitespace",
                     "text": f"Whitespace is very low ({ws*100:.1f}%). Add breathing room around key elements."})

    return recs


def score_to_breakdown(score: float, features: dict) -> dict:
    """
    Derive sub-scores for the 5 ring charts in the UI from raw features.
    All values are approximate — map feature values to 0-100 intuitively.
    """
    def clamp(v, lo=0, hi=100): return max(lo, min(hi, v))

    # Contrast: std of gray channel, ideal ≈ 60-80
    contrast_raw = features.get("contrast", 50)
    visual_score = clamp(int(contrast_raw / 1.3))

    # Entropy: higher = more complex; ideal 6-7 bits
    entropy = features.get("entropy", 5)
    entropy_score = clamp(int(100 - abs(entropy - 6.5) * 20))

    # Layout balance: lr + tb, lower deviation = better
    lr = features.get("lr_balance", 0.1)
    tb = features.get("tb_balance", 0.1)
    layout_score = clamp(int(100 - (lr + tb) * 200))

    # Colorfulness
    cf = features.get("colorfulness", 30)
    color_score = clamp(int(min(cf / 0.8, 100)))

    # Text density: ideal ≈ moderate
    td = features.get("text_density", 0.05)
    type_score = clamp(int(100 - abs(td - 0.05) * 500))

    return {
        "overall":    int(clamp(score)),
        "visual":     visual_score,
        "typography": type_score,
        "layout":     layout_score,
        "branding":   color_score,
        "contrast":   entropy_score,
    }


# ── Routes ───────────────────────────────────────────────────────────────────
@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model_loaded": is_model_loaded()})


@app.route("/api/score", methods=["POST"])
def score_poster():
    """
    Accepts: multipart/form-data with field 'poster' (image file)
    Returns: JSON  { score, scores{}, recommendations[], features{} }
    """
    if "poster" not in request.files:
        return jsonify({"error": "No file uploaded. Use field name 'poster'."}), 400

    file = request.files["poster"]
    if file.filename == "":
        return jsonify({"error": "Empty filename."}), 400
    if not allowed_file(file.filename):
        return jsonify({"error": f"File type not allowed. Use: {ALLOWED_EXT}"}), 400

    # Save to temp path
    ext      = file.filename.rsplit(".", 1)[1].lower()
    tmp_name = f"{uuid.uuid4().hex}.{ext}"
    tmp_path = os.path.join(app.config["UPLOAD_FOLDER"], tmp_name)
    file.save(tmp_path)
    log.info(f"Saved upload: {tmp_path}")

    try:
        result = evaluate_new_poster(tmp_path, STATE_PATH)
    except FileNotFoundError as e:
        os.remove(tmp_path)
        return jsonify({"error": f"Model state not found: {e}. Run training first."}), 500
    except Exception as e:
        log.exception("Scoring failed")
        os.remove(tmp_path)
        return jsonify({"error": f"Scoring failed: {str(e)}"}), 500
    finally:
        # Clean up upload (only if it still exists)
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    recommendations = build_recommendations(result["issues"], result["features"])
    scores          = score_to_breakdown(result["score"], result["features"])

    return jsonify({
        "score":           result["score"],
        "scores":          scores,
        "recommendations": recommendations,
        "features":        result["features"],
        "issues_raw":      result["issues"],
    })


@app.route("/api/enhance", methods=["POST"])
def enhance_poster():
    """
    Placeholder endpoint for your poster enhancement model.
    
    Expected input:  multipart/form-data
      - poster:           original image file
      - recommendations:  JSON string of recommendation list
    
    Returns: { enhanced_image_base64: str, changes_applied: list[str] }
    
    ─────────────────────────────────────────────────────────────────
    TO CONNECT YOUR ENHANCEMENT MODEL:
    Replace the section marked "# ← YOUR MODEL HERE" below with your
    actual model inference code. The image is saved at `tmp_path`.
    Return the enhanced image as base64.
    ─────────────────────────────────────────────────────────────────
    """
    if "poster" not in request.files:
        return jsonify({"error": "No file uploaded."}), 400

    file = request.files["poster"]
    recs_json = request.form.get("recommendations", "[]")
    recommendations = json.loads(recs_json)

    ext      = file.filename.rsplit(".", 1)[1].lower()
    tmp_name = f"{uuid.uuid4().hex}.{ext}"
    tmp_path = os.path.join(app.config["UPLOAD_FOLDER"], tmp_name)
    file.save(tmp_path)

    try:
        # ← YOUR ENHANCEMENT MODEL HERE ─────────────────────────────
        # Example interface:
        #
        #   from enhancer import enhance_poster_image
        #   enhanced_path = enhance_poster_image(tmp_path, recommendations)
        #   with open(enhanced_path, "rb") as f:
        #       import base64
        #       enhanced_b64 = base64.b64encode(f.read()).decode()
        #
        # For now, we echo back the original as a placeholder:
        import base64
        with open(tmp_path, "rb") as f:
            enhanced_b64 = base64.b64encode(f.read()).decode()

        changes = [rec["text"][:60] + "…" for rec in recommendations[:6]]
        # ─────────────────────────────────────────────────────────────

        return jsonify({
            "enhanced_image_base64": enhanced_b64,
            "image_format":         ext,
            "changes_applied":      changes,
        })

    except Exception as e:
        log.exception("Enhancement failed")
        return jsonify({"error": f"Enhancement failed: {str(e)}"}), 500
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
