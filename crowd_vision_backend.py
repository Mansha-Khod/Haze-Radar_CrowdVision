"""

"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from supabase import create_client
import os
import logging
import numpy as np
from PIL import Image
import io
import base64
import cv2

# =========================
# Logging Configuration
# =========================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("CrowdVision")

# =========================
# Flask App Setup
# =========================
app = Flask(__name__)
CORS(app)

# =========================
# Supabase Configuration
# =========================
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    logger.warning(" Supabase credentials not found in environment variables.")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# =========================
# Helper Functions
# =========================
def analyze_image_cv(image_data):
    """Enhanced haze analysis using OpenCV and brightness/contrast detection"""
    try:
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            raise ValueError("Invalid image format")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)
        contrast = img.std()

        # Simplified haze estimation formula
        haze_score = (255 - contrast) * 0.6 + (255 - brightness) * 0.4

        if haze_score < 100:
            return "Good", "Sky looks clear and visibility is excellent."
        elif haze_score < 160:
            return "Moderate", "Mild haze detected. Sensitive individuals should take precautions."
        else:
            return "Poor", "Heavy haze detected! Limit outdoor exposure."
    except Exception as e:
        logger.error(f"Image analysis failed: {e}")
        return "Unknown", "Unable to analyze image. Please try again."

# =========================
# Flask Route
# =========================
@app.route('/crowdvision/upload', methods=['POST'])
def upload_image():
    try:
        data = request.get_json()

        # --- Validation ---
        if not data:
            return jsonify({"error": "No data received"}), 400

        image_base64 = data.get("image")
        latitude = data.get("latitude")
        longitude = data.get("longitude")

        if not image_base64 or latitude is None or longitude is None:
            return jsonify({"error": "Missing required fields (image, latitude, longitude)"}), 400

        # --- Decode Base64 Image ---
        try:
            image_data = base64.b64decode(image_base64)
        except Exception:
            return jsonify({"error": "Invalid image format"}), 400

        # --- Analyze Image ---
        air_quality, message = analyze_image_cv(image_data)

        # --- Prepare Data ---
        upload_data = {
            "latitude": latitude,
            "longitude": longitude,
            "air_quality": air_quality,
            "message": message
        }

        # --- Store in Supabase ---
        try:
            response = supabase.table("crowdvision_submissions").insert(upload_data).execute()
            logger.info(f"Uploaded data: {upload_data}")
        except Exception as db_error:
            logger.error(f"Supabase insert failed: {db_error}")
            return jsonify({"error": "Database insert failed"}), 500

        # --- Success Response ---
        return jsonify({
            "success": True,
            "air_quality": air_quality,
            "message": message
        }), 200

    except Exception as e:
        logger.error(f" Unexpected error: {e}")
        return jsonify({"error": "Failed to process photo"}), 500


# =========================
# Main Entrypoint
# =========================
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)




