"""
PROPER CrowdVision Backend - Working Version
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from supabase import create_client
import os
import logging
import numpy as np
from PIL import Image
import io

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
    logger.warning("Supabase credentials not found in environment variables.")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# =========================
# Helper Functions
# =========================
def analyze_image_simple(image_data):
    """Enhanced haze analysis using PIL and brightness/contrast detection"""
    try:
        img = Image.open(io.BytesIO(image_data))
        img_array = np.array(img)
        
        # Convert to grayscale if needed
        if len(img_array.shape) == 3:
            gray = np.mean(img_array, axis=2)
        else:
            gray = img_array
            
        brightness = np.mean(gray)
        contrast = np.std(gray)

        # Enhanced haze estimation
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
# Flask Routes
# =========================
@app.route('/crowdvision/upload', methods=['POST'])
def upload_image():
    try:
        # --- Validation ---
        if 'image' not in request.files:
            return jsonify({"error": "No image provided"}), 400
        
        image_file = request.files['image']
        if image_file.filename == '':
            return jsonify({"error": "No file selected"}), 400

        # Get coordinates
        try:
            latitude = float(request.form.get('latitude', -6.2088))
            longitude = float(request.form.get('longitude', 106.8456))
        except:
            return jsonify({"error": "Invalid coordinates"}), 400

        # --- Read Image ---
        image_data = image_file.read()
        
        # --- Analyze Image ---
        air_quality, message = analyze_image_simple(image_data)

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
        logger.error(f"Unexpected error: {e}")
        return jsonify({"error": "Failed to process photo"}), 500

@app.route('/crowdvision/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "service": "CrowdVision API"}), 200

# =========================
# Main Entrypoint
# =========================
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port)

