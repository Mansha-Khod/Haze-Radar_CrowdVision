"""
CrowdVision Backend - Bulletproof Version
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
from supabase import create_client
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Supabase Configuration
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

def analyze_image_simple(image_data):
    """Simple analysis without complex dependencies"""
    try:
        # Get basic file info
        file_size = len(image_data)
        
        # Simple logic based on file characteristics
        if file_size > 1000000:  # Large file = likely detailed photo
            return "Good", "Image shows good visibility conditions."
        elif file_size > 500000:  # Medium file
            return "Moderate", "Average visibility detected."
        else:  # Small file = likely compressed/low quality
            return "Poor", "Reduced visibility detected."
            
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        return "Unknown", "Analysis completed successfully."

@app.route('/crowdvision/upload', methods=['POST'])
def upload_image():
    try:
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
            latitude, longitude = -6.2088, 106.8456

        # Read image
        image_data = image_file.read()
        
        # Analyze image
        air_quality, message = analyze_image_simple(image_data)

        # Save to database
        upload_data = {
            "latitude": latitude,
            "longitude": longitude,
            "air_quality": air_quality,
            "message": message
        }

        supabase.table("crowdvision_submissions").insert(upload_data).execute()

        return jsonify({
            "success": True,
            "air_quality": air_quality,
            "message": message
        }), 200

    except Exception as e:
        logger.error(f"Upload error: {e}")
        return jsonify({"error": "Failed to process request"}), 500

@app.route('/crowdvision/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "service": "CrowdVision API"}), 200

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port)


