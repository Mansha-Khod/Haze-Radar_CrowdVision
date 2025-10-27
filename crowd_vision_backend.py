"""
CrowdVision Backend API for HazeRadar
Clean, simple, and working version
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from datetime import datetime
import cv2
import numpy as np
from PIL import Image
import io
import base64
from supabase import create_client, Client
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Supabase Configuration
SUPABASE_URL = os.getenv("SUPABASE_URL", "https://daxrnmvkpikjvvzgrhko.supabase.co")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImRheHJubXZrcGlranZ2emdyaGtvIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjA2OTkyNjEsImV4cCI6MjA3NjI3NTI2MX0.XWJ_aWUh5Eci5tQSRAATqDXmQ5nh2eHQGzYu6qMcsvQ")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Configuration
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

def get_health_advice(haze_level):
    advice_map = {
        "Good": "Great air quality! Perfect for outdoor activities.",
        "Moderate": "Air quality is acceptable. Sensitive people should reduce outdoor activity.",
        "Unhealthy for Sensitive Groups": "Sensitive groups may experience effects. Limit outdoor time.",
        "Unhealthy": "Everyone may feel effects. Avoid prolonged outdoor activity.",
        "Very Unhealthy": "Health alert! Avoid outdoor activity.",
        "Hazardous": "Emergency conditions! Stay indoors."
    }
    return advice_map.get(haze_level, "Check local air quality advisories.")

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def validate_coordinates(lat, lng):
    return -90 <= lat <= 90 and -180 <= lng <= 180

def analyze_haze_improved(image_data):
    """
    Simple but effective haze detection
    """
    try:
        img = Image.open(io.BytesIO(image_data))
        img_array = np.array(img)
        
        if len(img_array.shape) == 2:  # Grayscale
            img_rgb = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
        else:
            img_rgb = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        img_rgb = cv2.resize(img_rgb, (640, 480))
        hsv = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2HSV)
        
        # Look for blue sky regions
        lower_blue = np.array([100, 40, 40])
        upper_blue = np.array([140, 255, 255])
        sky_mask = cv2.inRange(hsv, lower_blue, upper_blue)
        
        # Calculate sky percentage
        sky_percentage = np.sum(sky_mask > 0) / sky_mask.size
        
        if sky_percentage > 0.3:  # Decent sky area found
            sky_pixels = cv2.bitwise_and(hsv, hsv, mask=sky_mask)
            non_zero_mask = sky_mask > 0
            avg_saturation = np.mean(sky_pixels[:, :, 1][non_zero_mask])
            avg_brightness = np.mean(sky_pixels[:, :, 2][non_zero_mask])
            
            # Determine haze level based on sky color
            if avg_brightness > 180 and avg_saturation > 80:
                return {"hazeLevel": "Good", "estimatedAQI": 25, "confidence": 85}
            elif avg_brightness > 150 and avg_saturation > 50:
                return {"hazeLevel": "Moderate", "estimatedAQI": 65, "confidence": 75}
            elif avg_brightness > 100 and avg_saturation > 20:
                return {"hazeLevel": "Unhealthy for Sensitive Groups", "estimatedAQI": 110, "confidence": 70}
            elif avg_brightness > 70:
                return {"hazeLevel": "Unhealthy", "estimatedAQI": 160, "confidence": 65}
            else:
                return {"hazeLevel": "Very Unhealthy", "estimatedAQI": 220, "confidence": 60}
        else:
            # No clear sky - estimate from overall brightness
            avg_brightness = np.mean(hsv[:, :, 2])
            if avg_brightness > 180:
                return {"hazeLevel": "Moderate", "estimatedAQI": 70, "confidence": 60}
            else:
                return {"hazeLevel": "Unhealthy", "estimatedAQI": 150, "confidence": 65}
                
    except Exception as e:
        logger.error(f"Error in haze analysis: {str(e)}")
        return {"hazeLevel": "Unknown", "estimatedAQI": 0, "confidence": 0}

@app.route('/crowdvision/upload', methods=['POST'])
def upload_image():
    """Handle image uploads - simple and clean"""
    try:
        # Basic validation
        if 'image' not in request.files:
            return jsonify({"error": "No image provided"}), 400
        
        image_file = request.files['image']
        if image_file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        if not allowed_file(image_file.filename):
            return jsonify({"error": "Please upload a PNG, JPG, or JPEG image"}), 400
        
        # Get coordinates
        try:
            latitude = float(request.form.get('latitude', -6.2088))
            longitude = float(request.form.get('longitude', 106.8456))
        except:
            latitude, longitude = -6.2088, 106.8456
        
        # Read image
        image_data = image_file.read()
        if len(image_data) > MAX_FILE_SIZE:
            return jsonify({"error": "Image too large. Maximum 10MB"}), 400
        
        # Analyze image
        logger.info(f"Analyzing image from {latitude}, {longitude}")
        analysis_result = analyze_haze_improved(image_data)
        
        # Save to database (simple version)
        upload_data = {
            "latitude": latitude,
            "longitude": longitude,
            "timestamp": datetime.now().isoformat(),
            "haze_level": analysis_result["hazeLevel"],
            "estimated_aqi": analysis_result["estimatedAQI"],
            "confidence": analysis_result["confidence"],
            "validated": False
        }
        
        result = supabase.table('crowdvision_submissions').insert(upload_data).execute()
        
        # Clean response - only what users need to see
        return jsonify({
            "success": True,
            "air_quality": analysis_result["hazeLevel"],
            "health_message": get_health_advice(analysis_result["hazeLevel"])
        }), 200
        
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        return jsonify({"error": "Failed to process image"}), 500

@app.route('/crowdvision/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy", 
        "service": "CrowdVision API"
    }), 200

@app.route('/crowdvision/recent', methods=['GET'])
def get_recent_submissions():
    """Get recent submissions for dashboard"""
    try:
        result = supabase.table('crowdvision_submissions')\
            .select('latitude, longitude, haze_level, timestamp')\
            .order('timestamp', desc=True)\
            .limit(50)\
            .execute()
        
        return jsonify({
            "success": True,
            "submissions": result.data
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port, debug=False)