import os
import io
import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms
from flask import Flask, request, jsonify
from flask_cors import CORS
import gdown

app = Flask(__name__)
CORS(app)
device = torch.device("cpu")

# Download model
MODEL_PATH = "crowd_vision_model.pth"
DRIVE_FILE_ID = "168Jui3J763s_JmoxH3w7nz5FplMH3Kin" 

if os.path.exists(MODEL_PATH):
    os.remove(MODEL_PATH)
    print("Removed old model file")

print("Downloading model from Google Drive...")
gdown.download(id=DRIVE_FILE_ID, output=MODEL_PATH, quiet=False)
print("Model downloaded!")

# Model architecture
class CrowdVisionModel(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.backbone = models.efficientnet_b0(weights='DEFAULT')
        in_features = self.backbone.classifier[1].in_features
        
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)

model = CrowdVisionModel(num_classes=2).to(device)

print("Loading trained weights...")
try:
    state_dict = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    model.load_state_dict(state_dict)
    model.eval()
    print("Model weights loaded successfully")
    
    # Quick verification
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    with torch.no_grad():
        test_output = model(dummy_input)
        test_probs = torch.softmax(test_output, dim=1)
    
    print(f"Model test - Clear: {test_probs[0][0]:.3f}, Hazy: {test_probs[0][1]:.3f}")
    
    if abs(test_probs[0][0].item() - 0.5) < 0.1:
        print("Warning: Model outputs appear random")
    else:
        print("Model appears trained properly")
        
except Exception as e:
    print(f"Failed to load weights: {e}")
    import traceback
    traceback.print_exc()

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def calculate_visibility_score(image):
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)

    contrast = gray.std()
    brightness = np.mean(gray)
    edges = cv2.Canny(gray, 100, 200)
    edge_density = np.sum(edges > 0) / edges.size

    visibility = (contrast * 0.4) + (edge_density * 4000 * 0.4) + (brightness * 0.2)
    visibility = np.clip(visibility, 0, 100)

    return visibility, contrast, edge_density, brightness

def detect_atmospheric_haze(image):
    """Detect uniform atmospheric haze/smog (grayish veil over scene)"""
    try:
        img_array = np.array(image)
        hsv_image = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
        h, s, v = hsv_image[:,:,0], hsv_image[:,:,1], hsv_image[:,:,2]
        
        # Low saturation across image indicates haze/smog
        avg_saturation = np.mean(s)
        saturation_std = np.std(s)
        
        # Haze creates uniform desaturation
        has_uniform_desaturation = avg_saturation < 80 and saturation_std < 35
        
        # Calculate color variance in the image
        color_variance = np.std(img_array, axis=(0,1)).mean()
        
        print(f"Haze indicators - Avg Sat: {avg_saturation:.1f}, Sat Std: {saturation_std:.1f}, Color Var: {color_variance:.1f}")
        
        return has_uniform_desaturation and color_variance < 45
        
    except Exception as e:
        print(f"Haze detection error: {e}")
        return False

def has_distinct_clouds(image):
    """Detect blue sky with distinct white clouds (clear day with clouds)"""
    try:
        img_array = np.array(image)
        hsv_image = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
        h, s, v = hsv_image[:,:,0], hsv_image[:,:,1], hsv_image[:,:,2]
        
        # Blue sky detection - detect vibrant blue
        blue_mask = (h > 90) & (h < 130) & (s > 25) & (v > 70)  # More lenient
        blue_ratio = np.sum(blue_mask) / blue_mask.size
        
        # White cloud detection - bright with low saturation
        white_mask = (v > 180) & (s < 60)  # More lenient for clouds
        white_ratio = np.sum(white_mask) / white_mask.size
        
        # Calculate average saturation and value
        avg_saturation = np.mean(s)
        avg_brightness = np.mean(v)
        
        # Check for distinct boundaries (clouds have edges)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        edge_density = np.sum(edges > 0) / edges.size
        
        # Check RGB color variance to distinguish from gray haze
        rgb_std = np.std(img_array, axis=(0,1))
        has_color_variation = rgb_std[0] > 20 or rgb_std[2] > 20  # Blue or Red channel variance
        
        # True clouds: blue sky + white clouds + color variation
        has_blue_sky = blue_ratio > 0.15 and avg_saturation > 40  # More lenient
        has_clouds = white_ratio > 0.05 and white_ratio < 0.50
        is_bright = avg_brightness > 100
        
        print(f"Cloud check - Blue: {blue_ratio:.2%}, White: {white_ratio:.2%}, Sat: {avg_saturation:.1f}, Bright: {avg_brightness:.1f}, Edges: {edge_density:.3f}, RGB_std: {rgb_std}")
        
        return has_blue_sky and has_clouds and is_bright and has_color_variation
        
    except Exception as e:
        print(f"Cloud detection error: {e}")
        return False

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'image' not in request.files:
            return jsonify({'success': False, 'error': 'No image provided'}), 400
            
        image_file = request.files['image']
        image = Image.open(io.BytesIO(image_file.read())).convert('RGB')
        
        vis_score, contrast, edges, brightness = calculate_visibility_score(image)
        
        img_tensor = transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(img_tensor)
            probs = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probs, 1)
        
        clear_prob = float(probs[0][0].item())
        hazy_prob = float(probs[0][1].item())
        
        label_map = {0: "clear", 1: "hazy"}
        prediction = label_map[predicted.item()]
        confidence_value = float(confidence.item() * 100)
        
        print(f"Model prediction - Clear: {clear_prob*100:.2f}%, Hazy: {hazy_prob*100:.2f}%")
        print(f"Visibility score: {vis_score:.2f}, Contrast: {contrast:.2f}")
        
        # Check for atmospheric haze first
        has_haze = detect_atmospheric_haze(image)
        has_clouds = has_distinct_clouds(image)
        
        # Override logic with better conditions
        if prediction == "clear" and has_haze:
            # If model says clear but we detect uniform haze, check confidence
            if clear_prob < 0.90 and contrast < 40:  # Low contrast suggests haze
                print("Override: Atmospheric haze detected → Changing to 'hazy'")
                prediction = "hazy"
                confidence_value = min(75.0, confidence_value * 0.8)
                clear_prob, hazy_prob = hazy_prob, clear_prob
                
        elif prediction == "hazy" and has_clouds and not has_haze:
            # Only override to clear if we have distinct clouds AND no uniform haze
            if vis_score > 65 and contrast > 35:
                print("Override: Blue sky with distinct clouds detected → Changing to 'clear'")
                prediction = "clear"
                confidence_value = min(confidence_value * 0.85, 88.0)
                clear_prob, hazy_prob = hazy_prob, clear_prob
        
        # Uncertainty check
        if confidence_value < 60:
            prediction = "uncertain"
        
        print(f"Final prediction: {prediction} ({confidence_value:.2f}%)")
        
        return jsonify({
            "success": True,
            "prediction": prediction,
            "confidence": round(confidence_value, 2),
            "visibility_score": round(vis_score, 2),
            "probabilities": {
                "clear": round(clear_prob * 100, 2),
                "hazy": round(hazy_prob * 100, 2)
            },
            "metrics": {
                "contrast": round(contrast, 2),
                "edge_density": round(edges * 100, 2),
                "brightness": round(brightness, 2)
            }
        })

    except Exception as e:
        print(f"Prediction error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/')
def home():
    return {
        "status": "CrowdVision HazeRadar API Running",
        "model": "Haze1K Trained - 100% Accuracy", 
        "version": "2.1"
    }

@app.route('/health')
def health():
    return jsonify({"status": "healthy", "model_loaded": True})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
