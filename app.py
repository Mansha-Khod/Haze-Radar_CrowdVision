import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import gradio as gr
import numpy as np

# ========== MODEL DEFINITION ==========
class HazeDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(1280, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 4)
        )
    def forward(self, x):
        return self.backbone(x)

# ========== LOAD MODEL ==========
model = HazeDetector()
model.load_state_dict(torch.load("haze_detector_colab.pth", map_location=torch.device('cpu')))
model.eval()

# ========== CATEGORY INFO ==========
CATEGORIES = {
    0: {"name": "Clear Sky", "aqi": 35, "advice": "✅ SAFE - Go outside freely!"},
    1: {"name": "Fog/Mist", "aqi": 60, "advice": "🌫️ Natural fog - Air is clean but visibility low"},
    2: {"name": "Light Haze", "aqi": 120, "advice": "⚠️ Light pollution - Limit outdoor time"},
    3: {"name": "Heavy Haze", "aqi": 180, "advice": "🚨 STAY INDOORS - Air is hazardous"}
}

# ========== IMAGE TRANSFORM ==========
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ========== PREDICTION FUNCTION ==========
def analyze_image(img):
    image = transform(img).unsqueeze(0)
    with torch.no_grad():
        outputs = model(image)
        probs = torch.softmax(outputs, dim=1)
        conf, pred = probs.max(1)
    category = CATEGORIES[pred.item()]
    confidence = round(conf.item() * 100, 2)
    return {
        "Predicted": category["name"],
        "Confidence": f"{confidence}%",
        "Estimated AQI": category["aqi"],
        "Advice": category["advice"]
    }

# ========== GRADIO UI ==========
demo = gr.Interface(
    fn=analyze_image,
    inputs=gr.Image(type="pil", label="Upload Sky / Outdoor Photo"),
    outputs="json",
    title="🌫️ CrowdVision AI",
    description="Upload an image of the sky or landscape. AI will detect haze/fog level and estimate air quality.",
    allow_flagging="never",
)

if __name__ == "__main__":
    demo.launch()












