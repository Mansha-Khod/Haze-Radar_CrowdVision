

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2
from flask import Flask, request, jsonify
from flask_cors import CORS
from supabase import create_client
import os
import io
import logging

# ========================
# MODEL ARCHITECTURE
# ========================

class HazeClassifier(nn.Module):
    def __init__(self, num_classes=4):
        super(HazeClassifier, self).__init__()
        # Use pre-trained ResNet
        self.backbone = models.resnet50(pretrained=True)
        
        # Replace final layer for our classes
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.backbone.fc.in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        return self.backbone(x)

# ========================
# AI SERVICE
# ========================

class CrowdVisionAI:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.load_model()
        self.transform = self.get_transforms()
        self.class_names = ['clear', 'haze', 'fog', 'clouds']
        
    def load_model(self):
        """Load trained model - you'll train this with your datasets"""
        model = HazeClassifier(num_classes=4)
        
        # In production, this would load your trained weights
        # model.load_state_dict(torch.load('haze_model.pth', map_location=self.device))
        
        model.eval()
        return model.to(self.device)
    
    def get_transforms(self):
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def predict_image(self, image_data):
        """Real AI prediction"""
        try:
            # Convert to PIL Image
            image = Image.open(io.BytesIO(image_data))
            
            # Apply transforms
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Get prediction
            with torch.no_grad():
                outputs = self.model(image_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                predicted_class = torch.argmax(probabilities, 1).item()
                confidence = probabilities[0][predicted_class].item()
            
            class_name = self.class_names[predicted_class]
            
            # Map to air quality
            aq_mapping = {
                'clear': ('Good', 'Clear sky with excellent visibility'),
                'haze': ('Poor', 'Haze detected. Limit outdoor activities'),
                'fog': ('Moderate', 'Fog detected. Reduced visibility'),
                'clouds': ('Good', 'Cloudy but clear air')
            }
            
            air_quality, message = aq_mapping.get(class_name, ('Unknown', 'Unable to determine'))
            
            return {
                'air_quality': air_quality,
                'message': message,
                'confidence': round(confidence * 100, 2),
                'detected_condition': class_name
            }
            
        except Exception as e:
            logging.error(f"Prediction error: {e}")
            return {
                'air_quality': 'Unknown',
                'message': 'Analysis failed',
                'confidence': 0,
                'detected_condition': 'error'
            }

# ========================
# TRAINING SCRIPT
# ========================

def train_model():
    """Train the model on your datasets"""
    import torch.optim as optim
    from torch.utils.data import DataLoader, Dataset
    import pandas as pd
    from sklearn.model_selection import train_test_split
    
    class SkyDataset(Dataset):
        def __init__(self, image_paths, labels, transform=None):
            self.image_paths = image_paths
            self.labels = labels
            self.transform = transform
            
        def __len__(self):
            return len(self.image_paths)
        
        def __getitem__(self, idx):
            image = Image.open(self.image_paths[idx]).convert('RGB')
            label = self.labels[idx]
            
            if self.transform:
                image = self.transform(image)
                
            return image, label
    
    # Load your datasets here
    # You'll need to organize your downloaded datasets into:
    # datasets/
    #   clear/
    #   haze/ 
    #   fog/
    #   clouds/
    
    def load_dataset():
        data = []
        labels = []
        class_mapping = {'clear': 0, 'haze': 1, 'fog': 2, 'clouds': 3}
        
        for class_name, class_id in class_mapping.items():
            class_dir = f'datasets/{class_name}'
            if os.path.exists(class_dir):
                for img_file in os.listdir(class_dir):
                    if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        data.append(os.path.join(class_dir, img_file))
                        labels.append(class_id)
        
        return data, labels
    
    # Training setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = HazeClassifier(num_classes=4).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Load data
    image_paths, labels = load_dataset()
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths, labels, test_size=0.2, random_state=42
    )
    
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = SkyDataset(train_paths, train_labels, train_transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    # Training loop
    for epoch in range(10):
        model.train()
        running_loss = 0.0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        print(f'Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}')
    
    # Save model
    torch.save(model.state_dict(), 'haze_classifier.pth')
    print("Training complete! Model saved.")

# ========================
# FLASK APP
# ========================

app = Flask(__name__)
CORS(app)

# Initialize AI and Supabase
ai_service = CrowdVisionAI()
supabase = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_KEY")
)

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
        
        # REAL AI ANALYSIS
        analysis_result = ai_service.predict_image(image_data)
        
        # Save to database
        upload_data = {
            "latitude": latitude,
            "longitude": longitude,
            "air_quality": analysis_result['air_quality'],
            "message": analysis_result['message'],
            "confidence": analysis_result['confidence'],
            "detected_condition": analysis_result['detected_condition'],
            "timestamp": "now()"
        }

        supabase.table("crowdvision_submissions").insert(upload_data).execute()

        return jsonify({
            "success": True,
            **analysis_result
        }), 200

    except Exception as e:
        logging.error(f"Upload error: {e}")
        return jsonify({"error": "Failed to process photo"}), 500

@app.route('/crowdvision/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "service": "CrowdVision AI"})

if __name__ == '__main__':
    # Uncomment to train model first:
    # train_model()
    
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port)






