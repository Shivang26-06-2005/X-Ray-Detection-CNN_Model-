from flask import Flask, request, jsonify
import torch
import torch.nn as nn
import cv2
import numpy as np
import base64
from PIL import Image
import io
import os
from flask_cors import CORS
import gc

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# -----------------------------
# Model definition
# -----------------------------
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=4):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(2,16,3,padding=1)
        self.conv2 = nn.Conv2d(16,32,3,padding=1)
        self.conv3 = nn.Conv2d(32,64,3,padding=1)
        self.conv4 = nn.Conv2d(64,128,3,padding=1)
        self.pool = nn.MaxPool2d(2,2)
        self.relu = nn.ReLU()
        self.adapt_pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(128, num_classes)

    def forward(self,x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = self.pool(self.relu(self.conv4(x)))
        x = self.adapt_pool(x)
        x = x.view(x.size(0),-1)
        x = self.fc(x)
        return x

# -----------------------------
# Global model loading (Render-friendly)
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "XrayDetection.pth")  # Adjust filename if needed

model = SimpleCNN(num_classes=4).to(device)
try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print("Model loaded successfully!")
except FileNotFoundError:
    print(f"Warning: {MODEL_PATH} not found. Place the model in the project root.")
except Exception as e:
    print(f"Error loading model: {e}")

CLASS_NAMES = ["COVID", "Normal", "Lung_Opacity", "Viral Pneumonia"]

# -----------------------------
# Image preprocessing
# -----------------------------
def preprocess_image(image_data, is_base64=True):
    """Preprocess image data for model prediction"""
    try:
        if is_base64:
            image_data = base64.b64decode(image_data.split(',')[1])
            image = Image.open(io.BytesIO(image_data))
        else:
            image = Image.open(io.BytesIO(image_data))
        
        if image.mode != 'L':
            image = image.convert('L')
        
        image_array = np.array(image)
        image_resized = cv2.resize(image_array, (256, 256))
        image_normalized = image_resized / 255.0
        return image_normalized
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None

# -----------------------------
# Prediction function
# -----------------------------
def predict_images(xray_image, mask_image):
    try:
        xray_processed = preprocess_image(xray_image)
        mask_processed = preprocess_image(mask_image)
        
        if xray_processed is None or mask_processed is None:
            return None, "Error processing images"
        
        combined = np.stack([xray_processed, mask_processed], axis=0)  # Shape: (2, 256, 256)
        input_tensor = torch.tensor(combined, dtype=torch.float32).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            predicted_class = torch.argmax(outputs, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        
        # Free memory
        del input_tensor, outputs
        gc.collect()
        
        return {
            'predicted_class': CLASS_NAMES[predicted_class],
            'confidence': float(confidence),
            'all_probabilities': {
                CLASS_NAMES[i]: float(probabilities[0][i])
                for i in range(len(CLASS_NAMES))
            }
        }, None
        
    except Exception as e:
        return None, f"Prediction error: {str(e)}"

# -----------------------------
# Routes
# -----------------------------
@app.route('/')
def index():
    """Serve main webpage"""
    html_path = os.path.join(BASE_DIR, 'WebPage.html')
    if os.path.exists(html_path):
        with open(html_path, 'r', encoding='utf-8') as f:
            return f.read()
    else:
        return "<h1>Error: WebPage.html not found</h1><p>Place it in the project root.</p>"

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    try:
        data = request.json
        if not data or 'xray_image' not in data or 'mask_image' not in data:
            return jsonify({'success': False, 'error': 'Both xray_image and mask_image are required'}), 400
        
        result, error = predict_images(data['xray_image'], data['mask_image'])
        if error:
            return jsonify({'success': False, 'error': error}), 500
        
        return jsonify({'success': True, 'prediction': result})
    
    except Exception as e:
        return jsonify({'success': False, 'error': f'Server error: {str(e)}'}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'model_path': MODEL_PATH,
        'device': str(device),
        'classes': CLASS_NAMES,
        'server_info': 'Simple COVID-19 X-ray Classification Server'
    })

# -----------------------------
# Run only locally
# -----------------------------
if __name__ == '__main__':
    print(f"Starting Flask server on device: {device}")
    print(f"Model classes: {CLASS_NAMES}")
    app.run(debug=True, host='0.0.0.0', port=5000)
