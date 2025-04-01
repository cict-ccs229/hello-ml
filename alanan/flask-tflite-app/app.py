from flask import Flask, request, render_template, jsonify, send_from_directory
import tensorflow.lite as tflite
import numpy as np
import cv2
import json
import os
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables    
load_dotenv()

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load label mappings
with open("alanan/flask-tflite-app/models/labels.json", "r", encoding="utf-8") as f:
    labels = json.load(f)
    label_dict = {item["label"]: item["name"] for item in labels}

# Load TFLite model
interpreter = tflite.Interpreter(model_path="alanan/flask-tflite-app/models/model.tflite")
interpreter.allocate_tensors()

# Get input/output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]['shape'][1:3]

# Configure Gemini AI
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (input_shape[1], input_shape[0]))
    image = image.astype(np.float32) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

def get_description(class_name):
    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(f"Describe {class_name}.")
    return response.text if response else "No information available."

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify():
    if 'file' not in request.files:
        return render_template('index.html', message='No file uploaded')
    
    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', message='No file selected')
    
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)
    
    input_data = preprocess_image(filepath)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    predicted_label = np.argmax(output_data)
    
    label_list = list(label_dict.keys())
    predicted_class = label_list[predicted_label]
    class_name = label_dict.get(predicted_class, "Unknown")
    
    description = get_description(class_name)
    
    return render_template('index.html', classification=class_name, description=description, image=file.filename)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000)) 
    app.run(host='0.0.0.0', port=port)