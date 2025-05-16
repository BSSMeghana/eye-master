from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import numpy as np
import tensorflow as tf
import os

app = Flask(__name__)
CORS(app)

# Load TFLite model and allocate tensors
interpreter = tf.lite.Interpreter(model_path='eye_disease_model.tflite')
interpreter.allocate_tensors()

# Get input and output tensor details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

class_names = ['cataract', 'diabetic_retinopathy', 'glaucoma', 'normal']
image_size = (224, 224)

@app.route('/')
def home():
    return "Backend is running"

@app.route('/predict', methods=['POST'])
def predict():
    print("Received predict request")
    if 'file' not in request.files:
        print("No file in request")
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        print("Empty file received")
        return jsonify({'error': 'Empty file received'}), 400

    try:
        # Just open and verify image, no prediction
        img = Image.open(file).convert('RGB').resize(image_size)
        print("Image loaded and resized")
        
        # Dummy prediction response:
        result = {
            'model': 'Eye Disease Classifier',
            'name': 'dummy_class',
            'predicted_class': 0,
            'accuracy': "99.99%",
            'remedy': 'No remedy for dummy class'
        }
        print("Sending dummy prediction result")
        return jsonify({'result': [result]})
    except Exception as e:
        print(f"Error in prediction: {e}")
        return jsonify({'error': str(e)}), 500



def suggest_remedy(class_name):
    remedies = {
        'cataract': 'Consult an ophthalmologist for surgery options.',
        'diabetic_retinopathy': 'Maintain blood sugar levels and get regular eye checkups.',
        'glaucoma': 'Use prescribed eye drops and monitor eye pressure regularly.',
        'normal': 'Your eyes appear normal. Maintain a healthy lifestyle.'
    }
    return remedies.get(class_name, 'No specific remedy found.')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))

