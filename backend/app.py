from flask import Flask, request, jsonify
from flask_cors import CORS  
from PIL import Image
import numpy as np
import tensorflow as tf

app = Flask(__name__)
CORS(app)  

# Initialize TFLite interpreter properly
interpreter = tf.lite.Interpreter(model_path='eye_disease_model.tflite')
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

class_names = ['cataract', 'diabetic_retinopathy', 'glaucoma', 'normal']
image_size = (224, 224)

@app.route('/')
def home():
    return "Backend is running"

@app.route('/predict', methods=['POST'])
def predictt():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Empty file received'}), 400

    try:
        img = Image.open(file).convert('RGB').resize(image_size)
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()
        predictions = interpreter.get_tensor(output_details[0]['index'])

        class_index = int(np.argmax(predictions[0]))
        accuracy = float(np.max(predictions[0])) * 100.0

        result = {
            'model': 'Eye Disease Classifier',
            'name': class_names[class_index],
            'predicted_class': class_index,
            'accuracy': f"{accuracy:.2f}%",
            'remedy': suggest_remedy(class_names[class_index])
        }

        return jsonify({'result': [result]})
    except Exception as e:
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
    app.run(debug=True)
