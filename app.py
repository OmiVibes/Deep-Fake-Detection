import os
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from sklearn.metrics import accuracy_score
from PIL import Image

# Load the trained model
model_path = 'D:/Notes/Degree/Projects/DEEP FAKE/model/deepfake_detection_model_enhanced.h5'  # Path to your trained model
model = load_model(model_path)

app = Flask(__name__)

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    """Check if the file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_path, img_size=(224, 224)):
    """Preprocess the image for prediction"""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Error reading image: {image_path}")
    img = cv2.resize(img, img_size)  # Resize to match model input
    img = img / 255.0  # Normalize to [0, 1]
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

def predict_image(image_path):
    """Predicts whether the image is real or fake"""
    img = preprocess_image(image_path)
    prediction = model.predict(img)
    return 'Fake' if prediction[0][0] > 0.5 else 'Real'

# Serve the index page
@app.route('/')
def home():
    return render_template('index.html')  # Ensure index.html is in your templates folder

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join('uploads', filename)
        file.save(filepath)

        try:
            result = predict_image(filepath)
            return jsonify({'result': result})
        except Exception as e:
            return jsonify({'error': f"Error: {e}"})

    return jsonify({'error': 'Invalid file format'})

# Evaluate the model on test data
@app.route('/evaluate', methods=['GET'])
def evaluate_accuracy():
    test_folder = 'D:/Notes/Degree/Projects/DEEP FAKE/Data/test/'  # Path to your test folder
    y_true = []  # List to store actual labels
    y_pred = []  # List to store predicted labels

    # Loop through the "real" and "fake" subfolders
    for label, folder_name in zip([1, 0], ['real', 'fake']):  # 1 for real, 0 for fake
        folder_path = os.path.join(test_folder, folder_name)

        for filename in os.listdir(folder_path):
            if filename.endswith('.jpg') or filename.endswith('.png'):
                img_path = os.path.join(folder_path, filename)
                try:
                    result = predict_image(img_path)  # Predict label for the image
                    predicted_label = 1 if result == 'Real' else 0
                    y_true.append(label)
                    y_pred.append(predicted_label)
                except Exception as e:
                    print(f"Error processing {filename}: {e}")

    accuracy = accuracy_score(y_true, y_pred)
    return jsonify({'accuracy': accuracy * 100})  # Return accuracy in percentage

if __name__ == '__main__':
    app.run(debug=True)
