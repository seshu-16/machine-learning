from flask import Flask, request, jsonify, send_from_directory, render_template
import os
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm

# Initialize Flask app
app = Flask(__name__, static_url_path='/static', template_folder='templates')

# Load model and data
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False
model = tf.keras.Sequential([model, GlobalMaxPooling2D()])

# Load precomputed features and filenames
feature_list = np.array(pickle.load(open('embeddings.pkl', 'rb')))
filenames = pickle.load(open('filenames.pkl', 'rb'))

# Ensure upload directory exists
UPLOAD_FOLDER = 'uploads'
STATIC_FOLDER = 'static/images'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
if not os.path.exists(STATIC_FOLDER):
    os.makedirs(STATIC_FOLDER)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file:
        # Save the uploaded file temporarily
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)

        try:
            # Extract features from the uploaded image
            features = extract_features(file_path, model)
            
            # Find similar images using k-NN
            neighbors = NearestNeighbors(n_neighbors=5, algorithm='brute', metric='euclidean')
            neighbors.fit(feature_list)
            distances, indices = neighbors.kneighbors([features])
            
            # Get the filenames of the recommended images
            recommended_files = [filenames[idx] for idx in indices[0]]
            
            # Ensure that the recommended files are valid and construct their paths
            recommended_files = [os.path.join(STATIC_FOLDER, os.path.basename(file)) for file in recommended_files]
            
            return jsonify(recommended_files)
        
        finally:
            # Delete the temporary file
            os.remove(file_path)

@app.route('/static/images/<path:path>')
def send_static(path):
    return send_from_directory(STATIC_FOLDER, path)

def extract_features(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)
    return normalized_result

if __name__ == '__main__':
    app.run(debug=True)
