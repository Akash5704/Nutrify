from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model('Food.keras')

# Define class names
class_names = ['Besan_cheela', 'Biryani', 'Chapathi', 'Chole_bature', 'Dahl', 'Dhokla', 'Dosa', 'Gulab_jamun',
               'Idli', 'Jalebi', 'Kadai_paneer', 'Naan', 'Paani_puri', 'Pakoda', 'Pav_bhaji', 'Poha', 'Rolls', 'Samosa',
               'Vada_pav', 'chicken_curry', 'chicken_wings', 'donuts', 'fried_rice', 'grilled_salmon', 'hamburger',
               'ice_cream', 'not_food', 'pizza', 'ramen', 'steak', 'sushi']

# Function to load and preprocess image
def load_and_prep_image(image_path, img_shape=224, scale=False):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [img_shape, img_shape])
    if scale:
        img = img / 255.
    return img

@app.route('/')
def home():
    return "Keras Model API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    # Secure filename and save temporarily
    filename = secure_filename(file.filename)
    filepath = os.path.join("temp", filename)
    os.makedirs("temp", exist_ok=True)
    file.save(filepath)
    
    # Process the image
    img = load_and_prep_image(filepath)
    img = tf.expand_dims(img, axis=0)
    
    # Make predictions
    pred = model.predict(img)
    predicted_class = class_names[np.argmax(pred[0])]
    confidence_score = float(np.max(pred[0]))
    
    # Remove the temporary file
    os.remove(filepath)
    
    return jsonify({'prediction': predicted_class, 'confidence': confidence_score})

if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0',port=5000)
