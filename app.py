# import os
# import numpy as np
# import tensorflow as tf
# import boto3
# import requests
# from flask import Flask, render_template, request, redirect, url_for
# from werkzeug.utils import secure_filename
# import json


# # Flask app setup
# app = Flask(__name__)
# UPLOAD_FOLDER = './static/uploads'
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# app.config['ALLOWED_EXTENSIONS'] = {'jpg', 'jpeg', 'png'}

# ########################################

# # S3 Config
# bucket_name = 'skin-cancer-models-01'
# s3_key = 'skin_cancer_model.keras'
# local_model_path = 'models/skin_cancer_model.keras'

# # Ensure 'models' directory exists
# os.makedirs('models', exist_ok=True)

# # Download if not already present
# if not os.path.exists(local_model_path):
#     print("Downloading model from S3...")
#     s3 = boto3.client('s3')
#     s3.download_file(bucket_name, s3_key, local_model_path)
#     print("Model downloaded.")

# # Load the model locally
# # model = tf.keras.models.load_model(local_model_path)


# #########################################

# # model_url = 'https://skin-cancer-models-01.s3.ap-south-1.amazonaws.com/skin_cancer_model.keras'
# # model_path = './models/skin_cancer_model.keras'

# # # Download the model if not already present
# # if not os.path.exists(model_path):
# #     print("Downloading model...")
# #     r = requests.get(model_url)
# #     os.makedirs('models', exist_ok=True)
# #     with open(model_path, 'wb') as f:
# #         f.write(r.content)

# # # Load the model
# # model = tf.keras.models.load_model(model_path)

# #########################################

# # Load the trained model and class names
# # model = tf.keras.models.load_model('./models/skin_cancer_model.keras')
# # model = tf.keras.models.load_model(model_path)
# model = tf.keras.models.load_model(local_model_path)
# with open('./data/class_names.json', 'r') as f:
#     class_names = json.load(f) # Loaded class names from JSON

#     # on printing
#     # class_names = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']

#     # # Define the class names (this should correspond to your model's output labels)
#     # class_names = [
#     #     "Actinic Keratoses and Intraepithelial Carcinoma",  # Class 0
#     #     "Basal Cell Carcinoma",  # Class 1
#     #     "Benign Keratosis-like Lesions"  # Class 2
#     #     "Dermatofibroma",  # Class 3
#     #     "Melanoma",  # Class 4
#     #     "Melanocytic Nevi",  # Class 5
#     #     "Vascular Lesion",  # Class 6
#     # ]

# # Function to delete all files in upload folder
# def clear_uploads():
#     for filename in os.listdir(UPLOAD_FOLDER):
#         file_path = os.path.join(UPLOAD_FOLDER, filename)
#         try:
#             if os.path.isfile(file_path):
#                 os.remove(file_path)
#         except Exception as e:
#             print(f"Error deleting file {file_path}: {e}")

# # Helper function to check allowed file types
# def allowed_file(filename):
#     return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# # Error handler for 404 errors
# @app.errorhandler(404)
# def page_not_found(e):
#     return render_template('error.html', error = { "msg": "Page Not Found", "status": 404}), 404

# # Route for homepage
# @app.route('/')
# def index():
#     clear_uploads()  # Clear previous files on page load
#     return render_template('index.html')

# # Route for prediction
# @app.route('/predict', methods=['POST'])
# def predict():
#     # Check if a file is present in the request
#     if 'file' not in request.files:
#         return "No file part in the request", 400

#     file = request.files['file']

#     # Check if the file is empty
#     if file.filename == '':
#         return "No selected file", 400

#     # Validate file type and process
#     if file and allowed_file(file.filename):
#         clear_uploads()  # Clear old files before saving new one

#         # Secure and save the uploaded file
#         filename = secure_filename(file.filename)
#         filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#         file.save(filepath)

#         try:
#             # Load and preprocess the image
#             img = tf.keras.preprocessing.image.load_img(filepath, target_size=(128, 128))
#             img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
#             img_array = np.expand_dims(img_array, axis=0)

#             # Make prediction
#             prediction = model.predict(img_array)
#             predicted_class_index = np.argmax(prediction[0])
#             predicted_class_name = class_names[str(predicted_class_index)]
#             probability = prediction[0][predicted_class_index] * 100

#             # Render the result on the webpage
#             return render_template(
#                 'index.html',
#                 prediction=predicted_class_name,
#                 probability=f"{probability:.2f}%",
#                 image_path=filepath
#             )

#         except Exception as e:
#                 # Handle errors during prediction
#                 return render_template('error.html', error={"msg": f"Prediction Error: {str(e)}", "status": 500}), 500

#     # Return error for invalid file types
#     return render_template(
#             'error.html',
#             error = {"msg" : "Invalid File Type", "status": 400},
#         ), 400


# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5000, debug=True)


####################################################################################################################

import os
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import json

# Flask app setup
app = Flask(__name__)
UPLOAD_FOLDER = './static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['ALLOWED_EXTENSIONS'] = {'jpg', 'jpeg', 'png'}


# Ensure required directories exist
os.makedirs('models', exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Model and class names loading
model_path = 'models/skin_cancer_model.keras'
model = tf.keras.models.load_model(model_path)

with open('./data/class_names.json', 'r') as f:
    class_names = json.load(f)


# Function to clear uploaded files
def clear_uploads():
    for filename in os.listdir(UPLOAD_FOLDER):
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
        except Exception as e:
            print(f"Error deleting file {file_path}: {e}")

# Validate file extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


# 404 error handler
@app.errorhandler(404)
def page_not_found(e):
    return render_template('error.html', error={"msg": "Page Not Found", "status": 404}), 404

# Home route
@app.route('/')
def index():
    clear_uploads()
    return render_template('index.html')
# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('error.html', error={"msg": "No file part in the request", "status": 400}), 400

    file = request.files['file']

    # Check if the file is empty
    if file.filename == '':
        return render_template('error.html', error={"msg": "No selected file", "status": 400}), 400

    # Validate file type and process
    if file and allowed_file(file.filename):
        clear_uploads()
        # Secure and save the uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)


        try:
            # Load and preprocess the image
            img = tf.keras.preprocessing.image.load_img(filepath, target_size=(128, 128))
            img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension


            # Make prediction
            prediction = model.predict(img_array)
            predicted_class_index = np.argmax(prediction[0])  # Get the class index
            predicted_class_name = class_names[str(predicted_class_index)]  # Fetch class name from JSON
            probability = prediction[0][predicted_class_index] * 100  # Convert to percentage

            if predicted_class_name == 'non_cancerous_skin': 
                predicted_class_name = '✅ Great news! No signs of skin cancer detected.'

            if predicted_class_name == 'non_skin': 
                return render_template('error.html', error={"msg": "Unprocessable Entity: Please Input Valid Skin Image!", "status": 422}), 422

            return render_template(
                'index.html',
                prediction=predicted_class_name,
                probability=f"{probability:.2f}%",
                image_path=filepath
            )

        except Exception as e:
            # Handle errors during prediction
            return render_template('error.html', error={"msg": f"Prediction Error: {str(e)}", "status": 500}), 500

    # Return error for invalid file types
    return render_template('error.html', error={"msg": "Invalid File Type", "status": 400}), 400

# Run the app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
