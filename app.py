"""
Script by Susmita Banerjee
"""
from __future__ import division, print_function

# coding=utf-8
import os
import glob
import re
import numpy as np
import tensorflow as tf
# Flask utils
from flask import Flask, request, render_template
# Keras
from keras.applications.imagenet_utils import preprocess_input
from keras.models import load_model
from keras.applications.vgg16 import preprocess_input, decode_predictions
import keras.utils
from keras.preprocessing import image
from PIL import Image
# Flask utils
from flask import Flask, redirect, url_for, request, render_template, jsonify
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = 'leafsnap_model_vgg16.h5'

# Load your trained model
model = load_model(MODEL_PATH)
model.make_predict_function()          # Necessary
print('Model loaded. Start serving http://127.0.0.1:5000/')

# You can also use pretrained model from Keras
# Check https://keras.io/applications/
# from keras.applications.vgg16 import VGG16
# model = VGG16(weights='imagenet')
# model.save('')
# print('Model loaded. Check http://127.0.0.1:5000/')


def model_predict(img_path, model):
    img = tf.keras.utils.load_img(img_path, target_size=(224, 224))

    # Preprocessing the image
    x = tf.keras.utils.img_to_array(img)
    x = np.true_divide(x, 255)
    x = np.expand_dims(x, axis=0)

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    x = preprocess_input(x, data_format=None)

    preds = model.predict(x)
    return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)
        print('make predict', preds)
        # Process your result for human
        pred_proba = "{:.3f}".format(np.amax(preds))
        #pred_class = decode_predictions(preds, top=1)
        pred_class = preds.argmax(axis=-1)            # Simple argmax
        pred_class = decode_predictions(preds, top=1)   # ImageNet Decode
        result = str(pred_class[0][0][1])               # Convert to string
        result = result.replace('_', ' ').capitalize()

        # Serialize the result, you can add additional fields
        return jsonify(result=result, probability=pred_proba)

        return result
    return None


if __name__ == '__main__':
    app.run(debug=True)

