from flask import Flask, render_template, request
import os
import numpy as np
import cv2

from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier

from efficientnet.tfkeras import EfficientNetB0

#librarie for loading ANN, EfficientNET, CNN, and hybrid models
from tensorflow.keras.models import load_model
#librarie for loading SGDClassifier, SVC, KNeighborsClassifier, and VotingClassifier models
from sklearn.externals import joblib

app = Flask(__name__)

#here we will load the trained models
sgd_clf = joblib.load('path_to_sgd_model.pkl')
svm_clf = joblib.load('path_to_svm_model.pkl')
knn_clf = joblib.load('path_to_knn_model.pkl')
voting_clf = joblib.load('path_to_voting_model.pkl')

ann_model = load_model('path_to_ann_model.h5')
efficientnet_model = load_model('path_to_efficientnet_model.h5')
cnn_model = load_model('path_to_cnn_model.h5')
cnn_elm_model = load_elm_model('path_to_elm_model.h5')

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    #gere we retrieve the selected type of model and the  image uploaded
    model_type = request.form['model_type']
    uploaded_image = request.files['image']

    #here we process the uploaded image
    img = cv2.imread(uploaded_image)
    img = cv2.resize(img, (224, 224))
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)

    #here we will perform predictions
    if model_type == 'sgd':
        prediction = sgd_clf.predict(img)
    elif model_type == 'svm':
        prediction = svm_clf.predict(img)
    elif model_type == 'knn':
        prediction = knn_clf.predict(img)
    elif model_type == 'ann':
        prediction = ann_model.predict(img)
    elif model_type == 'efficientnet':
        prediction = efficientnet_model.predict(img)
    elif model_type == 'cnn':
        prediction = cnn_model.predict(img)
    elif model_type == 'cnn_elm':
        prediction = cnn_elm_model.predict(img)

    return f'Prediction result: {prediction}'

