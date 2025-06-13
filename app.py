from flask import Flask, render_template, request, redirect
from markupsafe import Markup
import numpy as np
import pandas as pd
from utils.disease import disease_dic
from utils.fertilizer import fertilizer_dic
import requests
import config
import pickle
import io


# Allowed file extensions for disease image prediction
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ==============================================================================================

# -------------------------LOADING THE TRAINED MODELS -----------------------------------------------

# Loading plant disease classification model
disease_classes = [...]  # List of disease classes

disease_model_path = 'models/plant_disease_model.pth'
crop_recommendation_model_path = 'models/RandomForest.pkl'

try:
    disease_model = ResNet9(3, len(disease_classes))
    disease_model.load_state_dict(torch.load(disease_model_path, map_location=torch.device('cpu')))
    disease_model.eval()
except Exception as e:
    print(f"Error loading disease model: {e}")

try:
    with open(crop_recommendation_model_path, 'rb') as file:
        crop_recommendation_model = pickle.load(file)
except Exception as e:
    print(f"Error loading crop recommendation model: {e}")

# =========================================================================================

# Custom functions for calculations
def weather_fetch(city_name):
    ...
    
def predict_image(img, model=disease_model):
    ...
    
# ===============================================================================================

# ------------------------------------ FLASK APP -------------------------------------------------

app = Flask(__name__)

@app.route('/')
def home():
    title = 'Harvestify - Home'
    return render_template('index.html', title=title)

@app.route('/crop-recommend')
def crop_recommend():
    title = 'Harvestify - Crop Recommendation'
    return render_template('crop.html', title=title)

@app.route('/fertilizer')
def fertilizer_recommendation():
    title = 'Harvestify - Fertilizer Suggestion'
    return render_template('fertilizer.html', title=title)

@app.route('/crop-predict', methods=['POST'])
def crop_prediction():
    title = 'Harvestify - Crop Recommendation'

    if request.method == 'POST':
        ...
        # Prediction logic here
        return render_template('crop-result.html', prediction=final_prediction, title=title)

@app.route('/fertilizer-predict', methods=['POST'])
def fert_recommend():
    title = 'Harvestify - Fertilizer Suggestion'
    crop_name = str(request.form['cropname'])
    N = int(request.form['nitrogen'])
    P = int(request.form['phosphorous'])
    K = int(request.form['pottasium'])
    ...
    return render_template('fertilizer-result.html', recommendation=response, title=title)

@app.route('/disease-predict', methods=['GET', 'POST'])
def disease_prediction():
    title = 'Harvestify - Disease Detection'
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files.get('file')
        if file and allowed_file(file.filename):
            try:
                img = file.read()
                prediction = predict_image(img)
                prediction = Markup(str(disease_dic[prediction]))
                return render_template('disease-result.html', prediction=prediction, title=title)
            except Exception as e:
                print(f"Error during prediction: {e}")
                return render_template('try_again.html', title=title)
        else:
            return render_template('try_again.html', title=title)
    return render_template('disease.html', title=title)

if __name__ == '__main__':
    app.run(debug=True)
