import pandas as pd
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
app = Flask(__name__)
@app.route('/',methods=['Get','Post'])
def index():
     return render_template('index.html')
@app.route('/predict',methods=['Get','Post'])
def predict():
     return render_template('Predict.html')
if __name__ == '__main__':
    app.run(port=5000,debug=True)
