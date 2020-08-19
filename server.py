
import flask
from flask import Flask,render_template,url_for,request
import base64
import numpy as np
import cv2
from keras.models import load_model 
import io 
import json
import pickle
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input

with open("hdict.bin","rb") as fr:
    hdict = pickle.load(fr)
    
init_Base64 = 22   # data:image/png;base64, 로 시작하
app = Flask(__name__)

@app.route('/')
def home():
    return "ok~~~"


@app.route('/image', methods=['POST'])
def upload():       
    draw = request.form['photo_cap']
    draw = draw[init_Base64:]
    draw_decoded = base64.b64decode(draw)
    image = np.asarray(bytearray(draw_decoded), dtype="uint8")
    
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)    
    image = cv2.resize(image, dsize=(224, 224), interpolation=cv2.INTER_AREA) 
    image = image.reshape(-1,224,224,3)
    
    model = ResNet50(weights='imagenet')
    
    image=preprocess_input(image)
    pred = model.predict(image)

    r = hdict[np.argmax(pred)]

    return f"인식 결과: {r}"
    
if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=8000)
