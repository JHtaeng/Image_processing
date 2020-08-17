
import flask
from flask import Flask,render_template,url_for,request
import base64
import numpy as np
import cv2
from keras.models import load_model

init_Base64 = 21   # data:image/png;base64, 로 시작하
app = Flask(__name__)
model = load_model('mnist_cnn.h5')

@app.route('/')
def home():
    return render_template("mnist.html")


@app.route('/upload', methods=['POST'])
def upload():
    
    draw = request.form['url']        # hidden 변수로 전달함            
    draw = draw[init_Base64:]
    draw_decoded = base64.b64decode(draw)
    image = np.asarray(bytearray(draw_decoded), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, dsize=(28,28), interpolation=cv2.INTER_AREA)

    image = image.reshape(1,28,28,1)
    p = model.predict(image)
#     cv2.imwrite('test.png', image)
    return f"result: {np.argmax(p)}"

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=8000)
