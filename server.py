
import cv2
import numpy as np
import datetime
from flask import Flask, request, render_template, redirect

image = None 

app = Flask(__name__)

def chromakey(img, background):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    patch = hsv[0:20, 0:20, :] 

    minH = np.min(patch[:,:,0])*0.9
    maxH = np.max(patch[:,:,0])*1.1

    minS = np.min(patch[:,:,1])*0.9
    maxS = np.max(patch[:,:,1])*1.1

    h = hsv[:,:,0]
    s = hsv[:,:,1]

    dest = img.copy()

    for r in range(img.shape[0]):
        for c in range(img.shape[1]):
            if h[r,c] >= minH and h[r,c] <= maxH and s[r,c] >= minS and s[r,c] <= maxS:
                dest[r,c,:] = background[r,c,:]
            else:
                dest[r,c,:] = img[r,c,:]
                
    return dest

@app.route('/')
def index():
   
    return render_template("imageprocessing.html", ctx={"title":"영상처리"})

@app.route('/upload', methods=["POST"])
def upload():
    global image
    f = request.files['file1']
    filename = "./static/" + f.filename
    f.save(filename)
    
    image = cv2.imread(filename)
    cv2.imwrite('./static/result.jpg', image)
    print(image.shape)
    
    return redirect("/")

@app.route('/imageprocess')
def imageprocess():
    global image
    method = request.args.get("method")
    if method == "emboss":
        print(image.shape)
        print("emboss")
        
        emboss = np.array([[-1,-1,0],
                           [-1,0,1],
                           [0,1,1]], np.float32)

        dst = cv2.filter2D(image, -1,emboss, delta=128)
        cv2.imwrite("./static/result.jpg", np.vstack((image,dst)))
        
        
    if method == "sharp":
        print(image.shape)
        
        sharp = np.array([[0,-1,0],
                           [-1,5,-1],
                           [0,-1,0]], np.float32)

        dst = cv2.filter2D(image, -1, sharp)
        cv2.imwrite("./static/result.jpg", np.vstack((image,dst)))
        
        
    if method == "blur":
        size = int(request.args.get("size",3))
        dst = cv2.blur(image,(size,size))
        cv2.imwrite("./static/result.jpg", np.vstack((image,dst)))
    
    return "hello!~@~!@~!@~!@~!@~!@~"

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=8000)
