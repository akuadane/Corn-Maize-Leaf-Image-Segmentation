
import os
import pickle
import cv2
from flask import Flask,jsonify,request
import numpy as np


app = Flask(__name__)

IMAGE_DIMS=(50,50,1)
MODEL="final_model.sav"
loaded_model = pickle.load(open(MODEL, 'rb'))


@app.route('/upload', methods=['POST'])
def upload():
   
    filename = "image.jpg"
    file = open(filename,"wb")
    file.write(request.get_data())

    features = processImage(filename)
    pred = loadAndPredict(features)

    return jsonify({"prediction": pred})


def fd_hu_moments(image):
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    feature = cv2.HuMoments(cv2.moments(image)).flatten()
    return feature

def fd_histogram(image, mask=None):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([image], [0,1,2], None, [8, 8, 8], [0,256,0,256,0,256])
    cv2.normalize(hist, hist)
    return hist.flatten()

def processImage(imagePath):
    image = cv2.imread(imagePath)
    image = cv2.resize(image,(IMAGE_DIMS[1],IMAGE_DIMS[0]))

    fv_histogram = fd_histogram(image)
    
    image= cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    fv_hu_moments = fd_hu_moments(image)
    
    features = np.hstack([fv_histogram,fv_hu_moments])

    return [features]

def loadAndPredict(features):
    return loaded_model.predict(features)[0]

#run server
if(__name__ == "__main__"):
    app.run(debug=True)