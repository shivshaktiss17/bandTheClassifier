from flask import Flask, request, jsonify

from PIL import Image
from numpy import asarray
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import  preprocess_input
from keras.applications.inception_v3 import  decode_predictions

app = Flask(__name__)


@app.route('/')
def homepage():
    return "Hello...This is HeadBand Application!"


@app.route('/process', methods=["POST", "GET"])
def process():
    recv_image = request.files.get('image', '')

    if recv_image:
      image = Image.open(recv_image)
      image = image.resize((299, 299), Image.ANTIALIAS)
      image = asarray(image)

      image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
      image = preprocess_input(image)

      model = InceptionV3()
      yhat = model.predict(image)
      label = decode_predictions(yhat)

      label = label[0][0]
      print('%s (%.2f%%)' % (label[1].replace("_", " "), label[2]*100))

      return jsonify(response_value_1=1,response_value_2="Image Processed!", response=label[1].replace("_", " ") )
    else:
        return jsonify(response_value_1=0,response_value_2="Image Loading Failure!")
