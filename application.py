import flask
from flask import Flask, request, render_template
import numpy as np
from model import get_prediction
import os
app = Flask(__name__)

@app.route("/")
@app.route("/index")
def index():
    return flask.render_template('index.html')


@app.route('/predict', methods=['POST'])
def make_prediction():
	if request.method=='POST':

		# get uploaded image file if it exists
		file = request.files['image']
		if not file: 
			return render_template('index.html', label="No file")

		img_bytes = file.read()
		class_id, class_name = get_prediction(image_bytes=img_bytes)

		return render_template('index.html', label=str(class_name))


if __name__ == '__main__':
	app.run(port=int(os.environ.get('PORT', 5000)), debug=True)
