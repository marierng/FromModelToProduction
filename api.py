from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
import base64
from PIL import Image
import io

app = Flask(__name__)
model = load_model('fashion_model.h5')

# Define categories if needed
categories = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
              'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert('L').resize((28, 28))
    image = np.array(image).reshape(1, 28, 28, 1) / 255.0
    return image

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    image_b64 = data['image']
    image_bytes = base64.b64decode(image_b64)
    img = preprocess_image(image_bytes)
    predictions = model.predict(img)
    probs = predictions[0].tolist()
    return jsonify(dict(zip(categories, probs)))

if __name__ == '__main__':
    app.run(debug=True)


