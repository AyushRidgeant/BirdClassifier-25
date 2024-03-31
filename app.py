import tensorflow as tf
import tensorflow_hub as hub
from flask import Flask, jsonify, request
import numpy as np
from PIL import Image
# Use Keras 2.
version_fn = getattr(tf.keras, "version", None)
if version_fn and version_fn().startswith("3."):
    import tf_keras as keras
else:
    keras = tf.keras

dct = {0: 'Asian Green Bee-Eater',
 1: 'Brown-Headed Barbet',
 2: 'Cattle Egret',
 3: 'Common Kingfisher',
 4: 'Common Myna',
 5: 'Common Rosefinch',
 6: 'Common Tailorbird',
 7: 'Coppersmith Barbet',
 8: 'Forest Wagtail',
 9: 'Gray Wagtail',
 10: 'Hoopoe',
 11: 'House Crow',
 12: 'Indian Grey Hornbill',
 13: 'Indian Peacock',
 14: 'Indian Pitta',
 15: 'Indian Roller',
 16: 'Jungle Babbler',
 17: 'Northern Lapwing',
 18: 'Red-Wattled Lapwing',
 19: 'Ruddy Shelduck',
 20: 'Rufous Treepie',
 21: 'Sarus Crane',
 22: 'White Wagtail',
 23: 'White-Breasted Kingfisher',
 24: 'White-Breasted Waterhen'}

class MyModel(keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.hub_layer = hub.KerasLayer("./b3", trainable=True)
        self.dense_layer1 = keras.layers.Dense(1024, activation='relu')
        self.dropout1 = keras.layers.Dropout(0.3)
        self.dense_layer2 = keras.layers.Dense(512, activation='relu')
        self.dropout2 = keras.layers.Dropout(0.3)
        self.dense_layer3 = keras.layers.Dense(512, activation='relu')
        self.dropout3 = keras.layers.Dropout(0.2)
        self.dense_output = keras.layers.Dense(25, activation='softmax')

    def call(self, inputs):
        x = self.hub_layer(inputs)
        x = self.dense_layer1(x)
        x = self.dropout1(x)
        x = self.dense_layer2(x)
        x = self.dropout2(x)
        x = self.dense_layer3(x)
        x = self.dropout3(x)
        return self.dense_output(x)


def load_model(model_path):

    model = MyModel()
    dummy_input = tf.zeros((1, 224, 224, 3))
    _ = model(dummy_input)  
    model.load_weights(model_path)

    return model

app = Flask(__name__)


@app.route("/")
def home():
    return "Hello, World!"


def preprocess_image(image):
    img = Image.open(image)
    img = img.resize((224, 224)) 
    img_array = np.array(img) / 255.0  
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'})
    model = load_model("./bird_classifier.h5")
    img = request.files['image']
    image_filename = img.filename
    print("Image filename:", image_filename)
    preprocessed_img = preprocess_image(img)
    prediction = model.predict(preprocessed_img)
    predicted_class = np.argmax(prediction[0])
    print(type(prediction))
    print(prediction)
    print(predicted_class)
    return jsonify({'class': dct[predicted_class]})

if __name__ == '__main__':
    app.run()