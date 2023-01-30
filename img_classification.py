import tensorflow as tf
from keras.models import load_model
from tensorflow import keras
from keras.preprocessing import image
from PIL import Image, ImageOps
import numpy as np


def teachable_machine_classification(img, weights_file):
    # Load the model
    model = keras.models.load_model(weights_file)

    # Create the array of the right shape to feed into the keras model
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    image = img
    #image sizing
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)

    #turn the image into a numpy array
    image_array = np.asarray(image)
    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

    # Load the test data and labels
    # x_test, y_test = ...

    # Evaluate the model on the test data
    # loss, accuracy = model.evaluate(x_test,y_test)

    # Load the image into the array
    data[0] = normalized_image_array

    # Output the accuracy
    # st.write(f'Accuracy: {accuracy:.2f}')

    # run the inference
    prediction = model.predict(data)

    # Calculate the accuracy of the prediction
    accuracy = np.max(prediction)
  #  accuracy = np.max(prediction)

    return np.argmax(prediction), accuracy # return position of the highest probability
