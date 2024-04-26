from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
from PIL import Image
import traceback


# Path to your custom image and model
model_path = r'C:\Users\Emotion Detection\Flask\VGG19_New3.h5'

current_dir = os.path.dirname(__file__)
model_path = os.path.join(current_dir, 'VGG19_New3.h5')

def predictEmotion(imagefile):
    
    try:
        print(model_path)
        print('Load the trained model...')
        model = tf.keras.models.load_model(model_path)

        print('Load and preprocess the image...')
        imgLoad = Image.open(imagefile)
        
        # Resize the image
        print('Resize the image...')
        img_resized = imgLoad.resize((224, 224))
        
        # img = image.load_img(img_resized, target_size=(224, 224))
        img_array = image.img_to_array(img_resized)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = tf.keras.applications.vgg19.preprocess_input(img_array)

        # Make predictions
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions, axis=1)

        # Optionally, map the predicted class index back to a class label
        class_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Neutral']
        predictedEmotion = class_labels[predicted_class[0]]
        print(f'Predicted class: {predictedEmotion}')
        
        return predictedEmotion
    
    except Exception as e:
        traceback.print_exc()
        print("Error!:", e)
        return None
