import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Function for model prediction
def model_prediction(test_image):
    model = tf.keras.models.load_model("trained_plant_disease_model.keras")  
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))  
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.expand_dims(input_arr, axis=0)  
    predictions = model.predict(input_arr)
    return np.argmax(predictions)

# Sidebar and Page Selection
st.sidebar.title("Plant Disease System for Sustainable Agriculture")
app_mode = st.sidebar.selectbox('select page', ['Home', 'Disease Recognition'])

# Display home image
try:
    img = Image.open('D:\POTATO-DISEASE-PROJECT\datasets\Diseases.png')
    st.image(img)
except FileNotFoundError:
    st.warning("Diseases.png not found. Please check the file path.")

# Home Page
if app_mode == 'Home':  # FIXED
    st.markdown("<h1 style='text-align: center;'>Plant Disease Detection System for Sustainable Agriculture</h1>", 
                unsafe_allow_html=True)

# Disease Recognition Page
elif app_mode == "Disease Recognition":
    st.header('Plant Disease Detection System for Sustainable Agriculture')

    # File Uploader
    test_image = st.file_uploader('Choose an image:', type=['jpg', 'png', 'jpeg'])  

    if test_image is not None:
        if st.button('Show Image'):
            st.image(test_image, width=300, use_column_width=True)  

        if st.button('Predict'):
            st.snow()
            st.write('Model is predicting...')
            result_index = model_prediction(test_image)  
            class_names = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']
            st.success(f'Model predicts: {class_names[result_index]}')  
