import streamlit as st
import time
from PIL import Image, ImageOps
from tensorflow import keras
from keras.preprocessing import image
from img_classification import teachable_machine_classification

st.title("Traditional Kuih Classification")
st.header("- This app is built to classify traditional kuih")
st.subheader("Upload a traditional kuih image for image classification to identify what kuih it is.")

import base64
def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
    )
add_bg_from_local('background.png')  


uploaded_file = st.file_uploader("Upload a traditional kuih image...", type="jpg")
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded MRI.', use_column_width=True)
    st.write("")
        
    with st.spinner("Classifying..."):
        time.sleep(1)
        label, accuracy = teachable_machine_classification(image, 'keras_model.h5')    
        
        if accuracy > 0.9:
            st.success('This is a success message!')
            if label == 0:        
                st.write(f'(Accuracy: {accuracy:.2f})')
                st.write("Kuih Lapis")
                with st.expander("See recipe"):
                    st.write("Ingredients:")
                    st.write("1 cup rice flour")
                    st.write("1/2 cup wheat flour")
                    st.write("2 tbsp tapioca flour")
                    st.write("1/2 cup sugar")
                    st.write("1 tsp salt")
                    st.write("2 1/2 cups coconut milk")
                    st.write("1 tsp red food coloring")
                    st.write("1/2 tbsp oil")
            elif label == 1:
                st.write(f'(Accuracy: {accuracy:.2f})')
                st.write("Onde-Onde")   
            elif label == 2:
                st.write(f'(Accuracy: {accuracy:.2f})')
                st.write("Kuih Talam")
        else:
            st.write("Unable to identify")
    
        
    

