import streamlit as st
import time
from PIL import Image, ImageOps
from tensorflow import keras
from keras.preprocessing import image
from img_classification import teachable_machine_classification

st.title("Image Classification with Google's Teachable Machine")
st.header("Traditional Kuih Classification")
st.subheader("This app is built to classify traditional kuih. Upload a traditional kuih image for image classification to identify what kuih it is")




uploaded_file = st.file_uploader("Upload a traditional kuih image...", type="jpg")
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded MRI.', use_column_width=True)
    st.write("")
        
    with st.spinner("Classifying..."):
        time.sleep(1)
        label = teachable_machine_classification(image, 'keras_model.h5')    
        if label == 0:
            st.success("This is a success message!")
            st.write("Kuih Lapis")
        elif label == 1:
            st.write("Onde-Onde")
        elif label == 2:
            st.write("Kuih Talam")
        else: 
            st.write("Not in class")
    
        
    

