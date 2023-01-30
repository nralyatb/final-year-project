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
                    st.write("**Ingredients**:")
                    st.write("- 1 cup rice flour")
                    st.write("- 1/2 cup wheat flour")
                    st.write("- 2 tbsp tapioca flour")
                    st.write("- 1/2 cup sugar")
                    st.write("- 1 tsp salt")
                    st.write("- 2 1/2 cups coconut milk")
                    st.write("- 1 tsp red food coloring")
                    st.write("- 1/2 tbsp oil")

                    st.write("**Instructions**:")
                    st.write("1. Prepare steamer over medium heat.")
                    st.write("2. Mix all dry ingredients in a bowl with a whisk or wooden spoon.")
                    st.write("3. Add coconut milk into the flour mixture and mix thoroughly until smooth. Ensure that there are no clumps.")
                    st.write("4. Divide the batter into two portions. Add red food coloring into one of the portions. Mix well.")
                    st.write("5. Using an 8 inch square or round mould, grease it with a bit of oil or line it with food-grade plastic. Place it into the steamer and pour 1/3 cup of pink batter into the mould. Cover with the lid to steam the batter for 5 minutes. Then, pour another 1/3 cup of white batter to steam. Repeat the process until batter is used up.")
            elif label == 1:
                st.write(f'(Accuracy: {accuracy:.2f})')
                st.write("Onde-Onde")
                with st.expander("See recipe"):
                    st.write("**Ingredients**:")
                    st.write("2 tbsp water")
                    st.write("100 g coconut desiccated")
                    st.write("1 tsp salt")
                    st.write("120 g glutinous rice flour")
                    st.write("60 g tapioca flour")
                    st.write("50 g sugar")
                    st.write("120 ml pandan leaf extract")
                    st.write("140 g palm sugar (gula melaka)")
                    
                    st.write("**Instructions**:")
                    st.write("1. Add water and salt to the desiccated coconut. Mix well and steam the mixture for 15 to 20 minutes.")
                    st.write("2. Mix glutinous rice flour, tapioca flour and sugar evenly before adding pandan leaf extract to create a dough. If the dough is too soft, add more glutinous rice flour.")
                    st.write("3. Divide the dough into 14 little balls. Flatten dough and wrap chunks of 5g palm sugar in it. Be careful that the dough is not too thin as it will expand during the cooking process. This will result in the dough cracking and the palm sugar flowing out.")
                    st.write("4. Put the little balls of filled dough into a pot of boiling water. Dish it out once they float or leave to boil for a further 5 to 10 minutes to allow the palm sugar to melt thoroughly.")
                    st.write("5. Coat 'onde-onde' with the steamed desiccated coconut. Leave to cool and enjoy!")         

            elif label == 2:
                st.write(f'(Accuracy: {accuracy:.2f})')
                st.write("Kuih Talam")
                with st.expander("See recipe"):
                    st.write("**Ingredients**:")
                    st.write("For green layer:")
                    st.write("- 300 ml pandan juice")
                    st.write("- 1 tsp alkaline water/lye water")
                    st.write("- 100 g rice flour")
                    st.write("- 65 g tapioca starch")
                    st.write("- 35 g green bean flour/mung bean flour")
                    st.write("- 400 ml plain water")
                    st.write("- 180 g granulated sugar")
                    st.write("- ½ tsp fine salt")

                    st.write("For white layer:")
                    st.write("- 390 g thick coconut milk")
                    st.write("- ¾ tsp fine salt")
                    st.write("- 70 g rice flour")
                    st.write("- 30 g green bean flour/mung bean flour")
                    st.write("- 140 ml hot water")

                    st.write("**Instructions**")
                    st.write("1. Coat 8-inch (20-cm) square baking pan with oil, line with baking paper, coat again with oil.")
                    st.write("2. *Pandan juice: Add 300 ml of water to 55 g pandan leaves, blend until smooth, extract 300 ml of pandan juice.")
                    st.write("3. Preparing green layer: Mix together pandan juice, alkaline water, rice flour, tapioca starch, and green bean flour. Combine plain water, sugar, and salt in a sauce pot, bring to the boil. Add hot sugar syrup to green mixture gradually while stirring, then strain mixture into a mixing bowl.")
                    st.write("4. Bring ½ pot of water to boil, turn to medium heat, sit mixing bowl with green mixture on hot water, cook until mixture has thickened, remove from heat, and mix until smooth.")
                    st.write("5. Pour thickened green mixture into the prepared baking pan (create an uneven surface to help both layers stick together), steam over medium heat for 15-16 minutes.")
                    st.write("6. Preparing white layer: Mix together thick coconut milk, salt, rice flour, and green bean flour. Add in hot water gradually while stirring, then strain mixture into a mixing bowl.")
                    st.write("7. Using the same method as in the green mixture, cook white mixture until thickened, remove from heat, and mix until smooth.")
                    st.write("8. Pour thickened white mixture onto cooked green layer, steam over medium heat for 15-16 minutes.")
                    st.write("9. Set aside the Steamed Kuih Talam to cool completely.")
                    st.write("10. When cooled, cut into pieces using an oil-coated knife (clean the knife with oil-coated kitchen towel after several cuts before continuing with the remaining).")

                
        else:
            st.error('Error to Identify', icon="🚨")
    
        
    

