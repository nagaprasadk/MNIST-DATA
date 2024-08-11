import streamlit as st
import pickle
from PIL import Image
import numpy as np

# Title of the app
st.title('Digit recognition')

# File uploader widget
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

with open('model.pkl', 'rb') as file:
    model = pickle.load(file)


if uploaded_file is not None:
    # Open the image file
    image = Image.open(uploaded_file)
    imageg = image.convert('L')
    imageresized=imageg.resize((28,28),Image.ANTIALIAS)
    imagea = np.array(imageresized)

    # Display the image
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
# Make prediction
if st.button('Predict'):
    prediction = model.predict(imagea.reshape(1,28,28)).argmax(axis=1)
    st.write(f'The prediction is: {prediction[0]}')
    
