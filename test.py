import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np
import wikipediaapi
import requests

# Load the model
model = tf.keras.models.load_model(r'C:\Users\Raksha.N\Desktop\CNN\assets\trained_model_CNN.h5')

# Specify a generic user agent
wiki_wiki = wikipediaapi.Wikipedia(
    language='en',
    extract_format=wikipediaapi.ExtractFormat.WIKI,
    user_agent="MyApp/1.0"  # Replace with any user agent string
)

st.title("Cat or Dog Classifier")

uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image.", use_column_width=True)
    st.write("")
    
    # Preprocess the image for your model
    image = image.resize((64, 64))  # Resize to match the model's expected input shape
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    
    # Make prediction
    prediction = model.predict(image)
    
    # Get the class label
    if prediction > 0.5:
        predicted_class = "Dog"
    else:
        predicted_class = "Cat"
    
    st.write("Prediction:", predicted_class)
    
    # Fetch breed information and images from Wikipedia
    breed_name = st.text_input(f"Enter the {predicted_class.lower()} breed:")
    if breed_name:
        # Fetch breed information
        page = wiki_wiki.page(breed_name)
        if page.exists():
            breed_info = page.summary
            st.write(f"Breed Information for {breed_name.capitalize()} from Wikipedia:")
            st.write(breed_info)
        else:
            st.write("Breed information not found.")
        
        # Fetch breed images from Wikimedia Commons
        response = requests.get(f"https://en.wikipedia.org/w/api.php?action=query&titles={breed_name}&prop=images&format=json")
        data = response.json()
        images = data['query']['pages'][list(data['query']['pages'].keys())[0]].get('images', [])
        
        # Display images
        st.write(f"Images of {breed_name.capitalize()} from Wikimedia Commons:")
        for img in images:
            img_title = img['title']
            img_info_response = requests.get(f"https://en.wikipedia.org/w/api.php?action=query&titles=File:{img_title}&prop=imageinfo&iiprop=url&format=json")
            img_info_data = img_info_response.json()
            img_url = img_info_data['query']['pages'][list(img_info_data['query']['pages'].keys())[0]]['imageinfo'][0]['url']
            st.image(img_url, caption=img_title, use_column_width=True)
