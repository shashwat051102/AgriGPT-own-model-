import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing import image
from PIL import Image
import os
from dotenv import load_dotenv
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
import openai
# 
# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")

# Initialize model
model = ChatGroq(model="Gemma2-9b-It", groq_api_key=groq_api_key)
os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')
os.environ['LANGCHAIN_TRACING_V2'] = "true"
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")

# Define prompt and parser
parser = StrOutputParser()
prompt = ChatPromptTemplate(
    [
        ('system', "You are an agriculture expert helping a farmer diagnose plant diseases. Given the disease {predicted_class} and {name}, provide a general solution on prevention and treatment in {language}."),
        ('user', '{question}')
    ]
)
chain = prompt | model | parser

# Load the trained model
pathogen_model = keras.models.load_model("pathogen_classifier.h5")

# Define class labels
class_labels = ["Bacteria", "Fungus", "Healthy", "Pests", "Virus"]

# Streamlit UI
st.title('🌿AgriGPT')
st.subheader("Plant Disease Classifier and Agriculture Expert")
st.markdown(
        """
        <style>
        .main {
            background-color: #f0f2f6;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
        }
        .stImage {
            display: flex;
            justify-content: center;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display image
    img = Image.open(uploaded_file)
    img = img.resize((150, 150))  
    

    # Convert image to array
    img_array = np.array(img) / 255.0  
    img_array = np.expand_dims(img_array, axis=0)  

    # Debugging
    # st.write("Preprocessed Image Shape:", img_array.shape)

    # Make prediction
    predictions = pathogen_model.predict(img_array)
    predicted_class = class_labels[np.argmax(predictions)]
    confidence = np.max(predictions) * 100

    # Language input
    col1, col2 = st.columns(2)
    with col1:
        language = st.text_input("Enter the language you want the response in", value="en")
    with col2:
        name = st.text_input("Enter the name of the plant")

    
    
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(img, caption='Uploaded Image', use_container_width=True)
        # Show results
        st.success(f"**Predicted Class:** {predicted_class}")
        # st.info(f"**Confidence:** {confidence:.2f}%")
    with col2:
        # Get and display the response from the chain
        st.subheader("What you need to do ")
        response = chain.invoke({'question': predicted_class, 'predicted_class': predicted_class, 'name':name, 'language': language})
        st.write(response)


