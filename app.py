import patch_sqlite
import streamlit as st
from xhtml2pdf import pisa
import io
import numpy as np
from PIL import Image
import os
import requests
# from tensorflow.keras.models import load_model
# from tensorflow.keras.layers import InputLayer
from keras.models import load_model
from keras.layers import InputLayer
from dotenv import load_dotenv
from pydantic import PrivateAttr
from langchain_groq import ChatGroq
from langchain.docstore.document import Document
from langchain.tools import Tool
from crewai import Agent, Task, Crew
from crewai.tools import BaseTool
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from langchain_openai import ChatOpenAI
import openai
import cassio
from langchain.vectorstores.cassandra import Cassandra
from langchain.embeddings import HuggingFaceEmbeddings
from utils.email_utils import send_email
from utils.pdf_utils import create_pdf
from utils.weather_utils import get_weather
from agents.agriculture_agent import get_agriculture_agent
from agents.recovery_agent import get_recovery_agent
from tasks.diagnosis_task import get_diagnosis_task
from tasks.recovery_task import get_recovery_task
from utils.astra_db_utils import get_astra_vectorstore, store_response, similarity_search
# Load environment variables
load_dotenv()

# API Keys and configuration
openai_api_key = os.getenv("OPENAI_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")
weather_api = os.getenv("WEATHER_API_KEY")

# Validate required API keys
if not openai_api_key:
    st.error("OpenAI API key not found in environment variables.")
    st.stop()

if not groq_api_key:
    st.error("GROQ_API_KEY is not set. Please check your .env file.")
    st.stop()

if not weather_api:
    st.warning("Weather API key not found. Weather information will not be available.")

# Initialize OpenAI
openai.api_key = openai_api_key
llm = ChatOpenAI(api_key=openai_api_key, model="gpt-4.1-mini")


Astra_token = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
Astra_DB_ID = os.getenv("ASTRA_DB_ID")
# Simple Knowledge Base Tool (replaces Astra DB)
class AstraSearchTool(BaseTool):
    name: str = "Astra Search Tool"
    description: str = "Retrieves plant disease treatments from Astra DB based on similarity."
    _vectorstore = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._vectorstore = get_astra_vectorstore()

    def _run(self, query: str) -> str:
        return similarity_search(self._vectorstore, query, k=3)

    def store_response(self, query: str, response):
        store_response(self._vectorstore, query, response)
        

# Initialize tools
knowledge_tool = AstraSearchTool()

# Agent and Task definitions
agriculture_agent = get_agriculture_agent(knowledge_tool, llm)
recovery_agent = get_recovery_agent(knowledge_tool, llm)
diagnosis_task = get_diagnosis_task(knowledge_tool, agriculture_agent)
recovery_task = get_recovery_task(knowledge_tool, recovery_agent)

# Weather API
def get_weather(location):
    if not weather_api:
        return {"error": "Weather API key not configured"}
    
    try:
        base_url = f"http://api.weatherapi.com/v1/current.json?key={weather_api}&q={location}&aqi=no"
        response = requests.get(base_url, timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"Unable to fetch data: {response.status_code} - {response.text}"}
    except requests.exceptions.RequestException as e:
        return {"error": f"Network error: {e}"}
    except Exception as e:
        return {"error": f"Unexpected error: {e}"}

# Load the disease classification model
try:
    pathogen_model = load_model("pathogen_classifier.h5", compile=False, custom_objects={"InputLayer": InputLayer})
    class_labels = ["Bacteria", "Fungus", "Healthy", "Pests", "Virus"]
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

# UI Starts
st.set_page_config(page_title="🌿 AgriGPT", page_icon="🌱", layout="wide")

st.title('🌿 AgriGPT')
st.subheader("Plant Disease Classifier and Agriculture Expert")
st.markdown("""
<style>
.card {
    background: #fff;
    border-radius: 16px;
    box-shadow: 0 2px 8px rgba(60,60,60,0.07);
    padding: 1.5rem 1.5rem 1rem 1.5rem;
    margin-bottom: 1.5rem;
}
.card-header {
    font-weight: 700;
    font-size: 1.2rem;
    margin-bottom: 0.5rem;
    display: flex;
    align-items: center;
}
.card-badge {
    background: #ffe0b2;
    color: #b26a00;
    border-radius: 8px;
    padding: 0.2rem 0.7rem;
    font-size: 0.9rem;
    margin-left: 0.5rem;
}
.card-success {
    background: #e8f5e9;
    border-left: 5px solid #43a047;
}
.card-warning {
    background: #fffde7;
    border-left: 5px solid #fbc02d;
}
.card-info {
    background: #e3f2fd;
    border-left: 5px solid #1976d2;
}
.card-purple {
    background: #f3e5f5;
    border-left: 5px solid #8e24aa;
}
.card-step {
    margin-bottom: 0.7rem;
    padding: 0.7rem 1rem;
    border-radius: 10px;
    background: #f9fbe7;
    border-left: 4px solid #aed581;
}
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### 🌱 Welcome to AgriGPT!")
    st.markdown("Use this tool to diagnose plant diseases and get expert advice tailored to your location and weather conditions.")
    location = st.text_input("📍 Enter your location (City, Country):")
    uploaded_file = st.file_uploader("📷 Upload an image of the plant:", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    try:
        img = Image.open(uploaded_file)
        img = img.resize((150, 150))  
        
        # Convert image to array
        img_array = np.array(img) / 255.0  
        img_array = np.expand_dims(img_array, axis=0)  

        # Make prediction
        predictions = pathogen_model.predict(img_array, verbose=0)
        predicted_class = class_labels[np.argmax(predictions)]
        confidence = np.max(predictions) * 100

        col1, col2 = st.columns(2)
        with col1:
            language = st.sidebar.text_input("🌍 Preferred language for response:", value="en")
        with col2:
            name = st.sidebar.text_input("🌿 Name of the plant:")
            email = st.sidebar.text_input("📧 Enter your email to receive the response:")

        # Initialize weather variables with defaults
        Temperature = "N/A"
        Condition = "N/A"
        Humidity = "N/A"
        Wind = "N/A"
        UV_index = "N/A"

        col1, col2, col3 = st.columns(3)
        with col1:
            if location and weather_api:
                with st.spinner("Fetching weather data..."):
                    data = get_weather(location)
                    if "error" in data:
                        st.error(data["error"])
                    else:
                        try:
                            loc = data["location"]
                            current = data["current"]
                            condition = current["condition"]
                            Temperature = current['temp_c']
                            Condition = condition['text']
                            Humidity = current['humidity']
                            Wind = f"{current['wind_kph']} kph {current['wind_dir']}"
                            UV_index = current['uv']

                            st.markdown(f"### 📍 Weather in {loc['name']}, {loc['country']}")
                            st.markdown(f"""
                            - **🌡️ Temperature**: {Temperature}°C (Feels like {current['feelslike_c']}°C)
                            - **🌤️ Condition**: {Condition}
                            - **💧 Humidity**: {Humidity}%
                            - **🌬️ Wind**: {Wind}
                            - **☀️ UV Index**: {UV_index}
                            """)
                        except KeyError as e:
                            st.error(f"Unexpected weather data format: {e}")
            else:
                st.warning("⚠️ Please enter location and ensure weather API is configured.")
        with col2:
            st.image(img, caption='Uploaded Image', width=200)
        with col3:
            st.success(f"**Predicted Class:** {predicted_class}")
            st.info(f"**Confidence:** {confidence:.1f}%")

        st.markdown("---")
        
        st.subheader("📝 Recommended Actions")
        try:
            crew = Crew(agents=[agriculture_agent], tasks=[diagnosis_task])
            result = crew.kickoff({
                'question': predicted_class,
                'predicted_class': predicted_class,
                'name': name,
                'language': language,
                'Temperature': Temperature,
                'Condition': Condition,
                'Humidity': Humidity,
                'Wind': Wind,
                'UV_index': UV_index
            })
            st.markdown(result, unsafe_allow_html=True)
            
            result_str = str(result)
            
            # Show download button
            pdf = create_pdf(result_str)
            if pdf:
                st.sidebar.download_button(
                    label="📄 Download PDF Report",
                    data=pdf,
                    file_name=f"{name}_plant_diagnosis.pdf",
                    mime="application/pdf"
                )
                
            st.markdown("---")
            crew_recovery = Crew(agents=[recovery_agent], tasks=[recovery_task])
            recovery_result = crew_recovery.kickoff({        
                'question': predicted_class,
                'predicted_class': predicted_class,
                'name': name,
                'language': language,
                'Temperature': Temperature,
                'Condition': Condition,
                'Humidity': Humidity,
                'Wind': Wind,
                'UV_index': UV_index})
            st.markdown("### 🌱 Recovery & Fertilizer Advice")
            st.markdown(recovery_result)
            
            if email:
                if send_email(email, "🌿 AgriGPT Plant Diagnosis Report", result_str):
                    st.success("📧 Email sent successfully!")
                else:
                    st.error("❌ Failed to send email. Please try again.")
        except Exception as e:
            st.error(f"Error processing recommendation: {e}")
            
    except Exception as e:
        st.error(f"Error processing image: {e}")

st.markdown("---")
st.markdown("### 🌟 Thank you for using AgriGPT!")
st.markdown("Feel free to reach out for any feedback or suggestions.")
