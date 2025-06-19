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

# --- Adaptive Glassmorphism CSS & Animations ---
st.markdown("""
<style>
:root {
  --card-bg: rgba(255,255,255,0.82);
  --card-bg-dark: rgba(30,30,30,0.72);
  --border: rgba(200,200,200,0.22);
  --border-dark: rgba(60,60,60,0.32);
  --shadow: 0 8px 32px 0 rgba(31,38,135,0.18);
  --shadow-dark: 0 8px 32px 0 rgba(0,0,0,0.28);
  --accent: #7c3aed;
  --accent-light: #a5b4fc;
  --accent-bg: #ede9fe;
  --accent-bg-dark: #2a213a;
  --success-bg: #e0f7fa;
  --success-bg-dark: #1b2b2b;
  --info-bg: #f3e8ff;
  --info-bg-dark: #1a2233;
  --expander-bg: linear-gradient(90deg,#ede9fe 60%,#f3e8ff 100%);
  --expander-bg-dark: linear-gradient(90deg,#2a213a 60%,#1a2233 100%);
}
@media (prefers-color-scheme: dark) {
  :root {
    --card-bg: var(--card-bg-dark);
    --border: var(--border-dark);
    --shadow: var(--shadow-dark);
    --accent-bg: var(--accent-bg-dark);
    --success-bg: var(--success-bg-dark);
    --info-bg: var(--info-bg-dark);
    --expander-bg: var(--expander-bg-dark);
  }
}
body { overflow-x: hidden; }
.glass-card {
    background: var(--card-bg);
    border-radius: 18px;
    box-shadow: var(--shadow);
    backdrop-filter: blur(7px);
    -webkit-backdrop-filter: blur(7px);
    border: 1px solid var(--border);
    padding: 1.5rem 1.5rem 1.2rem 1.5rem;
    margin-bottom: 1.5rem;
    transition: box-shadow 0.3s, transform 0.2s, background 0.3s;
}
.glass-card:hover {
    box-shadow: 0 12px 36px 0 var(--accent);
    transform: translateY(-2px) scale(1.01);
}
.stepper {
    display: flex;
    justify-content: space-between;
    margin-bottom: 1.5rem;
    gap: 0.5rem;
}
.step {
    flex: 1;
    background: var(--accent-bg);
    border-radius: 12px;
    padding: 0.7rem 0.5rem;
    text-align: center;
    font-weight: 700;
    color: var(--accent);
    border: 2px solid #d1c4e9;
    position: relative;
    letter-spacing: 0.5px;
    transition: background 0.2s, border 0.2s, color 0.2s;
}
@media (prefers-color-scheme: dark) {
  .step {
    background: var(--accent-bg-dark);
    border: 2px solid #3a2a4d;
    color: #a5b4fc;
  }
}
.step.active, .step.completed {
    background: linear-gradient(90deg,var(--accent-light),var(--accent)11);
    border: 2px solid var(--accent);
    color: #222;
}
@media (prefers-color-scheme: dark) {
  .step.active, .step.completed {
    color: #fff;
    background: linear-gradient(90deg,#a5b4fc99,var(--accent)33);
    border: 2px solid var(--accent);
  }
}
.step.completed:after {
    content: "✓";
    position: absolute;
    right: 12px;
    top: 8px;
    color: #43a047;
    font-size: 1.1rem;
}
.card-header {
    font-weight: 700;
    font-size: 1.1rem;
    color: var(--accent);
    margin-bottom: 0.5rem;
}
.glass-card-info {
    background: linear-gradient(90deg,#f3e8ff 60%,#ede9fe 100%);
}
@media (prefers-color-scheme: dark) {
  .glass-card-info {
    background: linear-gradient(90deg,#2a213a 60%,#1a2233 100%);
  }
}
.glass-card-success {
    background: linear-gradient(90deg,#e0f7fa 60%,#ede9fe 100%);
}
@media (prefers-color-scheme: dark) {
  .glass-card-success {
    background: linear-gradient(90deg,#1b2b2b 60%,#2a213a 100%);
  }
}
.sticky-summary {
    position: sticky;
    top: 12px;
    z-index: 10;
}
.animated-thankyou {
    animation: fadeIn 1.2s;
    background: linear-gradient(90deg,var(--accent-bg) 60%,#e0f7fa 100%);
    border-radius: 18px;
    box-shadow: 0 4px 24px var(--accent)22;
    margin-top: 2rem;
    padding: 1.5rem 1rem;
    transition: background 0.3s;
}
@media (prefers-color-scheme: dark) {
  .animated-thankyou {
    background: linear-gradient(90deg,var(--accent-bg-dark) 60%,#1b2b2b 100%);
  }
}
::-webkit-scrollbar-thumb { background: var(--accent)33; border-radius: 8px;}
::-webkit-scrollbar { width: 8px;}
.progress-bar-outer {
    width: 100%;
    height: 10px;
    background: var(--accent-bg);
    border-radius: 8px;
    margin: 0.7rem 0;
    box-shadow: none; /* Remove any glow or shadow */
}
@media (prefers-color-scheme: dark) {
  .progress-bar-outer { background: var(--accent-bg-dark); }
}
.progress-bar-inner {
    height: 100%; border-radius: 8px;
    background: linear-gradient(90deg,var(--accent-light),var(--accent));
    transition: width 0.4s;
}
.st-expanderHeader {
    font-weight: 700 !important;
    font-size: 1.08rem !important;
    color: var(--accent) !important;
    background: var(--expander-bg) !important;
    border-radius: 10px !important;
    padding: 0.6rem 1rem !important;
    margin-bottom: 0.2rem !important;
    box-shadow: 0 2px 8px #7c3aed22;
}
@media (prefers-color-scheme: dark) {
  .st-expanderHeader {
    background: var(--expander-bg-dark) !important;
  }
}
</style>
""", unsafe_allow_html=True)

# --- Sidebar with icons and quick tips ---
with st.sidebar:
    st.markdown("## 🌱 **AgriGPT**")
    st.markdown(
        "<span style='font-size:1rem; color:var(--accent);'>Diagnose plant diseases and get expert advice tailored to your location and weather.</span>",
        unsafe_allow_html=True,
    )
    st.markdown("---")
    location = st.text_input("📍 Location:", placeholder="e.g. Nairobi, Kenya", key="loc", help="City, Country")
    uploaded_file = st.file_uploader("📷 Upload Image", type=["jpg", "png", "jpeg"], key="img", help="Clear close-up of affected area")
    st.markdown("---")
    st.markdown(
        "<span style='font-size:0.95rem; color:var(--accent);'>💡 <b>Tip:</b> For best results, upload a clear image and provide accurate location.</span>",
        unsafe_allow_html=True,
    )

# --- Animated Stepper with progress bar ---
step = 1
if uploaded_file: step = 2
if uploaded_file and location: step = 3
# progress_pct = int((step-1)/2*100)
# st.markdown(f"""
# <div class="stepper">
#     <div class="step {'active' if step==1 else 'completed' if step>1 else ''}">1. Upload Image</div>
#     <div class="step {'active' if step==2 else 'completed' if step>2 else ''}">2. Location</div>
#     <div class="step {'active' if step==3 else ''}">3. Results</div>
# </div>
# <div class="progress-bar-outer">
#   <div class="progress-bar-inner" style="width:{progress_pct}%"></div>
# </div>
# """, unsafe_allow_html=True)

# --- Main Card Layout ---
with st.container():
    # st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### 🌿 Plant Disease Diagnosis")
    st.write("Upload a plant image and enter your location to get instant diagnosis and recommendations.")

    if uploaded_file is not None:
        try:
            img = Image.open(uploaded_file)
            img = img.resize((150, 150))
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            with st.spinner("🔎 Analyzing image..."):
                progress = st.progress(0)
                for i in range(1, 101, 10):
                    progress.progress(i)
                    import time; time.sleep(0.04)
                predictions = pathogen_model.predict(img_array, verbose=0)
                progress.progress(100)

            predicted_class = class_labels[np.argmax(predictions)]
            confidence = np.max(predictions) * 100

            # --- Advanced Options ---
            with st.expander("🔧 Advanced Options (customize language, plant name, and get your report by email!)", expanded=True):
                st.markdown(
                    "<span style='color:var(--accent);font-size:1rem;'>✨ Personalize your results below!</span>",
                    unsafe_allow_html=True,
                )
                language = st.text_input("🌍 Preferred language:", value="en", help="e.g. en, sw, hi")
                name = st.text_input("🌿 Plant name:")
                email = st.text_input("📧 Email for report:")

            # --- Responsive Columns ---
            col1, col2, col3 = st.columns([1.2,1,1])
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
                                st.markdown(f"""
                                <div class="glass-card" style="background:var(--info-bg);">
                                <div class="card-header">📍 Weather in {loc['name']}, {loc['country']}</div>
                                <ul>
                                    <li>🌡️ <b>Temperature:</b> {Temperature}°C (Feels like {current['feelslike_c']}°C)</li>
                                    <li>🌤️ <b>Condition:</b> {Condition}</li>
                                    <li>💧 <b>Humidity:</b> {Humidity}%</li>
                                    <li>🌬️ <b>Wind:</b> {Wind}</li>
                                    <li>☀️ <b>UV Index:</b> {UV_index}</li>
                                </ul>
                                </div>
                                """, unsafe_allow_html=True)
                            except KeyError as e:
                                st.error(f"Unexpected weather data format: {e}")
                else:
                    st.warning("⚠️ Please enter location and ensure weather API is configured.")

            with col2:
                st.image(img, caption='Uploaded Image', use_container_width=True, output_format="PNG")
                st.markdown('<div style="text-align:center; color:var(--accent); font-size:0.95rem;">Zoom for details</div>', unsafe_allow_html=True)
            with col3:
                st.markdown(f"""
                <div class="glass-card sticky-summary" style="background:var(--success-bg);">
                    <div class="card-header">🌿 Prediction</div>
                    <b>Class:</b> <span style="color:var(--accent)">{predicted_class}</span><br>
                    <b>Confidence:</b> {confidence:.1f}%
                </div>
                """, unsafe_allow_html=True)

            st.markdown("---")

            # --- Collapsible Results Section ---
            with st.expander("📝 Recommended Actions", expanded=True):
                st.markdown("### 📝 Recommended Actions")
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

                    # --- Download PDF Button with icon ---
                    pdf = create_pdf(result_str)
                    if pdf:
                        st.download_button(
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

                    # --- Toast notification for email ---
                    if email:
                        if send_email(email, "🌿 AgriGPT Plant Diagnosis Report", result_str):
                            st.success("📧 Email sent successfully!")
                        else:
                            st.error("❌ Failed to send email. Please try again.")
                except Exception as e:
                    st.error(f"Error processing recommendation: {e}")

        except Exception as e:
            st.error(f"Error processing image: {e}")

    st.markdown('</div>', unsafe_allow_html=True)

# --- Animated Thank You Card ---
st.markdown("""
<div class="animated-thankyou" style="text-align:center;">
    <h3>🌟 Thank you for using AgriGPT!</h3>
    <p>We hope your plants thrive.<br>For feedback or suggestions, reach out anytime.</p>
</div>
""", unsafe_allow_html=True)

