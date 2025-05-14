import patch_sqlite
import streamlit as st
from xhtml2pdf import pisa
import io
import numpy as np
from PIL import Image
import os
import requests
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import InputLayer
from dotenv import load_dotenv
from pydantic import PrivateAttr
from langchain_groq import ChatGroq
from langchain_community.vectorstores import Cassandra
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from langchain.tools import Tool
from crewai_tools import SerperDevTool, MultiOnTool
from crewai import Agent, Task, Crew
from crewai.tools import BaseTool
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import cassio

# Load environment variables
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
Astra_token = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
Astra_DB_ID = os.getenv("ASTRA_DB_ID")
weather_api = os.getenv("WEATHER_API_KEY")

if not api_key:
    raise ValueError("GROQ_API_KEY is not set. Please check your .env file.")





# Astra Search Tool

Astra_token = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
Astra_DB_ID = os.getenv("ASTRA_DB_ID")

class AstraSearchTool(BaseTool):
    name: str = "Astra Search Tool"
    description: str = "Useful for retrieving past plant disease treatments from Astra DB based on similarity."

    # 👇 Declare vectorstore as private Pydantic attribute
    _vectorstore: Cassandra = PrivateAttr()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        cassio.init(
            database_id=Astra_DB_ID,
            token=Astra_token
        )
        embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

        self._vectorstore = Cassandra(
            embedding=embedding,
            table_name="plant_responses",
            session=None,
            keyspace=None
        )

    def _run(self, query: str) -> str:
        docs = self._vectorstore.similarity_search(query, k=3)
        if not docs:
            return "No relevant treatments found in Astra DB."
        return "\n\n".join([doc.page_content for doc in docs])

    def store_response(self, query: str, response):
        content = response.raw if hasattr(response, "raw") else str(response)
        document = Document(page_content=content, metadata={"query": query})
        self._vectorstore.add_documents([document])






# Initialize tools and LLM
astra_tool = AstraSearchTool()
search = SerperDevTool()
llm = ChatGroq(api_key=api_key, model="groq/gemma2-9b-it")



def create_pdf(content):
    pdf_buffer = io.BytesIO()
    pisa_status = pisa.CreatePDF(io.StringIO(content), dest=pdf_buffer)
    if pisa_status.err:
        return None
    pdf_buffer.seek(0)
    return pdf_buffer



# Agent definition
browser_agent = Agent(
    role="🌾 Agriculture Expert",
    goal="""
You are an agriculture expert helping a farmer diagnose plant diseases.
You retrieve relevant information from Astra DB and provide actionable insights.

The disease is: {predicted_class}
The plant is: {name}
The farmer speaks: {language}

Current weather conditions:
- Temperature: {Temperature}°C
- Condition: {Condition}
- Humidity: {Humidity}%
- Wind: {Wind}
- UV Index: {UV_index}

Use Astra DB to fetch disease-related knowledge and provide a weather-aware prevention and treatment strategy in the requested language.
""",
    backstory="A farmer uploaded an image of a diseased plant. You use Astra DB for plant disease data and current weather to advise the farmer.",
    tools=[astra_tool, search],  # Astra DB tool takes priority
    llm=llm,
    allow_delegation=True,
    verbose=True,
)




recovery_agent = Agent(
    role="🌿 Recovery Specialist",
    goal="After prevention and treatment, suggest fertilizers and nutrients to help the plant recover.",
    backstory="An expert in plant nutrition helping farmers after disease control.",
    tools=[astra_tool],
    llm=llm
)







# Task definition
browse_task = Task(
    description="""
        Provide disease prevention and treatment steps for {predicted_class} affecting the plant {name} in {language}.

        Use Astra DB to retrieve disease and plant information.
        Ensure your advice is tailored to the current weather:
        - Temperature: {Temperature}°C
        - Condition: {Condition}
        - Humidity: {Humidity}%
        - Wind: {Wind}
        - UV Index: {UV_index}
        """,
    expected_output="Prevention and treatment guidance tailored to the disease, plant, and current weather conditions, using Astra DB as a primary source.",
    input_variables=[
        "question", "predicted_class", "name", "language", "uploaded_file",
        "Temperature", "Humidity", "Condition", "Wind", "UV_index"
    ],
    tools=[astra_tool, search],
    agent=browser_agent,
)




recovery_task = Task(
    description="After prevention and Treatment. Suggest fertilizers and nutrients to help {name} recover after {predicted_class}",
    expected_output="List of fertilizers, application tips, and timing for best recovery.",
    input_variables=["predicted_class", "name"],
    tools=[astra_tool],
    agent=recovery_agent
)




# Weather API
def get_weather(location):
    base_url = f"http://api.weatherapi.com/v1/current.json?key={weather_api}&q={location}&aqi=no"
    response = requests.get(base_url)
    if response.status_code == 200:
        return response.json()
    else:
        return {"error": f"Unable to fetch data: {response.status_code} - {response.text}"}




def send_email(recipient_email, subject, body):
    email_sender = os.getenv("EMAIL_ADDRESS")
    email_password = os.getenv("EMAIL_PASSWORD")

    msg = MIMEMultipart()
    msg["From"] = email_sender
    msg["To"] = recipient_email
    msg["Subject"] = subject

    msg.attach(MIMEText(body, "html"))

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(email_sender, email_password)
            server.sendmail(email_sender, recipient_email, msg.as_string())
        return True
    except Exception as e:
        print(f"Failed to send email: {e}")
        return False




# Load the disease classification model
pathogen_model = load_model("pathogen_classifier.h5", compile=False, custom_objects={"InputLayer": InputLayer})

class_labels = ["Bacteria", "Fungus", "Healthy", "Pests", "Virus"]








# UI Starts
st.set_page_config(page_title="🌿 AgriGPT", page_icon="🌱", layout="wide")

st.title('🌿 AgriGPT')
st.subheader("Plant Disease Classifier and Agriculture Expert")
st.markdown("""
<style>
    .main {
        background-color: #f7f9fc;
        font-family: 'Arial', sans-serif;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
        font-size: 16px;
    }
    .stImage {
        display: flex;
        justify-content: center;
        margin-bottom: 20px;
    }
    .stTextInput>div>div>input {
        border-radius: 8px;
        padding: 10px;
        font-size: 14px;
    }
    .stFileUploader>div>div>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
        font-size: 14px;
    }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    # st.image("https://via.placeholder.com/150", caption="AgriGPT Logo", use_container_width=True)
    st.markdown("### 🌱 Welcome to AgriGPT!")
    st.markdown("Use this tool to diagnose plant diseases and get expert advice tailored to your location and weather conditions.")
    location = st.text_input("📍 Enter your location (City, Country):")
    uploaded_file = st.file_uploader("📷 Upload an image of the plant:", type=["jpg", "png", "jpeg"])


if uploaded_file is not None:
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

    col1, col2 = st.columns(2)
    with col1:
        language = st.sidebar.text_input("🌍 Preferred language for response:", value="en")
    with col2:
        name = st.sidebar.text_input("🌿 Name of the plant:")
        email = st.sidebar.text_input("📧 Enter your email to receive the response:")

    col1, col2,col3 = st.columns(3)
    with col1:
        if location and api_key:
            with st.spinner("Fetching weather data..."):
                data = get_weather(location)
                if "error" in data:
                    st.error(data["error"])
                else:
                    loc = data["location"]
                    current = data["current"]
                    condition = current["condition"]
                    Temperature = current['temp_c']
                    Condition = condition['text']
                    Humidity = current['humidity']
                    Wind = f"{current['wind_kph']} kph {current['wind_dir']}"
                    UV_index = current['uv']

                    st.markdown(f"### 📍 Weather in {loc['name']}, {loc['country']}")
                    # st.image("https:" + condition["icon"], width=64)
                    st.markdown(f"""
                    - **🌡️ Temperature**: {Temperature}°C (Feels like {current['feelslike_c']}°C)
                    - **🌤️ Condition**: {Condition}
                    - **💧 Humidity**: {Humidity}%
                    - **🌬️ Wind**: {Wind}
                    - **☀️ UV Index**: {UV_index}
                    """)
        else:
            st.warning("⚠️ Please enter both the location and your API key.")
    with col2:
        st.image(img, caption='Uploaded Image', width=200)
    with col3:
            st.success(f"**Predicted Class:** {predicted_class}")

    st.markdown("---")
    
    st.subheader("📝 Recommended Actions")
    crew = Crew(agents=[browser_agent], tasks=[browse_task],memory = False)
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
    astra_tool.store_response(f"{predicted_class} {name} {location}", result)
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
    crew_recovery = Crew(agents=[recovery_agent], tasks=[recovery_task],memory = False)
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

st.markdown("---")
st.markdown("### 🌟 Thank you for using AgriGPT!")
st.markdown("Feel free to reach out for any feedback or suggestions.")
