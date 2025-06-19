# 🌿 AgriGPT - Plant Disease Classification and Agriculture Expert

AgriGPT is an intelligent plant disease classification and agricultural advisory system that combines deep learning, weather-aware recommendations, and generative AI. It helps farmers and gardeners identify plant diseases from images and provides tailored treatment and recovery advice based on current weather conditions and expert knowledge.

---

## 🌟 Features

- **Plant Disease Classification**: Upload images of plants to identify diseases using a trained deep learning model (`pathogen_classifier.h5`).
- **Weather-Aware Recommendations**: Get treatment advice tailored to your local weather conditions using the WeatherAPI.
- **Multi-language Support**: Receive responses in your preferred language.
- **PDF Report Generation**: Download detailed diagnosis and treatment reports as PDFs.
- **Email Notifications**: Receive diagnosis reports directly in your email inbox.
- **Recovery Guidance**: Get specific fertilizer and nutrient recommendations for plant recovery.
- **Knowledge Base**: Powered by Astra DB for storing and retrieving plant disease treatments and expert responses.

---

## 🛠️ Technologies Used

- **Frontend**: [Streamlit](https://streamlit.io/)
- **Machine Learning**: TensorFlow/Keras
- **Language Model**: Groq/Gemma2-9b-it, OpenAI GPT-4
- **Vector Database**: Astra DB (Cassandra)
- **Weather API**: [WeatherAPI.com](https://www.weatherapi.com/)
- **PDF Generation**: [xhtml2pdf](https://xhtml2pdf.readthedocs.io/)
- **Email**: SMTP (Gmail)
- **Other Libraries**: PIL, NumPy, dotenv, requests

---

## 📋 Prerequisites

- Python 3.8+
- Groq API Key
- OpenAI API Key
- Astra DB Credentials
- Weather API Key
- Gmail Account (for email notifications)

---

## 🔧 Installation

1. **Clone the repository:**
    ```bash
    git clone https://github.com/yourusername/agrigpt.git
    cd agrigpt
    ```

2. **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3. **Create a `.env` file in the project root with the following variables:**
    ```env
    OPENAI_API_KEY=your_openai_api_key
    GROQ_API_KEY=your_groq_api_key
    ASTRA_DB_APPLICATION_TOKEN=your_astra_db_token
    ASTRA_DB_ID=your_astra_db_id
    WEATHER_API_KEY=your_weather_api_key
    EMAIL_ADDRESS=your_gmail_address
    EMAIL_PASSWORD=your_gmail_app_password
    ```

---

## 🚀 Usage

1. **Start the Streamlit application:**
    ```bash
    streamlit run app.py
    ```

2. **Open your web browser and navigate to** `http://localhost:8501`

3. **Use the application:**
    - Enter your location (City, Country)
    - Upload a plant image
    - Specify your preferred language
    - Enter the plant name
    - (Optional) Provide your email for report delivery

---

## 📁 Project Structure

```
.
├── app.py                  # Main Streamlit application
├── pathogen_classifier.h5  # Trained Keras model for disease classification
├── requirements.txt        # Python dependencies
├── .env                    # Environment variables (not committed)
├── agents/                 # Agent logic for agriculture and recovery
│   ├── agriculture_agent.py
│   └── recovery_agent.py
├── tasks/                  # Task logic for diagnosis and recovery
│   ├── diagnosis_task.py
│   └── recovery_task.py
├── utils/                  # Utility modules (DB, email, PDF, weather)
│   ├── astra_db_utils.py
│   ├── email_utils.py
│   ├── pdf_utils.py
│   └── weather_utils.py
├── chroma_store/           # Vector store database (ignored in git)
└── README.md               # Project documentation
```

---

## 📸 Screenshots

_Add screenshots of your application here to showcase the UI and features._

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request or open an Issue for suggestions and improvements.

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- Thanks to all the open-source libraries and tools that made this project possible.
- Special thanks to the agricultural community for their valuable insights.

---

## 📧 Contact

For any questions or suggestions, please open an issue in the GitHub repository.

---

_Made with ❤️ for the agricultural community_
