# 🌿 AgriGPT - Plant Disease Classification and Agriculture Expert

AgriGPT is an intelligent plant disease classification and agricultural advisory system that combines machine learning with weather-aware recommendations. It helps farmers and gardeners identify plant diseases and provides tailored treatment and recovery advice based on current weather conditions.

## 🌟 Features

- **Plant Disease Classification**: Upload images of plants to identify diseases using a trained deep learning model
- **Weather-Aware Recommendations**: Get treatment advice tailored to your local weather conditions
- **Multi-language Support**: Receive responses in your preferred language
- **PDF Report Generation**: Download detailed diagnosis and treatment reports
- **Email Notifications**: Receive diagnosis reports directly in your email
- **Recovery Guidance**: Get specific fertilizer and nutrient recommendations for plant recovery
- **Knowledge Base**: Powered by Astra DB for storing and retrieving plant disease treatments

## 🛠️ Technologies Used

- **Frontend**: Streamlit
- **Machine Learning**: TensorFlow/Keras
- **Language Model**: Groq/Gemma2-9b-it
- **Vector Database**: Astra DB (Cassandra)
- **Weather API**: WeatherAPI.com
- **PDF Generation**: xhtml2pdf
- **Email**: SMTP (Gmail)

## 📋 Prerequisites

- Python 3.8+
- Groq API Key
- Astra DB Credentials
- Weather API Key
- Gmail Account (for email notifications)

## 🔧 Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/agrigpt.git
cd agrigpt
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the project root with the following variables:
```env
GROQ_API_KEY=your_groq_api_key
ASTRA_DB_APPLICATION_TOKEN=your_astra_db_token
ASTRA_DB_ID=your_astra_db_id
WEATHER_API_KEY=your_weather_api_key
EMAIL_ADDRESS=your_gmail_address
EMAIL_PASSWORD=your_gmail_app_password
```

## 🚀 Usage

1. Start the Streamlit application:
```bash
streamlit run app.py
```

2. Open your web browser and navigate to `http://localhost:8501`

3. Use the application:
   - Enter your location
   - Upload a plant image
   - Specify your preferred language
   - Enter the plant name
   - (Optional) Provide your email for report delivery

## 📸 Screenshots

[Add screenshots of your application here]

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Thanks to all the open-source libraries and tools that made this project possible
- Special thanks to the agricultural community for their valuable insights

## 📧 Contact

For any questions or suggestions, please open an issue in the GitHub repository.

---

Made with ❤️ for the agricultural community 