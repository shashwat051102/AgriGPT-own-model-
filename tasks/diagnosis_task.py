from crewai import Task

def get_diagnosis_task(knowledge_tool, agriculture_agent):
    return Task(
        description="""
        Provide disease prevention and treatment steps for {predicted_class} affecting the plant {name} in {language}.

        Use your knowledge of plant diseases and agricultural best practices.
        Ensure your advice is tailored to the current weather:
        - Temperature: {Temperature}°C
        - Condition: {Condition}
        - Humidity: {Humidity}%
        - Wind: {Wind}
        - UV Index: {UV_index}
        """,
        expected_output="Prevention and treatment guidance tailored to the disease, plant, and current weather conditions.",
        input_variables=[
            "question", "predicted_class", "name", "language", "uploaded_file",
            "Temperature", "Humidity", "Condition", "Wind", "UV_index"
        ],
        tools=[knowledge_tool],
        agent=agriculture_agent,
    )