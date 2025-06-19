from crewai import Agent

def get_recovery_agent(knowledge_tool, llm):
    return Agent(
        role="🌿 Recovery Specialist",
        goal="After prevention and treatment, suggest fertilizers and nutrients to help the plant recover.",
        backstory="An expert in plant nutrition helping farmers after disease control.",
        tools=[knowledge_tool],
        llm=llm
    )