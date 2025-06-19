from crewai import Task

def get_recovery_task(knowledge_tool, recovery_agent):
    return Task(
        description="After prevention and Treatment. Suggest fertilizers and nutrients to help {name} recover after {predicted_class}",
        expected_output="List of fertilizers, application tips, and timing for best recovery.",
        input_variables=["predicted_class", "name"],
        tools=[knowledge_tool],
        agent=recovery_agent
    )