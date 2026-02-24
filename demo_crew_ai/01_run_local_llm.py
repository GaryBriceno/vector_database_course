# CrewAI + Ollama
# To interact with local LLM

from crewai import Agent, Task, Crew, Process, LLM

from demo_crew_ai.utils.llms_models import local_llm


agent = Agent(
    role="Local Assistant",
    goal="To answer question using local LLM",
    backstory="To run locally, without cloud APIs",
    llm=local_llm,
    verbose=True
)

task = Task(
    agent=agent,
    description="To explain the goal of CrewAI in five bullets",
    expected_output="5 bullets"
)

crew = Crew(
    agents=[agent],
    tasks=[task],
    process=Process.sequential
)

result = crew.kickoff()
