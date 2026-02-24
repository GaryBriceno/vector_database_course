from crewai import Agent, Task, Crew, Process, LLM

# LLM local via Ollama
local_llm = LLM(
    model="ollama/llama3.2",
    base_url="http://localhost:11434",
    temperature=0.2
)

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
print("\n=== RESULT ===\n")
print(result)