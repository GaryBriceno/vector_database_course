# CrewAI + Ollama + Tools
# To read local files and write a resume using local Tools

from pathlib import Path
from crewai import Agent, Task, Crew, Process, LLM
from crewai_tools import DirectoryReadTool, FileReadTool

from demo_crew_ai.utils.llms_models import local_llm

docs_dir = Path("02_docs")
files = [str(p) for p in docs_dir.glob("**/*") if p.is_file()]

# Local Tools
dir_tool = DirectoryReadTool(directory="02_docs")
file_tool = FileReadTool()


# Agent 1: Find the files and decide the file to read
researcher_agent = Agent(
    role="Researcher",
    goal="Find and get information from local files",
    backstory="You are the best reading files and get information from them.",
    llm=local_llm,
    tools=[dir_tool, file_tool],
    verbose=True
)

writer_agent = Agent(
    role="Writer",
    goal="To write a final resume, very clearly.",
    backstory="To transform notes into a short text, very clearly.",
    llm=local_llm,
    verbose=True
)

task_research = Task(
    description=(
        f"You have this files:\n{files}\n\n"
        "Read 2-5 relevants files using the tool Read a file's content.\n"
        "IMPORTANT: Action Input need to be a JSON with a real values, for example:\n"
        '{"file_path":"02_docs/nota1.txt","start_line":1,"line_count":200}\n'
        "To return 5 bullets with findings."
    ),
    expected_output="3 bullets with the findings.",
    agent=researcher_agent,
)

task_write = Task(
    description=(
        "Based on the researcher's findings, write an 8–12 line summary explaining CrewAI + Ollama "
        "and how they connect in a local workflow. Include 3 practical steps to use it."
    ),
    expected_output="To resume 8-12 lines + 3 stepss.",
    agent=writer_agent,
    context=[task_research],  # To send the information from research to writer
)

crew = Crew(
    agents=[researcher_agent, writer_agent],
    tasks=[task_research, task_write],
    process=Process.sequential,
)

result = crew.kickoff()