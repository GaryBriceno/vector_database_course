from crewai import LLM

# LLM local via Ollama
local_llm = LLM(
    model="ollama/llama3.2",
    base_url="http://localhost:11434",
    temperature=0.2
)