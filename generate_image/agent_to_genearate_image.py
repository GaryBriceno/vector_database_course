# crew_run.py
from crewai import Agent, Task, Crew
from generate_image.generate_image import generate_visual

visual_agent = Agent(
    role="Visual Generator",
    goal="Generar imágenes offline de alta calidad con prompts claros y útiles",
    backstory=(
        "Eres un agente experto en convertir ideas en prompts visuales precisos "
        "y generar imágenes localmente usando un pipeline diffusion."
    ),
    tools=[generate_visual],
    verbose=True,
)

task = Task(
    description=(
        "Genera una imagen para este concepto:\n"
        "Prompt base: 'un desarrollador en estilo anime caminando por una carretera, "
        "con un dragón verde guardián volando detrás con alas extendidas, iluminación cinematográfica'.\n"
        "Asegúrate de que el prompt final sea detallado y optimizado."
    ),
    expected_output="La ruta del archivo PNG generado por la herramienta generate_visual.",
    agent=visual_agent,
)

crew = Crew(
    agents=[visual_agent],
    tasks=[task],
    verbose=True,
)

result = crew.kickoff()
print(result)