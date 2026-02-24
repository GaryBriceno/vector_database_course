from datetime import date
from crewai import Agent, Task, Crew, Process, LLM

# -----------------------------
# Local LLM (Ollama)
# -----------------------------
local_llm = LLM(
    model="ollama/llama3.2",
    base_url="http://localhost:11434",
    temperature=0.4,
)

# -----------------------------
# Daily inputs (edit these)
# -----------------------------
BRAND = {
    "niche": "personal finance + AI/tech",
    "audience": "young professionals in LATAM (English content)",
    "tone": "clear, practical, friendly, no hype",
    "objective": "educate + drive comments",
    "constraints": [
        "No promises of financial returns",
        "No copyrighted lyrics or long quotes",
        "Avoid legal/medical advice; add a short disclaimer if needed",
        "Caption <= ~1500 characters",
    ],
    "cta_style": "end with a question to encourage comments",
}

TOPIC_OF_THE_DAY = "How to start saving when your income is irregular"

# -----------------------------
# Agents
# -----------------------------
strategist = Agent(
    role="Content Strategist (English)",
    goal="Generate a daily Instagram post angle aligned with niche + objective in English",
    backstory="You plan high-signal, practical IG content with strong hooks.",
    llm=local_llm,
    verbose=True,
    max_iter=4,
)

copywriter = Agent(
    role="Instagram Copywriter (English)",
    goal="Write punchy, helpful captions in English with hook + CTA + hashtag set",
    backstory="You write concise social copy that drives saves and comments.",
    llm=local_llm,
    verbose=True,
    max_iter=5,
)

formatter = Agent(
    role="Editor/Formatter (English)",
    goal="Return a strict valid JSON output only (no extra text)",
    backstory="You enforce formatting rules and JSON validity.",
    llm=local_llm,
    verbose=True,
    max_iter=4,
)

# -----------------------------
# Tasks
# -----------------------------
task_idea = Task(
    description=(
        "IMPORTANT: Respond in ENGLISH.\n\n"
        f"Brand context: {BRAND}\n"
        f"Topic of the day: {TOPIC_OF_THE_DAY}\n\n"
        "Deliver:\n"
        "1) A specific angle (micro-focus)\n"
        "2) 3 value bullets (what the audience learns)\n"
        "3) 1 hook line (first sentence)\n"
        "4) Recommended post type: carousel or reel (one line why)\n"
    ),
    expected_output="Angle + 3 bullets + hook + recommended post type (English).",
    agent=strategist,
)

task_copy = Task(
    description=(
        "IMPORTANT: Respond in ENGLISH.\n\n"
        "Using the Strategist output, write:\n"
        "- 1 primary caption (with short paragraphs, skimmable)\n"
        "- 2 alternate captions (alt_1 shorter, alt_2 more direct)\n"
        "- 1 CTA question at the end\n"
        "- 12 to 20 relevant hashtags (mix: niche + broad + LATAM-friendly)\n"
        "- 1 carousel image prompt: cover + 5 slides (short, clear)\n"
        "Follow constraints and tone."
    ),
    expected_output="Captions + CTA + hashtags + carousel image prompt (English).",
    agent=copywriter,
    context=[task_idea],
)

task_format = Task(
    description=(
        "Return ONLY a valid JSON object. No markdown, no commentary.\n"
        "All fields must be in ENGLISH.\n"
        "Use this exact schema:\n"
        "{\n"
        '  "date": "YYYY-MM-DD",\n'
        '  "topic": "...",\n'
        '  "post_type": "carousel|reel|single",\n'
        '  "angle": "...",\n'
        '  "value_points": ["...", "...", "..."],\n'
        '  "hook": "...",\n'
        '  "caption_primary": "...",\n'
        '  "caption_alt_1": "...",\n'
        '  "caption_alt_2": "...",\n'
        '  "cta": "...",\n'
        '  "hashtags": ["..."],\n'
        '  "image_prompt": "...",\n'
        '  "disclaimer": "..." \n'
        "}\n\n"
        f'Set "date" to today: {date.today().isoformat()}.\n'
        'If no disclaimer is needed, set "disclaimer" to "".\n'
        "Ensure JSON is strictly valid (double quotes, no trailing commas)."
    ),
    expected_output="Strict valid JSON only (English).",
    agent=formatter,
    context=[task_idea, task_copy],
)

crew = Crew(
    agents=[strategist, copywriter, formatter],
    tasks=[task_idea, task_copy, task_format],
    process=Process.sequential,
)

print(crew.kickoff())