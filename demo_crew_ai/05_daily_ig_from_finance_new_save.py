from datetime import date
from pathlib import Path
import json
import re
import feedparser
import requests
from bs4 import BeautifulSoup

from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import tool

# -----------------------------
# Local LLM (Ollama)
# -----------------------------
local_llm = LLM(
    model="ollama/llama3.2",
    base_url="http://localhost:11434",
    temperature=0.3,
)

# -----------------------------
# Config
# -----------------------------
BRAND = {
    "niche": "personal finance",
    "audience": "young professionals",
    "tone": "clear, practical, no hype",
    "objective": "educate + drive comments",
    "language": "English",
}

RSS_FEEDS = [
    # Cambia esto por el feed del sitio que prefieras
    "https://finance.yahoo.com/rss/topstories",
]

KEYWORDS = [
    "inflation", "interest rates", "rates", "fed", "oil", "energy",
    "jobs", "recession", "stocks", "market", "bitcoin", "crypto",
    "earnings", "tariffs", "dollar"
]

OUTPUT_DIR = Path("post")  # carpeta requerida

def clean_html(text: str) -> str:
    if not text:
        return ""
    soup = BeautifulSoup(text, "html.parser")
    return re.sub(r"\s+", " ", soup.get_text(" ")).strip()

def score_item(title: str, summary: str) -> int:
    hay = f"{title} {summary}".lower()
    score = 0
    for kw in KEYWORDS:
        if kw.lower() in hay:
            score += 3
    if "breaking" in hay or "urgent" in hay:
        score += 2
    if len(title) >= 50:
        score += 1
    return score

@tool("Fetch top finance news from RSS")
def fetch_top_finance_news_from_rss() -> dict:
    """
    Fetches items from configured RSS feeds and returns the most relevant item
    based on keyword scoring.
    """
    candidates = []
    for url in RSS_FEEDS:
        feed = feedparser.parse(url)
        for entry in feed.entries[:20]:
            title = (entry.get("title") or "").strip()
            link = (entry.get("link") or "").strip()
            summary = clean_html(entry.get("summary") or entry.get("description") or "")
            s = score_item(title, summary)
            candidates.append({
                "source_feed": url,
                "title": title,
                "link": link,
                "summary": summary[:700],
                "score": s
            })

    if not candidates:
        return {"error": "No RSS items found."}

    candidates.sort(key=lambda x: x["score"], reverse=True)
    top = candidates[0]

    # Optional: fetch some page text (may fail due to blocks/paywalls)
    try:
        r = requests.get(top["link"], timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        if r.ok and "text/html" in (r.headers.get("Content-Type") or ""):
            top["article_text_excerpt"] = clean_html(r.text)[:1200]
    except Exception:
        pass

    return top

# -----------------------------
# Agents
# -----------------------------
news_analyst = Agent(
    role="Finance News Analyst (English)",
    goal="Pick the most relevant finance news and extract an angle for a personal finance IG post",
    backstory="You filter noisy headlines and translate them into everyday money implications.",
    llm=local_llm,
    tools=[fetch_top_finance_news_from_rss],
    verbose=True,
    max_iter=5,
)

ig_creator = Agent(
    role="Instagram Content Creator (English)",
    goal="Create an Instagram-ready topic + hook + caption + hashtags based on the selected news",
    backstory="You write practical, engaging posts—no hype, no promises.",
    llm=local_llm,
    verbose=True,
    max_iter=6,
)

# -----------------------------
# Tasks
# -----------------------------
task_pick_news = Task(
    description=(
        "Use the tool 'Fetch top finance news from RSS' to get the top news item.\n"
        "Then produce:\n"
        "- A 1-sentence summary of the news\n"
        "- Why it matters for personal finance (2 bullets)\n"
        "- A suggested IG topic (1 line)\n"
        "All in ENGLISH.\n\n"
        "Also include the exact fields: news_title, news_link."
    ),
    expected_output="Summary + why it matters + suggested topic + news_title + news_link (English).",
    agent=news_analyst,
)

task_generate_post = Task(
    description=(
        f"Brand context: {BRAND}\n"
        f"Date: {date.today().isoformat()}\n\n"
        "Using the News Analyst output, return ONLY valid JSON with:\n"
        "{\n"
        '  "date": "YYYY-MM-DD",\n'
        '  "news_title": "...",\n'
        '  "news_link": "...",\n'
        '  "topic": "...",\n'
        '  "hook": "...",\n'
        '  "caption": "...",\n'
        '  "cta": "...",\n'
        '  "hashtags": ["..."],\n'
        '  "carousel_outline": ["Cover: ...", "Slide 1: ...", "Slide 2: ...", "Slide 3: ...", "Slide 4: ...", "Slide 5: ..."],\n'
        '  "disclaimer": ""\n'
        "}\n\n"
        "Rules:\n"
        "- English only.\n"
        "- Practical, no hype, no promises.\n"
        "- CTA must be a question.\n"
        "- 12–18 hashtags.\n"
        "- Caption <= ~1500 chars.\n"
        "- Output MUST be JSON only (no markdown)."
    ),
    expected_output="Strict JSON only.",
    agent=ig_creator,
    context=[task_pick_news],
)

crew = Crew(
    agents=[news_analyst, ig_creator],
    tasks=[task_pick_news, task_generate_post],
    process=Process.sequential,
)

# -----------------------------
# Run + Save
# -----------------------------
raw = crew.kickoff()

# Ensure output dir exists
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Parse JSON produced by the LLM
# If the model returns extra whitespace, json.loads still handles it.
data = json.loads(str(raw))

out_path = OUTPUT_DIR / f"{data['date']}.json"
out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

# Optional: also write a human-friendly .md
md_path = OUTPUT_DIR / f"{data['date']}.md"
md = (
    f"# Instagram Post ({data['date']})\n\n"
    f"**News:** {data['news_title']}\n\n"
    f"**Link:** {data['news_link']}\n\n"
    f"## Topic\n{data['topic']}\n\n"
    f"## Hook\n{data['hook']}\n\n"
    f"## Caption\n{data['caption']}\n\n"
    f"## CTA\n{data['cta']}\n\n"
    f"## Hashtags\n" + " ".join(data["hashtags"]) + "\n\n"
    f"## Carousel Outline\n" + "\n".join(f"- {x}" for x in data["carousel_outline"]) + "\n"
)
md_path.write_text(md, encoding="utf-8")

print(f"Saved:\n- {out_path}\n- {md_path}")