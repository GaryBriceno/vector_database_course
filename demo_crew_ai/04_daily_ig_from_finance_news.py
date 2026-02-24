from datetime import date
import re
import feedparser
import requests
from bs4 import BeautifulSoup

from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import tool

# -----------------------------
# LLM local (Ollama)
# -----------------------------
local_llm = LLM(
    model="ollama/llama3.2",
    base_url="http://localhost:11434",
    temperature=0.3,
)

# -----------------------------
# Config (ajusta a tu caso)
# -----------------------------
BRAND = {
    "niche": "personal finance",
    "audience": "young professionals",
    "tone": "clear, practical, no hype",
    "objective": "educate + drive comments",
    "language": "English",
}

# Usa RSS (más estable y “legal-friendly”)
# Ejemplos típicos (pueden cambiar):
# - Yahoo Finance (topic feeds)
# - Investing.com (RSS)
# - MarketWatch (RSS)
# - Reuters: muchos links son paywall / restricciones
RSS_FEEDS = [
    # Reemplaza por el feed que tú quieras usar
    "https://finance.yahoo.com/rss/topstories",
]

KEYWORDS = [
    # Ajusta según tu nicho: energía, macro, crypto, tasas, etc.
    "inflation", "interest rates", "rates", "fed", "oil", "energy",
    "jobs", "recession", "stocks", "market", "bitcoin", "crypto",
    "earnings", "tariffs", "dollar"
]

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
    # Bonus por “breaking/urgent” (si aparece)
    if "breaking" in hay or "urgent" in hay:
        score += 2
    # Bonus por títulos más informativos
    if len(title) >= 50:
        score += 1
    return score

@tool("Fetch top finance news from RSS")
def fetch_top_finance_news_from_rss() -> dict:
    """
    Fetches items from configured RSS feeds and returns the most relevant item
    based on simple keyword scoring.
    """
    candidates = []
    for url in RSS_FEEDS:
        feed = feedparser.parse(url)
        for entry in feed.entries[:20]:
            title = entry.get("title", "").strip()
            link = entry.get("link", "").strip()
            summary = clean_html(entry.get("summary", "") or entry.get("description", ""))
            s = score_item(title, summary)
            candidates.append({
                "source_feed": url,
                "title": title,
                "link": link,
                "summary": summary[:600],
                "score": s
            })

    if not candidates:
        return {"error": "No RSS items found."}

    candidates.sort(key=lambda x: x["score"], reverse=True)
    top = candidates[0]

    # (Opcional) Intentar traer el HTML del link para más contexto.
    # OJO: algunos sitios bloquean scraping. RSS suele bastar para un post.
    try:
        r = requests.get(top["link"], timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        if r.ok and "text/html" in r.headers.get("Content-Type", ""):
            page_text = clean_html(r.text)
            top["article_text_excerpt"] = page_text[:1200]
    except Exception:
        pass

    return top

# -----------------------------
# Agents
# -----------------------------
news_analyst = Agent(
    role="Finance News Analyst (English)",
    goal="Pick the most relevant finance news item and extract the key angle for an IG post",
    backstory="You filter noisy news and identify what matters for personal finance audiences.",
    llm=local_llm,
    tools=[fetch_top_finance_news_from_rss],
    verbose=True,
    max_iter=5,
)

ig_creator = Agent(
    role="Instagram Content Creator (English)",
    goal="Generate an Instagram-ready topic + hook + caption + hashtags based on the selected news",
    backstory="You turn real news into practical, engaging content without hype.",
    llm=local_llm,
    verbose=True,
    max_iter=5,
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
        "All in ENGLISH."
    ),
    expected_output="News summary + why it matters + suggested IG topic (English).",
    agent=news_analyst,
)

task_generate_post = Task(
    description=(
        f"Brand context: {BRAND}\n"
        f"Date: {date.today().isoformat()}\n\n"
        "Using the News Analyst output, create a single Instagram post package in ENGLISH:\n"
        "Return ONLY valid JSON with:\n"
        '{\n'
        '  "date": "YYYY-MM-DD",\n'
        '  "news_title": "...",\n'
        '  "news_link": "...",\n'
        '  "topic": "...",\n'
        '  "hook": "...",\n'
        '  "caption": "...",\n'
        '  "cta": "...",\n'
        '  "hashtags": ["..."],\n'
        '  "carousel_outline": ["Cover: ...", "Slide 1: ...", "Slide 2: ...", "Slide 3: ...", "Slide 4: ...", "Slide 5: ..."]\n'
        '}\n'
        "Rules:\n"
        "- Practical, no hype, no promises.\n"
        "- CTA must be a question.\n"
        "- 12–18 hashtags.\n"
        "- Caption <= ~1500 chars.\n"
        "- Do not add text outside JSON."
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

print(crew.kickoff())