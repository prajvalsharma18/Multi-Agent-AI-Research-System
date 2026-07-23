

Topic (CLI)
    ↓
Search Agent ──→ Tavily API
    ↓
Reader Agent ──→ scrape_url (Trafilatura, top 3–5 URLs)
    ↓
Writer ──→ Draft report
    ↓
Critic ──→ Review + score
    ↓
Revision ──→ Final report
    ↓
Export ──→ reports/*.md + reports/*.pdf

Multi_research_ai_system/
├── pipeline.py          # Entry point (CLI)
├── graph.py             # LangGraph workflow
├── agents.py            # Agent + chain definitions
├── tools.py             # web_search, scrape_url
├── source_scoring.py    # URL quality scores
├── cache.py             # Scrape cache
├── export.py            # MD + PDF export
├── reports/             # Generated reports (gitignored)
└── .cache/              # Scrape cache (gitignored)
