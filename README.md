# 💡 Life After PCOS

An AI-powered recommender that helps people diagnosed with PCOS/PCOD discover personalized recovery suggestions — based on real-world journeys of people with similar symptoms, body types, and experiences.

---

## 🎯 Project Goal
To reduce confusion and overwhelm after a PCOS diagnosis by surfacing personalized lifestyle changes that worked for others — using real patient stories and NLP-driven matching.

---

## 🔧 Features (Planned for MVP)
- ✅ Input user profile: symptoms, type, age, activity level
- 🔍 Match with similar recovery journeys
- 🥗 Recommend lifestyle changes (diet, exercise, stress reduction)
- 📈 Visual timeline: "What others experienced over time"
- 🔄 Learn from more stories as the tool grows

---

## 🛠️ Tech Stack
- Python, BeautifulSoup, spaCy, Sentence-BERT
- FAISS / cosine similarity
- Streamlit (web UI)
- Git, GitHub

---

## 📁 Folder Structure
life-after-pcos/
│
├── 📁 data/               ← All your datasets

│   ├── raw/              ← Unprocessed scraped text

│   └── processed/        ← Cleaned, structured data (CSV, JSON)

│
├── 📁 notebooks/         ← Jupyter/Colab notebooks for EDA + modeling

│
├── 📁 src/               ← All core code files

│   ├── scraper.py        ← Web scraper logic (Reddit, forums)

│   ├── cleaner.py        ← NLP preprocessing and structuring

│   ├── matcher.py        ← Similarity model & recommendations

│   └── config.py         ← File paths, settings, constants
│
├── 📁 app/               ← Streamlit frontend

│   └── main.py           ← App entry point
│
├── .gitignore            ← Ignore virtual envs, logs, etc.

├── README.md             ← Project overview (already done!)

├── requirements.txt      ← Python dependencies

└── LICENSE (optional)    ← For public use (MIT recommended)


---

## 🚀 Project Status
✅ Project structure initialized  
🛠️ Phase 1: Web scraping PCOS recovery journeys (Reddit + forums)  

---

## 🙋‍♀️ Author
**Preethi Bommineni**  
_Machine Learning | Data Science | Health-Tech Enthusiast_

---