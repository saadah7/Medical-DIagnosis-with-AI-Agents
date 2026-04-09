# MediAgent — AI-Powered Medical Diagnosis System

> A multi-agent clinical decision support system that simulates a Multidisciplinary Medical Team (MDT) using locally-running Large Language Models. Built as a Final Year Project in Computer Science & Engineering.

⚠️ **Disclaimer:** This project is for research and educational purposes only. It is not intended for clinical use and must not be used to make real medical decisions.

---

## What It Does

MediAgent takes a patient's medical report or manually entered symptoms and routes them through a panel of 8 AI specialist agents — each scoped strictly to their medical domain. All agents run in parallel. Each one produces a structured differential diagnosis: listing possible conditions with supporting evidence, opposing evidence, and a likelihood ranking. A lead MDT agent then synthesizes all specialist findings into a single concise clinical summary with a risk level.

The entire system runs locally on your machine using [Ollama](https://ollama.com) and Meta's LLaMA 3.2 model. No API keys. No cloud. No cost.

---

## Features

- **8 specialist AI agents** — Cardiologist, Psychologist, Pulmonologist, Neurologist, Endocrinologist, Gastroenterologist, Dermatologist, Hematologist
- **Smart routing** — automatically selects only the relevant specialists based on keywords in the patient report
- **Differential diagnosis reasoning** — each agent lists conditions with FOR and AGAINST evidence, not just a guess
- **Confidence scoring** — each specialist rates their own certainty (High / Medium / Low) with a one-sentence justification
- **MDT synthesis** — a lead agent consolidates all findings into a ranked Top 3 Diagnoses with overall risk level
- **Parallel execution** — all agents run simultaneously via Python's `ThreadPoolExecutor`
- **Dual input modes** — upload a `.txt` report file or enter symptoms manually via a form
- **Web interface** — professional Streamlit dashboard with hospital-grade clinical UI
- **Session history** — every diagnosis saved to `logs/` as JSON and `results/` as `.txt`
- **100% local** — powered by Ollama, runs on CPU or GPU with no internet after setup

---

## Project Structure

```
MediAgent/
│
├── app.py                          # Streamlit web interface (run this)
├── main.py                         # CLI version
│
├── Utils/
│   └── agents.py                   # All agent classes, prompts, routing logic
│
├── Medical Reports/                # 10 pre-loaded synthetic patient cases
│   ├── Medical Report - Anna Thompson - Irritable Bowel Syndrome.txt
│   ├── Medical Report - Charles Baker - Prostate Cancer (Suspicion).txt
│   ├── Medical Report - David Wilson - Alzheimer's Disease.txt
│   ├── Medical Report - James Carter - Insomnia.txt
│   ├── Medical Report - Kevin Adams - Diabetic Neuropathy.txt
│   ├── Medical Report - Laura Garcia - Rheumatoid Arthritis.txt
│   ├── Medical Report - Maria Silva - Polycystic Ovary Syndrome.txt
│   ├── Medical Report - Olivia White - Recurrent Tonsillitis.txt
│   ├── Medical Report - Robert Miller - COPD.txt
│   └── Medical Report - Michael Johnson - Panic Attack Disorder.txt
│
├── results/                        # Auto-generated .txt diagnosis reports
├── logs/                           # Auto-generated .json session history
├── requirements.txt
├── README.md
└── README_LOCAL.md
```

---

## Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.11+ |
| LLM Runtime | [Ollama](https://ollama.com) |
| LLM Model | LLaMA 3.2 (Meta, open-source) |
| Agent Framework | LangChain (`langchain-ollama`) |
| Web UI | Streamlit |
| Concurrency | `concurrent.futures.ThreadPoolExecutor` |
| Session Storage | JSON (local `logs/` folder) |

---

## Setup & Installation

### Prerequisites

- Python 3.10 or higher
- [Ollama](https://ollama.com/download) installed on your machine
- ~2.5 GB disk space for the LLaMA 3.2 model

---

### Step 1 — Install Ollama

| Platform | Instructions |
|---|---|
| Windows | Download the `.exe` from [ollama.com/download](https://ollama.com/download) |
| macOS | Download the `.dmg` from [ollama.com/download](https://ollama.com/download) |
| Linux | `curl -fsSL https://ollama.com/install.sh \| sh` |

---

### Step 2 — Pull the LLM model

Open a terminal and run:

```bash
ollama pull llama3.2
```

This downloads ~2 GB and only needs to be done once.

**Alternative models** (change `OLLAMA_MODEL` in `Utils/agents.py`):

| Model | Command | Notes |
|---|---|---|
| LLaMA 3.2 (default) | `ollama pull llama3.2` | Fast, 3B params, recommended |
| Mistral 7B | `ollama pull mistral` | Strong instruction following |
| LLaMA 3.1 8B | `ollama pull llama3.1:8b` | Higher quality, needs more RAM |
| MedLLaMA 2 | `ollama pull medllama2` | Fine-tuned on medical text, slower |

---

### Step 3 — Start Ollama

```bash
ollama serve
```

Keep this terminal open while using the project. On Windows, Ollama may already be running in the system tray after installation — if you see a connection error when running `ollama serve`, it means it's already active.

**To make Ollama start automatically on Windows boot:**
1. Press `Win + R`, type `shell:startup`, press Enter
2. Create a file called `start_ollama.bat` with the content: `ollama serve`
3. Save it — Ollama will now start with Windows automatically

---

### Step 4 — Set up Python environment

```bash
# Clone the repository
git clone https://github.com/saadah7/Medical-DIagnosis-with-AI-Agents.git
cd Medical-DIagnosis-with-AI-Agents

# Create and activate virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

### Step 5 — Run the web interface

```bash
streamlit run app.py
```

The app opens automatically at `http://localhost:8501` in your browser.

**Or run the CLI version:**

```bash
python main.py
```

---

## How to Use

### Web Interface (`app.py`)

1. Open the app in your browser at `http://localhost:8501`
2. Choose an input mode:
   - **Report File** — upload a `.txt` file or select one of the 10 pre-loaded cases
   - **Manual Entry** — fill in patient name, age, gender, symptoms, medical history, medications
3. Click **Run Diagnostic Analysis**
4. Wait 1–3 minutes while agents process in parallel
5. Review the structured diagnosis report:
   - Patient Summary
   - Top 3 Diagnoses with evidence FOR and AGAINST each
   - Key Findings
   - Recommended Next Steps
   - Overall Risk Level (Low / Moderate / High)
   - Expandable specialist reports with full differential reasoning
6. Download the full report as a `.txt` file
7. View past sessions in the **History** tab

### CLI (`main.py`)

```bash
python main.py
```

Follow the prompts to select a report file or enter symptoms manually. The final diagnosis prints to console and saves to `results/`.

---

## How Agents Work

### Specialist Selection

Before any LLM call is made, `select_specialists()` reads the patient report and matches it against a keyword dictionary for each specialist. Only matching specialists are activated. At minimum, Cardiologist, Psychologist, and Pulmonologist are always included.

```python
# Example: a report mentioning "anxiety", "chest pain", "dizziness" activates:
# → Cardiologist (chest pain, palpitation)
# → Psychologist (anxiety, panic)
# → Pulmonologist (breath, dizziness)
# → Neurologist (dizziness, vertigo)
```

### Differential Diagnosis Structure

Each specialist returns a report structured in 4 plain-text sections:

```
FINDINGS
Relevant symptoms and test results from the report.

DIFFERENTIAL DIAGNOSIS
Condition: [name]
For: [supporting evidence from the report]
Against: [contradicting evidence or missing data]
Likelihood: Most Likely / Possible / Less Likely

PRIMARY DIAGNOSIS
Most likely condition in one sentence. Justification in one sentence.

NEXT STEPS
1. Most urgent investigation or treatment
2. ...
3. ...
```

### Confidence Scoring

Each specialist also returns:
- `CONFIDENCE: High / Medium / Low`
- `CONFIDENCE_REASON: one sentence referencing specific findings`

Rules: High = 3+ direct indicators found. Medium = 1–2 indirect signs. Low = precautionary inclusion only.

### MDT Synthesis

After all specialists complete, `MultidisciplinaryTeam` receives all reports and produces:
- Patient Summary (2 sentences)
- Top 3 Diagnoses ranked across all specialists (with evidence FOR / AGAINST per diagnosis)
- 4–5 Key Findings
- 4 Recommended Next Steps
- Overall Risk Level with justification

---

## Configuration

### Change the AI model

Open `Utils/agents.py` and edit line 6:

```python
OLLAMA_MODEL = "llama3.2"   # replace with any pulled model
```

### Change the default CLI report

Open `main.py` and edit the `REPORT_PATH` variable.

### Add a new specialist agent

1. Add keywords to `SPECIALIST_KEYWORDS` in `agents.py`
2. Add a prompt to `SPECIALIST_PROMPTS`
3. Create a class inheriting from `Agent`
4. Add to `SPECIALIST_REGISTRY`

---

## Troubleshooting

| Error | Cause | Fix |
|---|---|---|
| `[WinError 10061] Connection refused` | Ollama is not running | Run `ollama serve` in a terminal |
| `[WinError 10048] Address in use` | Ollama is already running | Skip `ollama serve` — it's active |
| `model not found` | Model not pulled yet | Run `ollama pull llama3.2` |
| `UnicodeEncodeError cp1252` | Old version of the code | Make sure you have the latest `main.py` with `encoding="utf-8"` |
| Responses contain `**bold**` text | Old version of `agents.py` | Update to latest `agents.py` with `NO_MARKDOWN` instruction |
| Very slow responses | Model too large for your RAM | Switch to `llama3.2` (3B, fastest) |
| Streamlit not found | Not installed | Run `pip install streamlit` |

---

## Future Enhancements

- **RAG (Retrieval Augmented Generation)** — connect agents to PubMed and clinical guidelines so they retrieve real research before diagnosing
- **Medical image analysis** — integrate vision models (LLaVA) to process X-rays, ECG images, and dermatology photos
- **Voice input** — speech-to-text so patients can describe symptoms verbally
- **Cloud deployment** — Groq free API + Streamlit Cloud for browser access from any device
- **EHR integration** — connect to Electronic Health Record systems via HL7/FHIR standard
- **Multilingual support** — Urdu, Hindi, Arabic for regional healthcare contexts
- **PDF report export** — formatted downloadable clinical PDF with branding and confidence charts

---

## Academic Context

This project was developed as a Final Year Project for a Bachelor of Engineering in Computer Science. It demonstrates the following concepts:

- Multi-agent system design and orchestration
- Large Language Model prompt engineering
- Concurrent programming with thread pools
- Natural Language Processing for clinical text
- Web application development with Streamlit
- Software architecture: separation of concerns (agents / UI / CLI)

---

## License

This project is for educational and research purposes. Not licensed for commercial or clinical use.

---

*Built with LangChain · Ollama · LLaMA 3.2 · Streamlit · Python*