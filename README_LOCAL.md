# Medical Diagnosis with AI Agents — Local Setup (Free, No API Key)

This version runs **100% locally** using [Ollama](https://ollama.com).  
No OpenAI key, no costs, no internet required after setup.

---

## 1. Install Ollama

Download and install from: https://ollama.com/download

- **Windows**: Download the `.exe` installer
- **Mac**: Download the `.dmg` installer  
- **Linux**: `curl -fsSL https://ollama.com/install.sh | sh`

---

## 2. Pull a free local model

Open a terminal and run:

```bash
ollama pull llama3.2
```

> This downloads ~2GB. Only needed once.  
> Alternative models (change `OLLAMA_MODEL` in `Utils/agents.py`):
> - `ollama pull mistral` — great at following instructions
> - `ollama pull medllama2` — fine-tuned on medical text (slower)
> - `ollama pull llama3.1:8b` — higher quality, needs more RAM

---

## 3. Start Ollama

```bash
ollama serve
```

Keep this terminal open while you use the project.

---

## 4. Set up the Python environment

```bash
# Create a virtual environment
python -m venv venv

# Activate it
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## 5. Run the project

```bash
python main.py
```

The final diagnosis will be printed to the console and saved to `results/final_diagnosis.txt`.

---

## Switching medical reports

Edit line 12 of `main.py` to point to a different report in the `Medical Reports/` folder:

```python
REPORT_PATH = "Medical Reports/Medical Report - Anna Thompson - Irritable Bowel Syndrome.txt"
```

---

## Switching the AI model

Open `Utils/agents.py` and change line 13:

```python
OLLAMA_MODEL = "llama3.2"   # change this to any model you've pulled
```

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `Connection refused` error | Run `ollama serve` in a terminal first |
| `model not found` error | Run `ollama pull llama3.2` |
| Slow responses | Use a smaller model like `llama3.2` or reduce context |
| Out of memory | Close other apps; use `llama3.2` (smallest) |
