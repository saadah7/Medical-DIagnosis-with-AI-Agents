from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
import re
import requests

OLLAMA_BASE_URL = "http://localhost:11434"


def check_ollama():
    """Raise RuntimeError with a clear message if Ollama is not reachable."""
    try:
        requests.get(OLLAMA_BASE_URL, timeout=3)
    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            "Cannot connect to Ollama at http://localhost:11434. "
            "Make sure Ollama is running (`ollama serve`) before starting MediAgent."
        )

OLLAMA_MODEL = "llama3.2"

SPECIALIST_KEYWORDS = {
    "Cardiologist":       ["heart", "cardiac", "ecg", "ekg", "chest pain", "palpitation",
                           "arrhythmia", "blood pressure", "hypertension", "holter", "echocardiogram"],
    "Psychologist":       ["anxiety", "depression", "panic", "stress", "mental", "mood",
                           "trauma", "ptsd", "insomnia", "psychological", "psychiatric", "phobia"],
    "Pulmonologist":      ["lung", "breath", "respiratory", "asthma", "copd", "oxygen",
                           "pulmonary", "spirometry", "cough", "wheeze", "inhaler"],
    "Neurologist":        ["headache", "migraine", "seizure", "neuro", "brain", "memory",
                           "alzheimer", "tremor", "dizziness", "vertigo", "nerve", "neuropathy",
                           "stroke", "ms", "multiple sclerosis", "parkinson"],
    "Endocrinologist":    ["diabetes", "thyroid", "insulin", "glucose", "hormone", "hba1c",
                           "pcos", "polycystic", "adrenal", "cortisol", "endocrine", "obesity",
                           "metabolic", "weight gain", "weight loss"],
    "Gastroenterologist": ["stomach", "bowel", "ibs", "crohn", "colitis", "gastro", "liver",
                           "hepatic", "nausea", "vomit", "diarrhea", "constipation", "acid reflux",
                           "gerd", "ulcer", "colon", "intestine", "abdominal"],
    "Dermatologist":      ["skin", "rash", "eczema", "psoriasis", "acne", "lesion", "itching",
                           "dermat", "hair loss", "alopecia", "nail", "wound", "sore"],
    "Hematologist":       ["blood", "anemia", "hemoglobin", "platelet", "wbc", "rbc", "leukemia",
                           "lymphoma", "clot", "coagulation", "iron", "ferritin", "hematology",
                           "bone marrow", "bleeding"],
}

# ─────────────────────────────────────────────────────────────────────────────
# CRITICAL: All prompts explicitly forbid markdown formatting.
# Plain text only — no asterisks, no bold, no headers with #.
# The UI handles all visual formatting.
# ─────────────────────────────────────────────────────────────────────────────

NO_MARKDOWN = (
    "\n\nCRITICAL FORMATTING RULES — strictly follow these:\n"
    "- Do NOT use asterisks (*) or double asterisks (**) anywhere.\n"
    "- Do NOT use markdown bold, italic, or headers.\n"
    "- Do NOT use # headings.\n"
    "- Write in plain text only.\n"
    "- Use the section labels exactly as specified, nothing else.\n"
)

CONFIDENCE_INSTRUCTION = (
    "\n\nEnd your response with exactly these two lines:\n"
    "CONFIDENCE: High, Medium, or Low\n"
    "CONFIDENCE_REASON: one plain sentence referencing specific findings\n"
    "Rules: High = 3+ direct specialty indicators. Medium = 1-2 indirect signs. Low = precautionary only.\n"
)

DIFFERENTIAL_INSTRUCTION = (
    "\n\nStructure your entire response in exactly these 4 plain-text sections:\n\n"
    "FINDINGS\n"
    "List only the relevant symptoms, signs, and test results from the report. One item per line. No bullets, no asterisks.\n\n"
    "DIFFERENTIAL DIAGNOSIS\n"
    "List up to 3 conditions, most likely first. For each write exactly:\n"
    "Condition: [name]\n"
    "For: [specific supporting evidence from the report]\n"
    "Against: [specific contradicting evidence or missing data]\n"
    "Likelihood: Most Likely, Possible, or Less Likely\n\n"
    "PRIMARY DIAGNOSIS\n"
    "State the most likely condition in one sentence. Then explain in one sentence why it is favoured over alternatives.\n\n"
    "NEXT STEPS\n"
    "List 3 specific investigations or treatments, one per line, numbered, no asterisks.\n"
)

SPECIALIST_PROMPTS = {
    "Cardiologist": (
        "You are an experienced Cardiologist. Scope: cardiovascular system only.\n"
        "Review the report below. Be concise and clinical.\n\n"
        "Patient Report:\n{medical_report}"
        + DIFFERENTIAL_INSTRUCTION + NO_MARKDOWN + CONFIDENCE_INSTRUCTION
    ),
    "Psychologist": (
        "You are an experienced Psychologist. Scope: mental health and psychological well-being only.\n"
        "Review the report below. Be concise and clinical.\n\n"
        "Patient Report:\n{medical_report}"
        + DIFFERENTIAL_INSTRUCTION + NO_MARKDOWN + CONFIDENCE_INSTRUCTION
    ),
    "Pulmonologist": (
        "You are an experienced Pulmonologist. Scope: respiratory system only.\n"
        "Review the report below. Be concise and clinical.\n\n"
        "Patient Report:\n{medical_report}"
        + DIFFERENTIAL_INSTRUCTION + NO_MARKDOWN + CONFIDENCE_INSTRUCTION
    ),
    "Neurologist": (
        "You are an experienced Neurologist. Scope: nervous system only.\n"
        "Review the report below. Be concise and clinical.\n\n"
        "Patient Report:\n{medical_report}"
        + DIFFERENTIAL_INSTRUCTION + NO_MARKDOWN + CONFIDENCE_INSTRUCTION
    ),
    "Endocrinologist": (
        "You are an experienced Endocrinologist. Scope: hormonal and metabolic systems only.\n"
        "Review the report below. Be concise and clinical.\n\n"
        "Patient Report:\n{medical_report}"
        + DIFFERENTIAL_INSTRUCTION + NO_MARKDOWN + CONFIDENCE_INSTRUCTION
    ),
    "Gastroenterologist": (
        "You are an experienced Gastroenterologist. Scope: digestive system only.\n"
        "Review the report below. Be concise and clinical.\n\n"
        "Patient Report:\n{medical_report}"
        + DIFFERENTIAL_INSTRUCTION + NO_MARKDOWN + CONFIDENCE_INSTRUCTION
    ),
    "Dermatologist": (
        "You are an experienced Dermatologist. Scope: skin, hair, nails only.\n"
        "Review the report below. Be concise and clinical.\n\n"
        "Patient Report:\n{medical_report}"
        + DIFFERENTIAL_INSTRUCTION + NO_MARKDOWN + CONFIDENCE_INSTRUCTION
    ),
    "Hematologist": (
        "You are an experienced Hematologist. Scope: blood and blood disorders only.\n"
        "Review the report below. Be concise and clinical.\n\n"
        "Patient Report:\n{medical_report}"
        + DIFFERENTIAL_INSTRUCTION + NO_MARKDOWN + CONFIDENCE_INSTRUCTION
    ),
}


def strip_markdown(text: str) -> str:
    """Remove all markdown formatting from model output."""
    # Remove bold/italic asterisks
    text = re.sub(r'\*{1,3}(.*?)\*{1,3}', r'\1', text)
    # Remove markdown headers
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
    # Remove horizontal rules
    text = re.sub(r'^[-_*]{3,}\s*$', '', text, flags=re.MULTILINE)
    # Remove leading dashes used as bullets (keep numbered lists)
    text = re.sub(r'^\s*[-•]\s+', '', text, flags=re.MULTILINE)
    # Collapse 3+ blank lines to 2
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def parse_confidence(raw_response: str) -> tuple:
    confidence = "Medium"
    reason     = "Based on available indicators in the report."

    match = re.search(r"CONFIDENCE:\s*(High|Medium|Low)", raw_response, re.IGNORECASE)
    if match:
        confidence = match.group(1).capitalize()

    reason_match = re.search(r"CONFIDENCE_REASON:\s*(.+?)(?:\n|$)", raw_response, re.IGNORECASE)
    if reason_match:
        reason = reason_match.group(1).strip()

    # Strip confidence block only; markdown cleanup happens at render time
    clean = re.sub(r"\nCONFIDENCE:.*", "", raw_response, flags=re.IGNORECASE | re.DOTALL).strip()
    return clean, confidence, reason


def select_specialists(medical_report: str) -> list:
    report_lower = medical_report.lower()
    selected = []
    for specialist, keywords in SPECIALIST_KEYWORDS.items():
        if any(kw in report_lower for kw in keywords):
            selected.append(specialist)
    base = ["Cardiologist", "Psychologist", "Pulmonologist"]
    for b in base:
        if b not in selected:
            selected.append(b)
    return selected


class Agent:
    def __init__(self, medical_report=None, role=None, extra_info=None):
        self.medical_report  = medical_report
        self.role            = role
        self.extra_info      = extra_info
        self.prompt_template = self.create_prompt_template()
        self.model = ChatOllama(
            model=OLLAMA_MODEL,
            temperature=0,
            base_url=OLLAMA_BASE_URL,
        )

    def create_prompt_template(self):
        if self.role == "MultidisciplinaryTeam":
            specialist_section = ""
            for specialist, data in self.extra_info.get("specialist_reports", {}).items():
                conf   = data.get("confidence", "Medium")
                reason = data.get("confidence_reason", "")
                report = data.get("report", "")
                specialist_section += (
                    f"[{specialist}] Confidence: {conf} — {reason}\n"
                    f"{report}\n\n"
                )

            template = (
                "You are the lead physician at an MDT meeting. Specialists have reviewed the patient.\n\n"
                + specialist_section +
                "Write a unified patient summary in plain text using EXACTLY this structure.\n"
                "No asterisks. No bold. No markdown. No extra commentary.\n\n"
                "PATIENT SUMMARY\n"
                "Write 2 sentences maximum describing the patient's overall condition in plain clinical language.\n\n"
                "TOP 3 DIAGNOSES\n"
                "For each diagnosis write exactly:\n"
                "1. [Condition name] — [Most Likely / Possible / Less Likely]\n"
                "   Flagged by: [specialist names, comma separated]\n"
                "   Evidence for: [2-3 key findings, plain text]\n"
                "   Evidence against: [1-2 findings or gaps]\n\n"
                "2. [Same format]\n\n"
                "3. [Same format]\n\n"
                "KEY FINDINGS\n"
                "List 4 findings only. One per line. No bullets. No asterisks. Just the text.\n\n"
                "NEXT STEPS\n"
                "List 4 actions only. Numbered. One per line. No asterisks.\n\n"
                "OVERALL RISK LEVEL\n"
                "Write: Low, Moderate, or High — then one sentence justification.\n\n"
                "Hard limit: 350 words total. No preamble. No sign-off."
            )
            return PromptTemplate.from_template(template)

        template = SPECIALIST_PROMPTS[self.role]
        return PromptTemplate.from_template(template)

    def run(self):
        print(f"  🔬 {self.role} is analyzing...")
        prompt = self.prompt_template.format(medical_report=self.medical_report or "")
        try:
            response = self.model.invoke(prompt)
            raw = response.content

            if self.role != "MultidisciplinaryTeam":
                clean_report, confidence, confidence_reason = parse_confidence(raw)
                return {
                    "report":            clean_report,
                    "confidence":        confidence,
                    "confidence_reason": confidence_reason,
                }
            return strip_markdown(raw)

        except Exception as e:
            print(f"  Error in {self.role}: {e}")
            if self.role != "MultidisciplinaryTeam":
                return {
                    "report":            f"{self.role} analysis unavailable: {e}",
                    "confidence":        "Low",
                    "confidence_reason": "Error during analysis.",
                    "failed":            True,
                }
            return f"MDT analysis unavailable: {e}"


def _make_specialist(role: str):
    return lambda medical_report: Agent(medical_report, role)


SPECIALIST_REGISTRY = {role: _make_specialist(role) for role in SPECIALIST_PROMPTS}


class MultidisciplinaryTeam(Agent):
    def __init__(self, specialist_reports: dict):
        super().__init__(role="MultidisciplinaryTeam",
                         extra_info={"specialist_reports": specialist_reports})