from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
import re

# ─────────────────────────────────────────────────────────────────────────────
# Change OLLAMA_MODEL to switch models. Make sure you've pulled it first.
# Recommended: "llama3.2" (fast), "llama3.1:8b" (smarter), "mistral" (good balance)
# ─────────────────────────────────────────────────────────────────────────────
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
# Confidence block appended to every specialist prompt.
# ─────────────────────────────────────────────────────────────────────────────
CONFIDENCE_INSTRUCTION = (
    "\n\nAt the very end of your response, you MUST add this block exactly:\n"
    "CONFIDENCE: <High|Medium|Low>\n"
    "CONFIDENCE_REASON: <one sentence explaining why, referencing specific findings>\n"
    "Rules for confidence level:\n"
    "  High   = 3+ direct indicators from your specialty found in the report\n"
    "  Medium = 1-2 indicators, or findings are indirect/suggestive\n"
    "  Low    = no direct indicators, included as precaution only\n"
)

# ─────────────────────────────────────────────────────────────────────────────
# Differential diagnosis reasoning instruction — appended to every specialist.
# This is the core of Step 4: agents now reason like real clinicians by
# explicitly considering, ranking, and ruling out competing diagnoses.
# ─────────────────────────────────────────────────────────────────────────────
DIFFERENTIAL_INSTRUCTION = (
    "\n\nYou MUST structure your response in EXACTLY these 4 sections:\n\n"
    "FINDINGS\n"
    "List the key symptoms, signs, and test results from this report that are relevant to your specialty.\n\n"
    "DIFFERENTIAL DIAGNOSIS\n"
    "List 2-3 possible conditions from most to least likely. For each one:\n"
    "  - Condition name\n"
    "  - SUPPORTING EVIDENCE: specific findings from the report that point to this condition\n"
    "  - AGAINST: specific findings or absences that argue against this condition\n"
    "  - LIKELIHOOD: Most Likely / Possible / Less Likely\n\n"
    "PRIMARY DIAGNOSIS\n"
    "State the single most likely condition and explain in 2 sentences why the evidence "
    "favours it over the alternatives.\n\n"
    "NEXT STEPS\n"
    "List 3-4 specific, prioritised investigations or treatments to confirm or manage the primary diagnosis.\n"
)

SPECIALIST_PROMPTS = {
    "Cardiologist": (
        "You are an experienced Cardiologist reviewing a patient's medical report.\n"
        "YOUR SCOPE: Cardiovascular system only — heart, blood vessels, circulation, blood pressure.\n"
        "Be specific and clinical. Do not comment outside your specialty.\n\n"
        "Patient Report:\n{medical_report}"
        + DIFFERENTIAL_INSTRUCTION
        + CONFIDENCE_INSTRUCTION
    ),
    "Psychologist": (
        "You are an experienced Psychologist/Psychiatrist reviewing a patient's medical report.\n"
        "YOUR SCOPE: Mental health, psychological well-being, behavioral patterns, emotional state.\n"
        "Be specific and clinical. Do not comment outside your specialty.\n\n"
        "Patient Report:\n{medical_report}"
        + DIFFERENTIAL_INSTRUCTION
        + CONFIDENCE_INSTRUCTION
    ),
    "Pulmonologist": (
        "You are an experienced Pulmonologist reviewing a patient's medical report.\n"
        "YOUR SCOPE: Respiratory system — lungs, airways, breathing, oxygen levels.\n"
        "Be specific and clinical. Do not comment outside your specialty.\n\n"
        "Patient Report:\n{medical_report}"
        + DIFFERENTIAL_INSTRUCTION
        + CONFIDENCE_INSTRUCTION
    ),
    "Neurologist": (
        "You are an experienced Neurologist reviewing a patient's medical report.\n"
        "YOUR SCOPE: Nervous system — brain, spinal cord, peripheral nerves, cognition, movement.\n"
        "Be specific and clinical. Do not comment outside your specialty.\n\n"
        "Patient Report:\n{medical_report}"
        + DIFFERENTIAL_INSTRUCTION
        + CONFIDENCE_INSTRUCTION
    ),
    "Endocrinologist": (
        "You are an experienced Endocrinologist reviewing a patient's medical report.\n"
        "YOUR SCOPE: Hormonal and metabolic systems — diabetes, thyroid, adrenal, pituitary, reproductive hormones.\n"
        "Be specific and clinical. Do not comment outside your specialty.\n\n"
        "Patient Report:\n{medical_report}"
        + DIFFERENTIAL_INSTRUCTION
        + CONFIDENCE_INSTRUCTION
    ),
    "Gastroenterologist": (
        "You are an experienced Gastroenterologist reviewing a patient's medical report.\n"
        "YOUR SCOPE: Digestive system — stomach, intestines, liver, pancreas, bowel.\n"
        "Be specific and clinical. Do not comment outside your specialty.\n\n"
        "Patient Report:\n{medical_report}"
        + DIFFERENTIAL_INSTRUCTION
        + CONFIDENCE_INSTRUCTION
    ),
    "Dermatologist": (
        "You are an experienced Dermatologist reviewing a patient's medical report.\n"
        "YOUR SCOPE: Skin, hair, nails — conditions, lesions, rashes, infections.\n"
        "Be specific and clinical. Do not comment outside your specialty.\n\n"
        "Patient Report:\n{medical_report}"
        + DIFFERENTIAL_INSTRUCTION
        + CONFIDENCE_INSTRUCTION
    ),
    "Hematologist": (
        "You are an experienced Hematologist reviewing a patient's medical report.\n"
        "YOUR SCOPE: Blood and blood disorders — CBC, anemia, clotting, blood cancers, iron levels.\n"
        "Be specific and clinical. Do not comment outside your specialty.\n\n"
        "Patient Report:\n{medical_report}"
        + DIFFERENTIAL_INSTRUCTION
        + CONFIDENCE_INSTRUCTION
    ),
}


# ─────────────────────────────────────────────────────────────────────────────
# Parse confidence score and reason out of a specialist response.
# Returns: (clean_report, confidence_level, confidence_reason)
# ─────────────────────────────────────────────────────────────────────────────
def parse_confidence(raw_response: str) -> tuple:
    confidence = "Medium"
    reason     = "Based on available indicators in the report."

    match = re.search(r"CONFIDENCE:\s*(High|Medium|Low)", raw_response, re.IGNORECASE)
    if match:
        confidence = match.group(1).capitalize()

    reason_match = re.search(r"CONFIDENCE_REASON:\s*(.+?)(?:\n|$)", raw_response, re.IGNORECASE)
    if reason_match:
        reason = reason_match.group(1).strip()

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
            base_url="http://localhost:11434",
        )

    def create_prompt_template(self):
        if self.role == "MultidisciplinaryTeam":
            specialist_section = ""
            for specialist, data in self.extra_info.get("specialist_reports", {}).items():
                conf   = data.get("confidence", "Medium")
                reason = data.get("confidence_reason", "")
                report = data.get("report", "")
                specialist_section += (
                    f"[{specialist}] (Confidence: {conf} — {reason})\n"
                    f"{report}\n\n"
                )

            template = (
                "You are the lead physician chairing a Multidisciplinary Team (MDT) meeting.\n"
                "Each specialist has provided their findings, a differential diagnosis, "
                "and a primary diagnosis with supporting and opposing evidence.\n\n"
                + specialist_section +
                "YOUR TASK: Synthesize all specialist differentials into one unified patient report "
                "using EXACTLY this format:\n\n"
                "PATIENT SUMMARY\n"
                "---------------\n"
                "2-3 sentences describing the patient's overall condition in plain language.\n\n"
                "TOP 3 DIAGNOSES\n"
                "---------------\n"
                "Rank the top 3 diagnoses across ALL specialists by overall likelihood. For each:\n"
                "1. [Condition] — [Likelihood: Most Likely/Possible/Less Likely]\n"
                "   Flagged by: [specialist names]\n"
                "   Key evidence FOR: [2-3 specific findings]\n"
                "   Key evidence AGAINST: [1-2 specific findings or gaps]\n\n"
                "2. [Condition] — [Likelihood]\n"
                "   Flagged by: ...\n"
                "   Key evidence FOR: ...\n"
                "   Key evidence AGAINST: ...\n\n"
                "3. [Condition] — [Likelihood]\n"
                "   Flagged by: ...\n"
                "   Key evidence FOR: ...\n"
                "   Key evidence AGAINST: ...\n\n"
                "KEY FINDINGS\n"
                "---------------\n"
                "- [finding 1]\n"
                "- [finding 2]\n"
                "- [finding 3]\n"
                "- [finding 4]\n"
                "- [finding 5]\n\n"
                "RECOMMENDED NEXT STEPS\n"
                "---------------\n"
                "1. [Most urgent action]\n"
                "2. [Second action]\n"
                "3. [Third action]\n"
                "4. [Fourth action]\n\n"
                "OVERALL RISK LEVEL\n"
                "---------------\n"
                "Low / Moderate / High — one sentence justification.\n\n"
                "Be direct and clinical. No preamble. No repetition. Keep under 500 words."
            )
            return PromptTemplate.from_template(template)

        template = SPECIALIST_PROMPTS[self.role]
        return PromptTemplate.from_template(template)

    def run(self):
        print(f"  🔬 {self.role} is analyzing the report...")
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
            return raw

        except Exception as e:
            print(f"  ❌ Error in {self.role}: {e}")
            print("  Tip: Make sure Ollama is running — run: ollama serve")
            if self.role != "MultidisciplinaryTeam":
                return {
                    "report":            f"[{self.role} analysis unavailable: {e}]",
                    "confidence":        "Low",
                    "confidence_reason": "Error occurred during analysis.",
                }
            return f"[MDT analysis unavailable: {e}]"


class Cardiologist(Agent):
    def __init__(self, medical_report): super().__init__(medical_report, "Cardiologist")

class Psychologist(Agent):
    def __init__(self, medical_report): super().__init__(medical_report, "Psychologist")

class Pulmonologist(Agent):
    def __init__(self, medical_report): super().__init__(medical_report, "Pulmonologist")

class Neurologist(Agent):
    def __init__(self, medical_report): super().__init__(medical_report, "Neurologist")

class Endocrinologist(Agent):
    def __init__(self, medical_report): super().__init__(medical_report, "Endocrinologist")

class Gastroenterologist(Agent):
    def __init__(self, medical_report): super().__init__(medical_report, "Gastroenterologist")

class Dermatologist(Agent):
    def __init__(self, medical_report): super().__init__(medical_report, "Dermatologist")

class Hematologist(Agent):
    def __init__(self, medical_report): super().__init__(medical_report, "Hematologist")


SPECIALIST_REGISTRY = {
    "Cardiologist":       Cardiologist,
    "Psychologist":       Psychologist,
    "Pulmonologist":      Pulmonologist,
    "Neurologist":        Neurologist,
    "Endocrinologist":    Endocrinologist,
    "Gastroenterologist": Gastroenterologist,
    "Dermatologist":      Dermatologist,
    "Hematologist":       Hematologist,
}


class MultidisciplinaryTeam(Agent):
    def __init__(self, specialist_reports: dict):
        extra_info = {"specialist_reports": specialist_reports}
        super().__init__(role="MultidisciplinaryTeam", extra_info=extra_info)