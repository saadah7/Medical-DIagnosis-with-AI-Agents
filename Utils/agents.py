from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama

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

SPECIALIST_PROMPTS = {
    "Cardiologist": (
        "You are an experienced Cardiologist reviewing a patient's medical report.\n"
        "YOUR SCOPE: Cardiovascular system only — heart, blood vessels, circulation, blood pressure.\n\n"
        "Review the report and address the following:\n"
        "1. FINDINGS: List any cardiac-related symptoms, signs, or test results (ECG, Holter, echo, blood pressure, lipids).\n"
        "2. CONCERNS: Identify possible cardiac conditions (e.g. arrhythmia, structural abnormality, hypertension, coronary disease).\n"
        "3. NEXT STEPS: Recommend specific cardiac investigations or treatments needed.\n\n"
        "Be specific and clinical. Do not comment outside your specialty.\n\n"
        "Patient Report:\n{medical_report}"
    ),
    "Psychologist": (
        "You are an experienced Psychologist/Psychiatrist reviewing a patient's medical report.\n"
        "YOUR SCOPE: Mental health, psychological well-being, behavioral patterns, emotional state.\n\n"
        "Review the report and address the following:\n"
        "1. FINDINGS: List any mental health symptoms mentioned (anxiety, depression, sleep issues, panic, stress, trauma).\n"
        "2. CONCERNS: Identify possible psychological or psychiatric conditions with reasoning.\n"
        "3. NEXT STEPS: Recommend specific interventions (therapy type, counseling, psychiatric referral, medications if indicated).\n\n"
        "Be specific and clinical. Do not comment outside your specialty.\n\n"
        "Patient Report:\n{medical_report}"
    ),
    "Pulmonologist": (
        "You are an experienced Pulmonologist reviewing a patient's medical report.\n"
        "YOUR SCOPE: Respiratory system — lungs, airways, breathing, oxygen levels.\n\n"
        "Review the report and address the following:\n"
        "1. FINDINGS: List any respiratory symptoms or test results (spirometry, O2 sat, chest X-ray, breath sounds).\n"
        "2. CONCERNS: Identify possible respiratory conditions (e.g. asthma, COPD, infection, pulmonary embolism).\n"
        "3. NEXT STEPS: Recommend specific pulmonary investigations or treatments.\n\n"
        "Be specific and clinical. Do not comment outside your specialty.\n\n"
        "Patient Report:\n{medical_report}"
    ),
    "Neurologist": (
        "You are an experienced Neurologist reviewing a patient's medical report.\n"
        "YOUR SCOPE: Nervous system — brain, spinal cord, peripheral nerves, cognition, movement.\n\n"
        "Review the report and address the following:\n"
        "1. FINDINGS: List any neurological symptoms (headaches, memory loss, dizziness, tremors, numbness, seizures).\n"
        "2. CONCERNS: Identify possible neurological conditions (e.g. migraine, Alzheimer's, neuropathy, stroke risk, Parkinson's).\n"
        "3. NEXT STEPS: Recommend specific neurological tests (MRI, EEG, nerve conduction studies) or treatments.\n\n"
        "Be specific and clinical. Do not comment outside your specialty.\n\n"
        "Patient Report:\n{medical_report}"
    ),
    "Endocrinologist": (
        "You are an experienced Endocrinologist reviewing a patient's medical report.\n"
        "YOUR SCOPE: Hormonal and metabolic systems — diabetes, thyroid, adrenal, pituitary, reproductive hormones.\n\n"
        "Review the report and address the following:\n"
        "1. FINDINGS: List any metabolic or hormonal markers (glucose, HbA1c, TSH, insulin, cortisol, hormone levels).\n"
        "2. CONCERNS: Identify possible endocrine conditions (e.g. Type 2 Diabetes, hypothyroidism, PCOS, Cushing's).\n"
        "3. NEXT STEPS: Recommend specific endocrine tests or management strategies.\n\n"
        "Be specific and clinical. Do not comment outside your specialty.\n\n"
        "Patient Report:\n{medical_report}"
    ),
    "Gastroenterologist": (
        "You are an experienced Gastroenterologist reviewing a patient's medical report.\n"
        "YOUR SCOPE: Digestive system — stomach, intestines, liver, pancreas, bowel.\n\n"
        "Review the report and address the following:\n"
        "1. FINDINGS: List any GI symptoms (nausea, vomiting, abdominal pain, diarrhea, constipation, bloating, reflux).\n"
        "2. CONCERNS: Identify possible GI conditions (e.g. IBS, Crohn's, GERD, liver disease, ulcers).\n"
        "3. NEXT STEPS: Recommend specific GI investigations (endoscopy, colonoscopy, stool tests, LFTs) or treatments.\n\n"
        "Be specific and clinical. Do not comment outside your specialty.\n\n"
        "Patient Report:\n{medical_report}"
    ),
    "Dermatologist": (
        "You are an experienced Dermatologist reviewing a patient's medical report.\n"
        "YOUR SCOPE: Skin, hair, nails — conditions, lesions, rashes, infections.\n\n"
        "Review the report and address the following:\n"
        "1. FINDINGS: List any skin-related symptoms (rashes, lesions, itching, hair loss, nail changes, wounds).\n"
        "2. CONCERNS: Identify possible dermatological conditions (e.g. eczema, psoriasis, alopecia, infection, skin cancer risk).\n"
        "3. NEXT STEPS: Recommend specific dermatological tests or treatments (biopsy, topical/systemic therapy).\n\n"
        "Be specific and clinical. Do not comment outside your specialty.\n\n"
        "Patient Report:\n{medical_report}"
    ),
    "Hematologist": (
        "You are an experienced Hematologist reviewing a patient's medical report.\n"
        "YOUR SCOPE: Blood and blood disorders — CBC, anemia, clotting, blood cancers, iron levels.\n\n"
        "Review the report and address the following:\n"
        "1. FINDINGS: List any blood-related findings (hemoglobin, WBC, RBC, platelets, iron, ferritin, clotting).\n"
        "2. CONCERNS: Identify possible hematological conditions (e.g. anemia, thrombocytopenia, leukemia, clotting disorder).\n"
        "3. NEXT STEPS: Recommend specific blood tests or hematology referrals needed.\n\n"
        "Be specific and clinical. Do not comment outside your specialty.\n\n"
        "Patient Report:\n{medical_report}"
    ),
}


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
        self.medical_report = medical_report
        self.role = role
        self.extra_info = extra_info
        self.prompt_template = self.create_prompt_template()
        self.model = ChatOllama(
            model=OLLAMA_MODEL,
            temperature=0,
            base_url="http://localhost:11434",
        )

    def create_prompt_template(self):
        if self.role == "MultidisciplinaryTeam":
            specialist_section = ""
            for specialist, report in self.extra_info.get("specialist_reports", {}).items():
                specialist_section += f"[{specialist}]\n{report}\n\n"

            template = (
                "You are the lead physician chairing a Multidisciplinary Team (MDT) meeting.\n"
                "The following specialists have reviewed the same patient:\n\n"
                + specialist_section +
                "YOUR TASK: Write a single concise Patient Summary Report in EXACTLY this format:\n\n"
                "PATIENT SUMMARY\n"
                "---------------\n"
                "2-3 sentences describing the patient's overall condition in plain language.\n\n"
                "TOP 3 DIAGNOSES\n"
                "---------------\n"
                "1. [Condition] - one sentence: what it is and which specialists flagged it.\n"
                "2. [Condition] - one sentence: what it is and which specialists flagged it.\n"
                "3. [Condition] - one sentence: what it is and which specialists flagged it.\n\n"
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
                "Low / Moderate / High - one sentence justification.\n\n"
                "Keep the entire report under 400 words. Be direct. No repetition. No preamble."
            )
            return PromptTemplate.from_template(template)

        template = SPECIALIST_PROMPTS[self.role]
        return PromptTemplate.from_template(template)

    def run(self):
        print(f"  🔬 {self.role} is analyzing the report...")
        prompt = self.prompt_template.format(medical_report=self.medical_report or "")
        try:
            response = self.model.invoke(prompt)
            return response.content
        except Exception as e:
            print(f"  ❌ Error in {self.role}: {e}")
            print("  Tip: Make sure Ollama is running — run: ollama serve")
            return f"[{self.role} analysis unavailable: {e}]"


class Cardiologist(Agent):
    def __init__(self, medical_report):
        super().__init__(medical_report, "Cardiologist")

class Psychologist(Agent):
    def __init__(self, medical_report):
        super().__init__(medical_report, "Psychologist")

class Pulmonologist(Agent):
    def __init__(self, medical_report):
        super().__init__(medical_report, "Pulmonologist")

class Neurologist(Agent):
    def __init__(self, medical_report):
        super().__init__(medical_report, "Neurologist")

class Endocrinologist(Agent):
    def __init__(self, medical_report):
        super().__init__(medical_report, "Endocrinologist")

class Gastroenterologist(Agent):
    def __init__(self, medical_report):
        super().__init__(medical_report, "Gastroenterologist")

class Dermatologist(Agent):
    def __init__(self, medical_report):
        super().__init__(medical_report, "Dermatologist")

class Hematologist(Agent):
    def __init__(self, medical_report):
        super().__init__(medical_report, "Hematologist")


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