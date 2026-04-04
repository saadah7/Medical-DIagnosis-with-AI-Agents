from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama

# Change this to any model you have pulled via ollama
OLLAMA_MODEL = "llama3.2"


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
            template = (
                "Act like a multidisciplinary team of healthcare professionals.\n"
                "You will receive a medical report of a patient visited by a Cardiologist, Psychologist, and Pulmonologist.\n"
                "Task: Review the reports and come up with a list of 3 possible health issues of the patient.\n"
                "Return bullet points of 3 possible health issues with a reason for each.\n\n"
                "Cardiologist Report: " + self.extra_info.get("cardiologist_report", "") + "\n\n"
                "Psychologist Report: " + self.extra_info.get("psychologist_report", "") + "\n\n"
                "Pulmonologist Report: " + self.extra_info.get("pulmonologist_report", "")
            )
            return PromptTemplate.from_template(template)

        cardiologist_template = (
            "Act like a cardiologist. You will receive a medical report of a patient.\n"
            "Task: Review the patient cardiac workup, including ECG, blood tests, Holter monitor results, and echocardiogram.\n"
            "Focus: Determine if there are any subtle signs of cardiac issues that could explain the patient symptoms.\n"
            "Recommendation: Provide guidance on further cardiac testing or monitoring needed.\n"
            "Please only return the possible causes and the recommended next steps.\n\n"
            "Medical Report: {medical_report}"
        )

        psychologist_template = (
            "Act like a psychologist. You will receive a patient report.\n"
            "Task: Review the patient report and provide a psychological assessment.\n"
            "Focus: Identify any potential mental health issues such as anxiety, depression, or trauma.\n"
            "Recommendation: Offer guidance on how to address these mental health concerns.\n"
            "Please only return the possible mental health issues and the recommended next steps.\n\n"
            "Patient Report: {medical_report}"
        )

        pulmonologist_template = (
            "Act like a pulmonologist. You will receive a patient report.\n"
            "Task: Review the patient report and provide a pulmonary assessment.\n"
            "Focus: Identify any potential respiratory issues such as asthma, COPD, or lung infections.\n"
            "Recommendation: Offer guidance on how to address these respiratory concerns.\n"
            "Please only return the possible respiratory issues and the recommended next steps.\n\n"
            "Patient Report: {medical_report}"
        )

        templates = {
            "Cardiologist": cardiologist_template,
            "Psychologist": psychologist_template,
            "Pulmonologist": pulmonologist_template,
        }
        return PromptTemplate.from_template(templates[self.role])

    def run(self):
        print(f"{self.role} is running (local Ollama model: {OLLAMA_MODEL})...")
        prompt = self.prompt_template.format(medical_report=self.medical_report)
        try:
            response = self.model.invoke(prompt)
            return response.content
        except Exception as e:
            print(f"Error in {self.role}:", e)
            print("Tip: Make sure Ollama is running — run: ollama serve")
            return None


class Cardiologist(Agent):
    def __init__(self, medical_report):
        super().__init__(medical_report, "Cardiologist")


class Psychologist(Agent):
    def __init__(self, medical_report):
        super().__init__(medical_report, "Psychologist")


class Pulmonologist(Agent):
    def __init__(self, medical_report):
        super().__init__(medical_report, "Pulmonologist")


class MultidisciplinaryTeam(Agent):
    def __init__(self, cardiologist_report, psychologist_report, pulmonologist_report):
        extra_info = {
            "cardiologist_report": cardiologist_report,
            "psychologist_report": psychologist_report,
            "pulmonologist_report": pulmonologist_report,
        }
        super().__init__(role="MultidisciplinaryTeam", extra_info=extra_info)