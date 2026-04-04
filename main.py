# Importing the needed modules
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
from concurrent.futures import ThreadPoolExecutor, as_completed
from Utils.agents import Cardiologist, Psychologist, Pulmonologist, MultidisciplinaryTeam
import os

# NOTE: No API key or .env file needed — this project runs 100% locally via Ollama.
# Make sure Ollama is installed and running before executing this script.
# See README_LOCAL.md for setup instructions.
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Read the medical report (change this path to try different reports)
REPORT_PATH = "Medical Reports/Medical Rerort - Michael Johnson - Panic Attack Disorder.txt"

with open(REPORT_PATH, "r") as file:
    medical_report = file.read()

print(f"\n📄 Loaded report: {REPORT_PATH}\n")
print("=" * 60)

agents = {
    "Cardiologist": Cardiologist(medical_report),
    "Psychologist": Psychologist(medical_report),
    "Pulmonologist": Pulmonologist(medical_report)
}

# Function to run each agent and get their response
def get_response(agent_name, agent):
    response = agent.run()
    return agent_name, response

# Run the agents concurrently and collect responses
responses = {}
with ThreadPoolExecutor() as executor:
    futures = {executor.submit(get_response, name, agent): name for name, agent in agents.items()}
    for future in as_completed(futures):
        agent_name, response = future.result()
        responses[agent_name] = response

team_agent = MultidisciplinaryTeam(
    cardiologist_report=responses["Cardiologist"],
    psychologist_report=responses["Psychologist"],
    pulmonologist_report=responses["Pulmonologist"]
)

# Run the MultidisciplinaryTeam agent to generate the final diagnosis
final_diagnosis = team_agent.run()
final_diagnosis_text = "### Final Diagnosis:\n\n" + final_diagnosis

txt_output_path = "results/final_diagnosis.txt"
os.makedirs(os.path.dirname(txt_output_path), exist_ok=True)

with open(txt_output_path, "w") as txt_file:
    txt_file.write(final_diagnosis_text)

print("\n" + "=" * 60)
print(final_diagnosis_text)
print("=" * 60)
print(f"\n✅ Final diagnosis saved to: {txt_output_path}")
