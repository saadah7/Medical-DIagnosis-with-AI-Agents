# ─────────────────────────────────────────────────────────────────────────────
# Medical Diagnosis with AI Agents
# Branch: feature/more-agents
# Runs 100% locally via Ollama — no API key needed.
# ─────────────────────────────────────────────────────────────────────────────
from concurrent.futures import ThreadPoolExecutor, as_completed
from Utils.agents import select_specialists, SPECIALIST_REGISTRY, MultidisciplinaryTeam
import os

# ── Mode Selection ────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("   🏥  MEDICAL DIAGNOSIS — AI AGENT SYSTEM")
print("=" * 65)
print("\nHow would you like to provide patient data?")
print("  1. Load from a Medical Report file")
print("  2. Enter patient symptoms manually")
print()

while True:
    mode = input("Enter 1 or 2: ").strip()
    if mode in ("1", "2"):
        break
    print("  ⚠️  Please enter 1 or 2.")

# ── Mode 1: Load from file (existing behaviour) ───────────────────────────────
if mode == "1":
    print("\nAvailable reports in 'Medical Reports/' folder:")
    reports_dir = "Medical Reports"
    report_files = [f for f in os.listdir(reports_dir) if f.endswith(".txt")]
    for i, f in enumerate(report_files, 1):
        print(f"  {i}. {f}")
    print()

    while True:
        try:
            choice = int(input("Enter report number: ").strip())
            if 1 <= choice <= len(report_files):
                break
            print(f"  ⚠️  Enter a number between 1 and {len(report_files)}.")
        except ValueError:
            print("  ⚠️  Please enter a valid number.")

    REPORT_PATH = os.path.join(reports_dir, report_files[choice - 1])
    with open(REPORT_PATH, "r", encoding="utf-8") as file:
        medical_report = file.read()

    source_label = f"File: {REPORT_PATH}"
    OUTPUT_PATH = "results/final_diagnosis.txt"

# ── Mode 2: Manual symptom input ─────────────────────────────────────────────
else:
    print("\n" + "-" * 65)
    print("📝  PATIENT INFORMATION")
    print("-" * 65)
    print("Please provide the following details.\n")

    name    = input("Patient Name       : ").strip() or "Unknown"
    age     = input("Age                : ").strip() or "Unknown"
    gender  = input("Gender             : ").strip() or "Unknown"

    print("\nDescribe the patient's symptoms (press Enter twice when done):")
    print("Example: chest pain, shortness of breath, dizziness, anxiety\n")
    symptoms_lines = []
    while True:
        line = input()
        if line == "" and symptoms_lines and symptoms_lines[-1] == "":
            break
        symptoms_lines.append(line)
    symptoms = "\n".join(symptoms_lines).strip()

    print("\nAny known medical history? (press Enter to skip):")
    history = input("> ").strip() or "None provided."

    print("\nAny current medications? (press Enter to skip):")
    medications = input("> ").strip() or "None provided."

    # Build a structured report from user input
    medical_report = (
        f"Patient Name: {name}\n"
        f"Age: {age}\n"
        f"Gender: {gender}\n\n"
        f"Presenting Symptoms:\n{symptoms}\n\n"
        f"Medical History:\n{history}\n\n"
        f"Current Medications:\n{medications}\n"
    )

    source_label = f"Manual Input — Patient: {name}"
    safe_name = name.replace(" ", "_")
    OUTPUT_PATH = f"results/{safe_name}_diagnosis.txt"

# ── Display what we're working with ──────────────────────────────────────────
print("\n" + "=" * 65)
print(f"📄 {source_label}\n")

# ── Auto-select Relevant Specialists ─────────────────────────────────────────
selected = select_specialists(medical_report)
print(f"🤖 Auto-selected {len(selected)} specialists based on patient data:")
for s in selected:
    print(f"   • {s}")
print()

# ── Instantiate Selected Agents ───────────────────────────────────────────────
agents = {name: SPECIALIST_REGISTRY[name](medical_report) for name in selected}

# ── Run Agents Concurrently ───────────────────────────────────────────────────
print("=" * 65)
print("⚙️  Running specialist agents in parallel...\n")

responses = {}

def get_response(agent_name, agent):
    response = agent.run()
    return agent_name, response

with ThreadPoolExecutor() as executor:
    futures = {executor.submit(get_response, name, agent): name for name, agent in agents.items()}
    for future in as_completed(futures):
        agent_name, response = future.result()
        responses[agent_name] = response
        print(f"  ✅ {agent_name} done.")

# ── Run Multidisciplinary Team ────────────────────────────────────────────────
print(f"\n{'=' * 65}")
print("🧠  Multidisciplinary Team synthesizing all findings...\n")

team_agent = MultidisciplinaryTeam(specialist_reports=responses)
final_summary = team_agent.run()

# ── Save full output ──────────────────────────────────────────────────────────
specialists_used = ", ".join(responses.keys())
full_output = (
    f"MEDICAL DIAGNOSIS REPORT\n"
    f"{'=' * 65}\n"
    f"Source       : {source_label}\n"
    f"Specialists  : {specialists_used}\n"
    f"{'=' * 65}\n\n"
    f"INDIVIDUAL SPECIALIST REPORTS\n"
    f"{'-' * 65}\n"
)
for specialist, report in responses.items():
    full_output += f"\n[{specialist}]\n{report}\n"

full_output += (
    f"\n{'=' * 65}\n"
    f"FINAL PATIENT SUMMARY (Multidisciplinary Team)\n"
    f"{'=' * 65}\n"
    f"{final_summary}\n"
)

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    f.write(full_output)

# ── Print final summary ───────────────────────────────────────────────────────
print("=" * 65)
print("📋  FINAL PATIENT SUMMARY\n")
print(final_summary)
print("\n" + "=" * 65)
print(f"✅  Full report saved to: {OUTPUT_PATH}")
print("=" * 65 + "\n")