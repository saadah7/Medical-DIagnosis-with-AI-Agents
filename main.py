# ─────────────────────────────────────────────────────────────────────────────
# Medical Diagnosis with AI Agents — CLI
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

# ── Mode 1: File ──────────────────────────────────────────────────────────────
if mode == "1":
    reports_dir  = "Medical Reports"
    report_files = [f for f in os.listdir(reports_dir) if f.endswith(".txt")]
    print("\nAvailable reports:")
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
    with open(REPORT_PATH, "r", encoding="utf-8") as f:
        medical_report = f.read()
    source_label = f"File: {REPORT_PATH}"
    OUTPUT_PATH  = "results/final_diagnosis.txt"

# ── Mode 2: Manual Input ──────────────────────────────────────────────────────
else:
    print("\n" + "-" * 65)
    print("📝  PATIENT INFORMATION")
    print("-" * 65)
    name   = input("Patient Name       : ").strip() or "Unknown"
    age    = input("Age                : ").strip() or "Unknown"
    gender = input("Gender             : ").strip() or "Unknown"

    print("\nDescribe symptoms (press Enter twice when done):")
    symptoms_lines = []
    while True:
        line = input()
        if line == "" and symptoms_lines and symptoms_lines[-1] == "":
            break
        symptoms_lines.append(line)
    symptoms = "\n".join(symptoms_lines).strip()

    history     = input("\nMedical history (Enter to skip): ").strip() or "None provided."
    medications = input("Current medications (Enter to skip): ").strip() or "None provided."

    medical_report = (
        f"Patient Name: {name}\nAge: {age}\nGender: {gender}\n\n"
        f"Presenting Symptoms:\n{symptoms}\n\n"
        f"Medical History:\n{history}\n\n"
        f"Current Medications:\n{medications}\n"
    )
    source_label = f"Manual Input — Patient: {name}"
    OUTPUT_PATH  = f"results/{name.replace(' ','_')}_diagnosis.txt"

# ── Auto-select Specialists ───────────────────────────────────────────────────
print("\n" + "=" * 65)
print(f"📄 {source_label}\n")
selected = select_specialists(medical_report)
print(f"🤖 Auto-selected {len(selected)} specialists:")
for s in selected:
    print(f"   • {s}")
print()

# ── Run Agents ────────────────────────────────────────────────────────────────
agents = {name: SPECIALIST_REGISTRY[name](medical_report) for name in selected}
print("=" * 65)
print("⚙️  Running specialist agents in parallel...\n")

responses = {}

def get_response(agent_name, agent):
    return agent_name, agent.run()

with ThreadPoolExecutor() as executor:
    futures = {executor.submit(get_response, name, agent): name for name, agent in agents.items()}
    for future in as_completed(futures):
        agent_name, response = future.result()
        responses[agent_name] = response
        conf = response.get("confidence", "?")
        print(f"  ✅ {agent_name} done. [Confidence: {conf}]")

# ── Print Confidence Summary ──────────────────────────────────────────────────
print(f"\n{'=' * 65}")
print("🎯  CONFIDENCE SCORES\n")
for sp, data in responses.items():
    conf   = data.get("confidence", "?")
    reason = data.get("confidence_reason", "")
    bar    = {"High": "████████████", "Medium": "████████    ", "Low": "████        "}.get(conf, "?")
    print(f"  {sp:<22} [{bar}] {conf:<6}  {reason}")

# ── Run MDT ───────────────────────────────────────────────────────────────────
print(f"\n{'=' * 65}")
print("🧠  Multidisciplinary Team synthesizing all findings...\n")
team_agent    = MultidisciplinaryTeam(specialist_reports=responses)
final_summary = team_agent.run()

# ── Save Output ───────────────────────────────────────────────────────────────
specialists_used = ", ".join(responses.keys())
full_output = (
    f"MEDICAL DIAGNOSIS REPORT\n{'=' * 65}\n"
    f"Source      : {source_label}\n"
    f"Specialists : {specialists_used}\n"
    f"{'=' * 65}\n\n"
    f"CONFIDENCE SCORES\n{'-' * 65}\n"
)
for sp, data in responses.items():
    full_output += f"  {sp}: {data.get('confidence','?')} — {data.get('confidence_reason','')}\n"

full_output += f"\nINDIVIDUAL SPECIALIST REPORTS\n{'-' * 65}\n"
for sp, data in responses.items():
    full_output += f"\n[{sp}]\n{data.get('report','')}\n"

full_output += f"\n{'=' * 65}\nFINAL PATIENT SUMMARY\n{'=' * 65}\n{final_summary}\n"

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    f.write(full_output)

# ── Print Final ───────────────────────────────────────────────────────────────
print("=" * 65)
print("📋  FINAL PATIENT SUMMARY\n")
print(final_summary)
print("\n" + "=" * 65)
print(f"✅  Full report saved to: {OUTPUT_PATH}")
print("=" * 65 + "\n")