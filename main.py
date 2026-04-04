# ─────────────────────────────────────────────────────────────────────────────
# Medical Diagnosis with AI Agents
# Branch: feature/more-agents
# Runs 100% locally via Ollama — no API key needed.
# ─────────────────────────────────────────────────────────────────────────────
from concurrent.futures import ThreadPoolExecutor, as_completed
from Utils.agents import select_specialists, SPECIALIST_REGISTRY, MultidisciplinaryTeam
import os

# ── Config ────────────────────────────────────────────────────────────────────
REPORT_PATH = "Medical Reports/Medical Rerort - Michael Johnson - Panic Attack Disorder.txt"
OUTPUT_PATH = "results/final_diagnosis.txt"

# ── Load Report ───────────────────────────────────────────────────────────────
with open(REPORT_PATH, "r") as file:
    medical_report = file.read()

print("\n" + "=" * 65)
print("   🏥  MEDICAL DIAGNOSIS — AI AGENT SYSTEM")
print("=" * 65)
print(f"\n📄 Report: {REPORT_PATH}\n")

# ── Auto-select Relevant Specialists ─────────────────────────────────────────
selected = select_specialists(medical_report)
print(f"🤖 Auto-selected {len(selected)} specialists based on report content:")
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

# ── Save full output (specialist reports + summary) to file ──────────────────
specialists_used = ", ".join(responses.keys())
full_output = (
    f"MEDICAL DIAGNOSIS REPORT\n"
    f"{'=' * 65}\n"
    f"Report File  : {REPORT_PATH}\n"
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

# ── Print only the final summary to console ───────────────────────────────────
print("=" * 65)
print("📋  FINAL PATIENT SUMMARY\n")
print(final_summary)
print("\n" + "=" * 65)
print(f"✅  Full report (with all specialist notes) saved to: {OUTPUT_PATH}")
print("=" * 65 + "\n")