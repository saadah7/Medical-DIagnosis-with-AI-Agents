# ─────────────────────────────────────────────────────────────────────────────
# Medical Diagnosis with AI Agents — Streamlit Web UI
# Run with: streamlit run app.py
# ─────────────────────────────────────────────────────────────────────────────
import streamlit as st
import os
import json
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from Utils.agents import select_specialists, SPECIALIST_REGISTRY, MultidisciplinaryTeam, OLLAMA_MODEL

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Medical Diagnosis AI",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #0f1117; }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #161b27;
        border-right: 1px solid #2a3245;
    }

    /* Cards */
    .card {
        background: #161b27;
        border: 1px solid #2a3245;
        border-radius: 12px;
        padding: 20px 24px;
        margin-bottom: 16px;
    }
    .card-green  { border-left: 4px solid #22c55e; }
    .card-blue   { border-left: 4px solid #3b82f6; }
    .card-yellow { border-left: 4px solid #f59e0b; }
    .card-red    { border-left: 4px solid #ef4444; }
    .card-purple { border-left: 4px solid #a855f7; }

    /* Agent status badges */
    .badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
        margin: 3px;
    }
    .badge-running  { background: #1e3a5f; color: #60a5fa; }
    .badge-done     { background: #14532d; color: #4ade80; }
    .badge-pending  { background: #1f2937; color: #9ca3af; }

    /* Risk level pills */
    .risk-low      { background:#14532d; color:#4ade80; padding:4px 14px; border-radius:20px; font-weight:700; }
    .risk-moderate { background:#451a03; color:#fbbf24; padding:4px 14px; border-radius:20px; font-weight:700; }
    .risk-high     { background:#450a0a; color:#f87171; padding:4px 14px; border-radius:20px; font-weight:700; }

    /* Section headers */
    .section-header {
        font-size: 13px;
        font-weight: 700;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        color: #6b7280;
        margin-bottom: 8px;
        margin-top: 4px;
    }

    /* Diagnosis number circles */
    .diag-num {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 28px; height: 28px;
        border-radius: 50%;
        background: #3b82f6;
        color: white;
        font-weight: 700;
        font-size: 13px;
        margin-right: 10px;
        flex-shrink: 0;
    }
    .diag-row { display: flex; align-items: flex-start; margin-bottom: 10px; }

    /* Divider */
    .divider { border-top: 1px solid #2a3245; margin: 16px 0; }

    /* Hide streamlit branding */
    #MainMenu { visibility: hidden; }
    footer     { visibility: hidden; }

    /* Metric boxes */
    [data-testid="metric-container"] {
        background: #161b27;
        border: 1px solid #2a3245;
        border-radius: 10px;
        padding: 12px 16px;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] { background: #161b27; border-radius: 8px; }
    .stTabs [data-baseweb="tab"]      { color: #9ca3af; }
    .stTabs [aria-selected="true"]    { color: #3b82f6 !important; }

    /* Buttons */
    .stButton > button {
        background: #3b82f6;
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        padding: 10px 24px;
        width: 100%;
    }
    .stButton > button:hover { background: #2563eb; }

    /* Text input / textarea */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea {
        background: #1f2937;
        border: 1px solid #374151;
        color: #f9fafb;
        border-radius: 8px;
    }

    /* File uploader */
    [data-testid="stFileUploader"] {
        background: #1f2937;
        border: 2px dashed #374151;
        border-radius: 12px;
        padding: 12px;
    }
</style>
""", unsafe_allow_html=True)

# ── Session State Init ────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []
if "diagnosis_done" not in st.session_state:
    st.session_state.diagnosis_done = False
if "last_result" not in st.session_state:
    st.session_state.last_result = None

# ── Helpers ───────────────────────────────────────────────────────────────────
LOGS_DIR = "logs"
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs("results", exist_ok=True)

def save_log(record: dict):
    fname = os.path.join(LOGS_DIR, f"{record['id']}.json")
    with open(fname, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)

def load_logs() -> list:
    logs = []
    for fname in sorted(os.listdir(LOGS_DIR), reverse=True):
        if fname.endswith(".json"):
            with open(os.path.join(LOGS_DIR, fname), encoding="utf-8") as f:
                logs.append(json.load(f))
    return logs

def parse_risk(summary: str) -> str:
    s = summary.upper()
    if "OVERALL RISK LEVEL" in s:
        after = s.split("OVERALL RISK LEVEL")[-1]
        if "HIGH"     in after[:80]: return "High"
        if "MODERATE" in after[:80]: return "Moderate"
        if "LOW"      in after[:80]: return "Low"
    return "Unknown"

def risk_badge(risk: str) -> str:
    cls = {"High": "risk-high", "Moderate": "risk-moderate", "Low": "risk-low"}.get(risk, "badge-pending")
    return f'<span class="{cls}">{risk}</span>'

def run_diagnosis(medical_report: str, source_label: str):
    """Core diagnosis pipeline — returns result dict."""
    selected = select_specialists(medical_report)

    # Run specialists concurrently
    agents = {name: SPECIALIST_REGISTRY[name](medical_report) for name in selected}
    responses = {}

    status_placeholder = st.empty()

    def update_status(done_list, running_list, pending_list):
        badges = ""
        for s in done_list:
            badges += f'<span class="badge badge-done">✅ {s}</span> '
        for s in running_list:
            badges += f'<span class="badge badge-running">⚙️ {s}</span> '
        for s in pending_list:
            badges += f'<span class="badge badge-pending">⏳ {s}</span> '
        status_placeholder.markdown(
            f'<div class="card"><div class="section-header">Specialist Agents</div>{badges}</div>',
            unsafe_allow_html=True
        )

    done, pending = [], list(selected)
    update_status([], [], pending)

    def get_response(agent_name, agent):
        response = agent.run()
        return agent_name, response

    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(get_response, name, agent): name for name, agent in agents.items()}
        for future in as_completed(futures):
            agent_name, response = future.result()
            responses[agent_name] = response
            done.append(agent_name)
            if agent_name in pending:
                pending.remove(agent_name)
            update_status(done, [], pending)

    # Run MDT
    status_placeholder.markdown(
        '<div class="card card-blue"><div class="section-header">🧠 Multidisciplinary Team synthesizing all findings...</div></div>',
        unsafe_allow_html=True
    )
    team_agent = MultidisciplinaryTeam(specialist_reports=responses)
    final_summary = team_agent.run()
    status_placeholder.empty()

    # Build result
    risk = parse_risk(final_summary)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    result = {
        "id": run_id,
        "timestamp": datetime.now().strftime("%d %b %Y, %I:%M %p"),
        "source": source_label,
        "specialists": selected,
        "specialist_reports": responses,
        "final_summary": final_summary,
        "risk": risk,
        "model": OLLAMA_MODEL,
    }

    # Save log & result file
    save_log(result)
    output_path = f"results/{run_id}_diagnosis.txt"
    full_text = f"Source: {source_label}\nSpecialists: {', '.join(selected)}\n\n"
    for sp, rep in responses.items():
        full_text += f"[{sp}]\n{rep}\n\n"
    full_text += f"{'='*65}\nFINAL SUMMARY\n{'='*65}\n{final_summary}\n"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(full_text)
    result["output_path"] = output_path

    return result

def render_summary(result: dict):
    """Renders the final diagnosis summary card."""
    summary = result["final_summary"]
    risk    = result["risk"]
    specs   = result["specialists"]
    ts      = result["timestamp"]

    # Header row
    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        st.markdown(f"**🏥 Diagnosis Report** — {result['source']}")
    with col2:
        st.markdown(f"🕐 {ts}")
    with col3:
        st.markdown(f"Risk: {risk_badge(risk)}", unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # Metrics row
    m1, m2, m3 = st.columns(3)
    m1.metric("Specialists Consulted", len(specs))
    m2.metric("AI Model", result.get("model", OLLAMA_MODEL))
    m3.metric("Overall Risk", risk)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # Parse and render summary sections
    sections = {
        "PATIENT SUMMARY":        ("card-blue",   "👤 Patient Summary"),
        "TOP 3 DIAGNOSES":        ("card-purple",  "🔬 Top 3 Diagnoses"),
        "KEY FINDINGS":           ("card-yellow",  "📋 Key Findings"),
        "RECOMMENDED NEXT STEPS": ("card-green",   "✅ Recommended Next Steps"),
        "OVERALL RISK LEVEL":     ("card-red",     "⚠️ Overall Risk Level"),
    }

    lines = summary.split("\n")
    current_section = None
    current_lines   = []
    rendered        = {}

    def flush(section, lines, style, label):
        content = "\n".join(l for l in lines if l.strip() and "---" not in l).strip()
        if content:
            st.markdown(
                f'<div class="card {style}"><div class="section-header">{label}</div>'
                f'<div style="color:#e5e7eb;line-height:1.7">{content.replace(chr(10), "<br>")}</div></div>',
                unsafe_allow_html=True
            )
        rendered[section] = True

    for line in lines:
        matched = None
        for key in sections:
            if key in line.upper():
                matched = key
                break
        if matched:
            if current_section and current_section in sections:
                style, label = sections[current_section]
                flush(current_section, current_lines, style, label)
            current_section = matched
            current_lines   = []
        elif current_section:
            current_lines.append(line)

    if current_section and current_section in sections and current_section not in rendered:
        style, label = sections[current_section]
        flush(current_section, current_lines, style, label)

    # If parsing failed, just show the raw summary
    if not rendered:
        st.markdown(
            f'<div class="card"><div style="color:#e5e7eb;white-space:pre-wrap">{summary}</div></div>',
            unsafe_allow_html=True
        )

    # Download button
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.download_button(
        label="⬇️ Download Full Report (.txt)",
        data=open(result["output_path"], encoding="utf-8").read(),
        file_name=f"diagnosis_{result['id']}.txt",
        mime="text/plain",
        use_container_width=True
    )

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏥 Medical Diagnosis AI")
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    st.markdown("**🤖 Model**")
    st.markdown(
        f'<div class="card"><code style="color:#60a5fa">{OLLAMA_MODEL}</code><br>'
        f'<span style="color:#6b7280;font-size:12px">Change in Utils/agents.py</span></div>',
        unsafe_allow_html=True
    )

    st.markdown("**🩺 Available Specialists**")
    specialists = [
        "Cardiologist", "Psychologist", "Pulmonologist",
        "Neurologist", "Endocrinologist", "Gastroenterologist",
        "Dermatologist", "Hematologist"
    ]
    for sp in specialists:
        st.markdown(f"&nbsp;&nbsp;• {sp}")

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown(
        '<div style="color:#4b5563;font-size:12px">⚠️ For educational/research use only. '
        'Not a substitute for professional medical advice.</div>',
        unsafe_allow_html=True
    )

# ── Main Tabs ─────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["🔬 Diagnose", "📂 History"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — DIAGNOSE
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("## 🔬 Run a Diagnosis")
    st.markdown("Choose how to provide patient data, then click **Run Diagnosis**.")
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    input_mode = st.radio(
        "Input Mode",
        ["📄 Upload / Select Report File", "✏️ Enter Symptoms Manually"],
        horizontal=True,
        label_visibility="collapsed"
    )

    medical_report = None
    source_label   = ""

    # ── Mode A: File ──────────────────────────────────────────────────────────
    if "Upload" in input_mode:
        col_upload, col_select = st.columns(2)

        with col_upload:
            st.markdown("**Upload a report file**")
            uploaded = st.file_uploader("Upload .txt report", type=["txt"], label_visibility="collapsed")
            if uploaded:
                medical_report = uploaded.read().decode("utf-8", errors="ignore")
                source_label   = f"Uploaded: {uploaded.name}"
                st.success(f"✅ Loaded: {uploaded.name}")

        with col_select:
            st.markdown("**Or pick from existing reports**")
            reports_dir  = "Medical Reports"
            report_files = []
            if os.path.isdir(reports_dir):
                report_files = [f for f in os.listdir(reports_dir) if f.endswith(".txt")]

            if report_files:
                chosen = st.selectbox("Select report", ["— select —"] + report_files, label_visibility="collapsed")
                if chosen != "— select —":
                    path = os.path.join(reports_dir, chosen)
                    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
                        medical_report = fh.read()
                    source_label = f"File: {chosen}"
                    st.success(f"✅ Loaded: {chosen}")
            else:
                st.info("No reports found in 'Medical Reports/' folder.")

        if medical_report:
            with st.expander("📄 Preview report content"):
                st.text(medical_report[:800] + ("..." if len(medical_report) > 800 else ""))

    # ── Mode B: Manual Input ──────────────────────────────────────────────────
    else:
        st.markdown("**Patient Information**")
        c1, c2, c3 = st.columns(3)
        name   = c1.text_input("Patient Name",   placeholder="e.g. John Smith")
        age    = c2.text_input("Age",             placeholder="e.g. 45")
        gender = c3.selectbox("Gender", ["— select —", "Male", "Female", "Other", "Prefer not to say"])

        st.markdown("**Presenting Symptoms**")
        symptoms = st.text_area(
            "Symptoms",
            placeholder="Describe all symptoms in detail...\ne.g. chest pain, shortness of breath, dizziness, anxiety attacks for 3 months",
            height=130,
            label_visibility="collapsed"
        )

        col_hist, col_meds = st.columns(2)
        with col_hist:
            st.markdown("**Medical History**")
            history = st.text_area("History", placeholder="e.g. hypertension, diabetes diagnosed 2020...", height=100, label_visibility="collapsed")
        with col_meds:
            st.markdown("**Current Medications**")
            medications = st.text_area("Medications", placeholder="e.g. Metformin 500mg, Lisinopril 10mg...", height=100, label_visibility="collapsed")

        if symptoms.strip():
            g = gender if gender != "— select —" else "Not specified"
            medical_report = (
                f"Patient Name: {name or 'Unknown'}\n"
                f"Age: {age or 'Unknown'}\n"
                f"Gender: {g}\n\n"
                f"Presenting Symptoms:\n{symptoms}\n\n"
                f"Medical History:\n{history or 'None provided.'}\n\n"
                f"Current Medications:\n{medications or 'None provided.'}\n"
            )
            source_label = f"Manual Input — {name or 'Unknown Patient'}"

    # ── Run Button ────────────────────────────────────────────────────────────
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    run_clicked = st.button("🚀 Run Diagnosis", disabled=(medical_report is None), use_container_width=True)

    if run_clicked and medical_report:
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.markdown("### ⚙️ Running Analysis...")

        with st.spinner("Agents are analyzing the patient data..."):
            result = run_diagnosis(medical_report, source_label)

        st.session_state.last_result = result
        st.session_state.history.insert(0, result)
        st.session_state.diagnosis_done = True

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.markdown("### 📋 Diagnosis Complete")
        render_summary(result)

    elif st.session_state.diagnosis_done and st.session_state.last_result:
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.markdown("### 📋 Last Diagnosis Result")
        render_summary(st.session_state.last_result)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — HISTORY
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("## 📂 Diagnosis History")
    st.markdown("All past diagnosis sessions are saved locally in the `logs/` folder.")
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    logs = load_logs()

    if not logs:
        st.markdown(
            '<div class="card" style="text-align:center;color:#6b7280">'
            '📭 No diagnosis history yet. Run a diagnosis first.</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(f"**{len(logs)} session(s) found**")
        for log in logs:
            risk   = log.get("risk", "Unknown")
            rclass = {"High": "card-red", "Moderate": "card-yellow", "Low": "card-green"}.get(risk, "")
            specs  = ", ".join(log.get("specialists", []))

            with st.expander(f"🗓️ {log.get('timestamp', 'Unknown')}  —  {log.get('source', '')}  —  Risk: {risk}"):
                col1, col2, col3 = st.columns(3)
                col1.metric("Specialists", len(log.get("specialists", [])))
                col2.metric("Risk Level", risk)
                col3.metric("Model", log.get("model", "—"))

                st.markdown(f"**Specialists:** {specs}")
                st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
                st.markdown("**Final Summary:**")
                st.markdown(
                    f'<div class="card {rclass}"><div style="color:#e5e7eb;white-space:pre-wrap;font-size:14px">'
                    f'{log.get("final_summary", "")}</div></div>',
                    unsafe_allow_html=True
                )

                # Re-download
                out_path = log.get("output_path", "")
                if out_path and os.path.exists(out_path):
                    with open(out_path, encoding="utf-8") as fh:
                        st.download_button(
                            "⬇️ Download Full Report",
                            data=fh.read(),
                            file_name=f"diagnosis_{log['id']}.txt",
                            mime="text/plain",
                            key=f"dl_{log['id']}"
                        )