# ─────────────────────────────────────────────────────────────────────────────
# Medical Diagnosis with AI Agents — Streamlit Web UI
# Run with: streamlit run app.py
# ─────────────────────────────────────────────────────────────────────────────
import streamlit as st
import os
import json
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
    .stApp { background-color: #0f1117; }
    [data-testid="stSidebar"] { background-color: #161b27; border-right: 1px solid #2a3245; }

    .card {
        background: #161b27; border: 1px solid #2a3245;
        border-radius: 12px; padding: 20px 24px; margin-bottom: 16px;
    }
    .card-green  { border-left: 4px solid #22c55e; }
    .card-blue   { border-left: 4px solid #3b82f6; }
    .card-yellow { border-left: 4px solid #f59e0b; }
    .card-red    { border-left: 4px solid #ef4444; }
    .card-purple { border-left: 4px solid #a855f7; }

    .badge { display:inline-block; padding:3px 10px; border-radius:20px; font-size:12px; font-weight:600; margin:3px; }
    .badge-running { background:#1e3a5f; color:#60a5fa; }
    .badge-done    { background:#14532d; color:#4ade80; }
    .badge-pending { background:#1f2937; color:#9ca3af; }

    .risk-low      { background:#14532d; color:#4ade80; padding:4px 14px; border-radius:20px; font-weight:700; }
    .risk-moderate { background:#451a03; color:#fbbf24; padding:4px 14px; border-radius:20px; font-weight:700; }
    .risk-high     { background:#450a0a; color:#f87171; padding:4px 14px; border-radius:20px; font-weight:700; }

    .conf-high   { background:#14532d; color:#4ade80; padding:2px 10px; border-radius:12px; font-size:12px; font-weight:700; }
    .conf-medium { background:#451a03; color:#fbbf24; padding:2px 10px; border-radius:12px; font-size:12px; font-weight:700; }
    .conf-low    { background:#1f2937; color:#9ca3af; padding:2px 10px; border-radius:12px; font-size:12px; font-weight:700; }

    .section-header { font-size:13px; font-weight:700; letter-spacing:1.5px; text-transform:uppercase; color:#6b7280; margin-bottom:8px; }
    .divider { border-top: 1px solid #2a3245; margin: 16px 0; }

    #MainMenu { visibility: hidden; }
    footer     { visibility: hidden; }

    [data-testid="metric-container"] { background:#161b27; border:1px solid #2a3245; border-radius:10px; padding:12px 16px; }
    .stTabs [data-baseweb="tab-list"] { background:#161b27; border-radius:8px; }
    .stTabs [data-baseweb="tab"]      { color:#9ca3af; }
    .stTabs [aria-selected="true"]    { color:#3b82f6 !important; }

    .stButton > button { background:#3b82f6; color:white; border:none; border-radius:8px; font-weight:600; padding:10px 24px; width:100%; }
    .stButton > button:hover { background:#2563eb; }

    .stTextInput > div > div > input,
    .stTextArea  > div > div > textarea { background:#1f2937; border:1px solid #374151; color:#f9fafb; border-radius:8px; }

    [data-testid="stFileUploader"] { background:#1f2937; border:2px dashed #374151; border-radius:12px; padding:12px; }

    /* Confidence bar track */
    .conf-bar-track { background:#1f2937; border-radius:999px; height:8px; width:100%; margin-top:4px; }
    .conf-bar-fill-high   { background:#22c55e; border-radius:999px; height:8px; }
    .conf-bar-fill-medium { background:#f59e0b; border-radius:999px; height:8px; }
    .conf-bar-fill-low    { background:#6b7280; border-radius:999px; height:8px; }
</style>
""", unsafe_allow_html=True)

# ── Session State ─────────────────────────────────────────────────────────────
if "diagnosis_done"  not in st.session_state: st.session_state.diagnosis_done  = False
if "last_result"     not in st.session_state: st.session_state.last_result     = None

# ── Helpers ───────────────────────────────────────────────────────────────────
LOGS_DIR = "logs"
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs("results", exist_ok=True)

def save_log(record: dict):
    with open(os.path.join(LOGS_DIR, f"{record['id']}.json"), "w", encoding="utf-8") as f:
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
    cls = {"High":"risk-high","Moderate":"risk-moderate","Low":"risk-low"}.get(risk,"badge-pending")
    return f'<span class="{cls}">{risk}</span>'

def conf_badge(conf: str) -> str:
    cls = {"High":"conf-high","Medium":"conf-medium","Low":"conf-low"}.get(conf,"conf-low")
    return f'<span class="{cls}">{conf}</span>'

def conf_bar(conf: str) -> str:
    widths = {"High": "100%", "Medium": "60%", "Low": "25%"}
    cls    = {"High": "conf-bar-fill-high", "Medium": "conf-bar-fill-medium", "Low": "conf-bar-fill-low"}
    w = widths.get(conf, "25%")
    c = cls.get(conf, "conf-bar-fill-low")
    return f'<div class="conf-bar-track"><div class="{c}" style="width:{w}"></div></div>'

def run_diagnosis(medical_report: str, source_label: str):
    selected = select_specialists(medical_report)
    agents   = {name: SPECIALIST_REGISTRY[name](medical_report) for name in selected}
    responses = {}

    status_placeholder = st.empty()

    def update_status(done_list, pending_list):
        badges = "".join(f'<span class="badge badge-done">✅ {s}</span> ' for s in done_list)
        badges += "".join(f'<span class="badge badge-pending">⏳ {s}</span> ' for s in pending_list)
        status_placeholder.markdown(
            f'<div class="card"><div class="section-header">Specialist Agents</div>{badges}</div>',
            unsafe_allow_html=True
        )

    done, pending = [], list(selected)
    update_status([], pending)

    def get_response(agent_name, agent):
        return agent_name, agent.run()

    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(get_response, name, agent): name for name, agent in agents.items()}
        for future in as_completed(futures):
            agent_name, response = future.result()
            responses[agent_name] = response
            done.append(agent_name)
            if agent_name in pending: pending.remove(agent_name)
            update_status(done, pending)

    status_placeholder.markdown(
        '<div class="card card-blue"><div class="section-header">🧠 Multidisciplinary Team synthesizing all findings...</div></div>',
        unsafe_allow_html=True
    )

    team_agent   = MultidisciplinaryTeam(specialist_reports=responses)
    final_summary = team_agent.run()
    status_placeholder.empty()

    risk   = parse_risk(final_summary)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    result = {
        "id":                run_id,
        "timestamp":         datetime.now().strftime("%d %b %Y, %I:%M %p"),
        "source":            source_label,
        "specialists":       selected,
        "specialist_reports": responses,
        "final_summary":     final_summary,
        "risk":              risk,
        "model":             OLLAMA_MODEL,
    }

    # Save full text report
    output_path = f"results/{run_id}_diagnosis.txt"
    full_text = f"Source: {source_label}\nSpecialists: {', '.join(selected)}\n\n"
    for sp, data in responses.items():
        conf   = data.get("confidence", "?")
        reason = data.get("confidence_reason", "")
        report = data.get("report", "")
        full_text += f"[{sp}] — Confidence: {conf} ({reason})\n{report}\n\n"
    full_text += f"{'='*65}\nFINAL SUMMARY\n{'='*65}\n{final_summary}\n"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(full_text)

    result["output_path"] = output_path
    save_log(result)
    return result

def render_confidence_panel(specialist_reports: dict):
    """Renders the confidence score panel for all specialists."""
    st.markdown('<div class="section-header">🎯 Specialist Confidence Scores</div>', unsafe_allow_html=True)

    cols = st.columns(2)
    for i, (specialist, data) in enumerate(specialist_reports.items()):
        conf   = data.get("confidence", "Medium")
        reason = data.get("confidence_reason", "")
        with cols[i % 2]:
            st.markdown(
                f'<div class="card" style="padding:14px 18px;margin-bottom:10px">'
                f'<div style="display:flex;justify-content:space-between;align-items:center">'
                f'<span style="color:#e5e7eb;font-weight:600;font-size:14px">{specialist}</span>'
                f'{conf_badge(conf)}'
                f'</div>'
                f'{conf_bar(conf)}'
                f'<div style="color:#6b7280;font-size:12px;margin-top:6px">{reason}</div>'
                f'</div>',
                unsafe_allow_html=True
            )

def render_summary(result: dict):
    summary  = result["final_summary"]
    risk     = result["risk"]
    specs    = result["specialists"]
    ts       = result["timestamp"]
    sp_data  = result.get("specialist_reports", {})

    # Header
    col1, col2, col3 = st.columns([3, 1, 1])
    with col1: st.markdown(f"**🏥 Diagnosis Report** — {result['source']}")
    with col2: st.markdown(f"🕐 {ts}")
    with col3: st.markdown(f"Risk: {risk_badge(risk)}", unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # Metrics
    m1, m2, m3 = st.columns(3)
    m1.metric("Specialists Consulted", len(specs))
    m2.metric("AI Model", result.get("model", OLLAMA_MODEL))
    m3.metric("Overall Risk", risk)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # ── Confidence Panel ──────────────────────────────────────────────────────
    if sp_data:
        render_confidence_panel(sp_data)
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # ── Final Summary Sections ────────────────────────────────────────────────
    sections = {
        "PATIENT SUMMARY":        ("card-blue",   "👤 Patient Summary"),
        "TOP 3 DIAGNOSES":        ("card-purple",  "🔬 Top 3 Diagnoses"),
        "KEY FINDINGS":           ("card-yellow",  "📋 Key Findings"),
        "RECOMMENDED NEXT STEPS": ("card-green",   "✅ Recommended Next Steps"),
        "OVERALL RISK LEVEL":     ("card-red",     "⚠️ Overall Risk Level"),
    }

    lines           = summary.split("\n")
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

    if not rendered:
        st.markdown(
            f'<div class="card"><div style="color:#e5e7eb;white-space:pre-wrap">{summary}</div></div>',
            unsafe_allow_html=True
        )

    # Download
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    if os.path.exists(result.get("output_path", "")):
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
    st.markdown("**🩺 Confidence Scale**")
    st.markdown(
        '<div class="card" style="padding:12px 16px">'
        f'<div style="margin-bottom:6px">{conf_badge("High")} &nbsp; 3+ specialty indicators found</div>'
        f'<div style="margin-bottom:6px">{conf_badge("Medium")} &nbsp; 1-2 indirect indicators</div>'
        f'<div>{conf_badge("Low")} &nbsp; Included as precaution</div>'
        '</div>',
        unsafe_allow_html=True
    )
    st.markdown("**🩺 Available Specialists**")
    for sp in ["Cardiologist","Psychologist","Pulmonologist","Neurologist",
                "Endocrinologist","Gastroenterologist","Dermatologist","Hematologist"]:
        st.markdown(f"&nbsp;&nbsp;• {sp}")
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown('<div style="color:#4b5563;font-size:12px">⚠️ For educational/research use only.</div>', unsafe_allow_html=True)

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
        horizontal=True, label_visibility="collapsed"
    )

    medical_report = None
    source_label   = ""

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
            report_files = [f for f in os.listdir(reports_dir) if f.endswith(".txt")] if os.path.isdir(reports_dir) else []
            if report_files:
                chosen = st.selectbox("Select report", ["— select —"] + report_files, label_visibility="collapsed")
                if chosen != "— select —":
                    with open(os.path.join(reports_dir, chosen), "r", encoding="utf-8", errors="ignore") as fh:
                        medical_report = fh.read()
                    source_label = f"File: {chosen}"
                    st.success(f"✅ Loaded: {chosen}")
            else:
                st.info("No reports found in 'Medical Reports/' folder.")

        if medical_report:
            with st.expander("📄 Preview report content"):
                st.text(medical_report[:800] + ("..." if len(medical_report) > 800 else ""))
    else:
        st.markdown("**Patient Information**")
        c1, c2, c3 = st.columns(3)
        name   = c1.text_input("Patient Name", placeholder="e.g. John Smith")
        age    = c2.text_input("Age",           placeholder="e.g. 45")
        gender = c3.selectbox("Gender", ["— select —", "Male", "Female", "Other", "Prefer not to say"])

        st.markdown("**Presenting Symptoms**")
        symptoms = st.text_area(
            "Symptoms",
            placeholder="Describe all symptoms in detail...\ne.g. chest pain, shortness of breath, dizziness, anxiety attacks for 3 months",
            height=130, label_visibility="collapsed"
        )
        col_hist, col_meds = st.columns(2)
        with col_hist:
            st.markdown("**Medical History**")
            history = st.text_area("History", placeholder="e.g. hypertension, diabetes...", height=100, label_visibility="collapsed")
        with col_meds:
            st.markdown("**Current Medications**")
            medications = st.text_area("Medications", placeholder="e.g. Metformin 500mg...", height=100, label_visibility="collapsed")

        if symptoms.strip():
            g = gender if gender != "— select —" else "Not specified"
            medical_report = (
                f"Patient Name: {name or 'Unknown'}\nAge: {age or 'Unknown'}\nGender: {g}\n\n"
                f"Presenting Symptoms:\n{symptoms}\n\n"
                f"Medical History:\n{history or 'None provided.'}\n\n"
                f"Current Medications:\n{medications or 'None provided.'}\n"
            )
            source_label = f"Manual Input — {name or 'Unknown Patient'}"

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    run_clicked = st.button("🚀 Run Diagnosis", disabled=(medical_report is None), use_container_width=True)

    if run_clicked and medical_report:
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.markdown("### ⚙️ Running Analysis...")
        with st.spinner("Agents are analyzing the patient data..."):
            result = run_diagnosis(medical_report, source_label)
        st.session_state.last_result    = result
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
    st.markdown("All past diagnosis sessions saved in the `logs/` folder.")
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
            risk  = log.get("risk", "Unknown")
            specs = ", ".join(log.get("specialists", []))
            rclass = {"High":"card-red","Moderate":"card-yellow","Low":"card-green"}.get(risk,"")

            with st.expander(f"🗓️ {log.get('timestamp','')}  —  {log.get('source','')}  —  Risk: {risk}"):
                col1, col2, col3 = st.columns(3)
                col1.metric("Specialists", len(log.get("specialists",[])))
                col2.metric("Risk Level", risk)
                col3.metric("Model", log.get("model","—"))

                # Confidence scores from history
                sp_reports = log.get("specialist_reports", {})
                if sp_reports and isinstance(list(sp_reports.values())[0], dict):
                    st.markdown('<div class="section-header" style="margin-top:12px">🎯 Confidence Scores</div>', unsafe_allow_html=True)
                    conf_cols = st.columns(4)
                    for i, (sp, data) in enumerate(sp_reports.items()):
                        conf = data.get("confidence","?")
                        with conf_cols[i % 4]:
                            st.markdown(
                                f'<div style="text-align:center;padding:6px">'
                                f'<div style="font-size:12px;color:#9ca3af">{sp}</div>'
                                f'{conf_badge(conf)}'
                                f'</div>',
                                unsafe_allow_html=True
                            )

                st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
                st.markdown(
                    f'<div class="card {rclass}"><div style="color:#e5e7eb;white-space:pre-wrap;font-size:14px">'
                    f'{log.get("final_summary","")}</div></div>',
                    unsafe_allow_html=True
                )

                out_path = log.get("output_path","")
                if out_path and os.path.exists(out_path):
                    with open(out_path, encoding="utf-8") as fh:
                        st.download_button(
                            "⬇️ Download Full Report", data=fh.read(),
                            file_name=f"diagnosis_{log['id']}.txt",
                            mime="text/plain", key=f"dl_{log['id']}"
                        )