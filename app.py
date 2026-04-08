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

st.set_page_config(
    page_title="MediAgent — AI Diagnostic System",
    page_icon="⚕️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700;1,9..40,300&family=DM+Serif+Display:ital@0;1&display=swap');

/* ── Reset & Base ─────────────────────────────────────────────────────────── */
*, *::before, *::after { box-sizing: border-box; }

html, body, .stApp {
    background-color: #F0F4F9 !important;
    font-family: 'DM Sans', sans-serif;
    color: #1a2740;
}

/* ── Hide Streamlit chrome ───────────────────────────────────────────────── */
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }
[data-testid="stToolbar"] { display: none; }

/* ── Sidebar ─────────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: #1E3A5F !important;
    border-right: none !important;
    padding-top: 0 !important;
}
[data-testid="stSidebar"] > div:first-child {
    padding: 0;
}
[data-testid="stSidebar"] * {
    color: #cbd5e1 !important;
}
[data-testid="stSidebar"] .stMarkdown p {
    color: #94a3b8 !important;
    font-size: 13px;
}

/* ── Main content padding ────────────────────────────────────────────────── */
.main .block-container {
    padding: 0 !important;
    max-width: 100% !important;
}

/* ── Page Header ─────────────────────────────────────────────────────────── */
.page-header {
    background: linear-gradient(135deg, #1E3A5F 0%, #1a5276 60%, #1a6ea8 100%);
    padding: 28px 40px 24px;
    border-bottom: 3px solid #0EA5E9;
    position: relative;
    overflow: hidden;
}
.page-header::before {
    content: '';
    position: absolute;
    inset: 0;
    background-image: repeating-linear-gradient(
        0deg, transparent, transparent 24px,
        rgba(255,255,255,0.03) 24px, rgba(255,255,255,0.03) 25px
    ), repeating-linear-gradient(
        90deg, transparent, transparent 24px,
        rgba(255,255,255,0.03) 24px, rgba(255,255,255,0.03) 25px
    );
}
.page-header-inner {
    position: relative;
    display: flex;
    align-items: center;
    justify-content: space-between;
}
.brand {
    display: flex;
    align-items: center;
    gap: 16px;
}
.brand-icon {
    width: 52px; height: 52px;
    background: rgba(255,255,255,0.12);
    border: 1.5px solid rgba(255,255,255,0.25);
    border-radius: 14px;
    display: flex; align-items: center; justify-content: center;
    font-size: 26px;
    backdrop-filter: blur(8px);
}
.brand-name {
    font-family: 'DM Serif Display', serif;
    font-size: 26px;
    color: #ffffff !important;
    letter-spacing: -0.3px;
    line-height: 1;
}
.brand-sub {
    font-size: 12px;
    color: rgba(255,255,255,0.55) !important;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-top: 3px;
}
.header-meta {
    text-align: right;
}
.header-meta-label {
    font-size: 11px;
    color: rgba(255,255,255,0.45) !important;
    letter-spacing: 1.5px;
    text-transform: uppercase;
}
.header-meta-value {
    font-size: 13px;
    color: rgba(255,255,255,0.8) !important;
    margin-top: 2px;
}

/* ── Content wrapper ─────────────────────────────────────────────────────── */
.content-wrap {
    padding: 32px 40px;
}

/* ── Section title ───────────────────────────────────────────────────────── */
.section-title {
    font-family: 'DM Serif Display', serif;
    font-size: 22px;
    color: #1E3A5F;
    margin-bottom: 4px;
    letter-spacing: -0.3px;
}
.section-sub {
    font-size: 13px;
    color: #64748b;
    margin-bottom: 24px;
}

/* ── Clinical cards ──────────────────────────────────────────────────────── */
.clin-card {
    background: #ffffff;
    border: 1px solid #dde5f0;
    border-radius: 12px;
    padding: 24px 28px;
    margin-bottom: 16px;
    box-shadow: 0 1px 3px rgba(30,58,95,0.06), 0 4px 12px rgba(30,58,95,0.04);
    transition: box-shadow 0.2s;
}
.clin-card:hover { box-shadow: 0 2px 8px rgba(30,58,95,0.1), 0 8px 24px rgba(30,58,95,0.06); }
.clin-card-blue   { border-top: 3px solid #1E3A5F; }
.clin-card-teal   { border-top: 3px solid #0EA5E9; }
.clin-card-green  { border-top: 3px solid #10B981; }
.clin-card-amber  { border-top: 3px solid #F59E0B; }
.clin-card-red    { border-top: 3px solid #EF4444; }
.clin-card-purple { border-top: 3px solid #8B5CF6; }

/* ── Card label ──────────────────────────────────────────────────────────── */
.card-label {
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #94a3b8;
    margin-bottom: 10px;
}
.card-body {
    font-size: 14px;
    color: #334155;
    line-height: 1.75;
}

/* ── Divider ─────────────────────────────────────────────────────────────── */
.rule {
    border: none;
    border-top: 1px solid #e2e8f0;
    margin: 20px 0;
}

/* ── Risk badges ─────────────────────────────────────────────────────────── */
.risk-pill {
    display: inline-flex; align-items: center; gap: 6px;
    padding: 5px 14px; border-radius: 999px;
    font-size: 12px; font-weight: 700; letter-spacing: 0.5px;
}
.risk-low    { background: #D1FAE5; color: #065F46; }
.risk-mod    { background: #FEF3C7; color: #92400E; }
.risk-high   { background: #FEE2E2; color: #991B1B; }
.risk-unk    { background: #F1F5F9; color: #64748b; }

/* ── Confidence badges ───────────────────────────────────────────────────── */
.conf-pill {
    display: inline-block;
    padding: 3px 11px; border-radius: 999px;
    font-size: 11px; font-weight: 700; letter-spacing: 0.3px;
}
.conf-high   { background: #D1FAE5; color: #065F46; }
.conf-medium { background: #FEF3C7; color: #92400E; }
.conf-low    { background: #F1F5F9; color: #64748b; }

/* ── Confidence bar ──────────────────────────────────────────────────────── */
.conf-track {
    background: #EFF6FF;
    border-radius: 999px; height: 6px;
    width: 100%; margin-top: 6px;
    overflow: hidden;
}
.conf-fill-high   { background: linear-gradient(90deg,#10B981,#34D399); height:6px; border-radius:999px; }
.conf-fill-medium { background: linear-gradient(90deg,#F59E0B,#FBBF24); height:6px; border-radius:999px; }
.conf-fill-low    { background: #CBD5E1; height:6px; border-radius:999px; }

/* ── Agent status badges ─────────────────────────────────────────────────── */
.agent-badge {
    display: inline-flex; align-items: center; gap: 6px;
    padding: 5px 13px; border-radius: 8px;
    font-size: 12px; font-weight: 600; margin: 4px;
    border: 1px solid transparent;
}
.agent-pending { background:#F8FAFC; border-color:#E2E8F0; color:#94a3b8; }
.agent-running {
    background:#EFF6FF; border-color:#BFDBFE; color:#1d4ed8;
    animation: pulse-border 1.4s ease-in-out infinite;
}
.agent-done    { background:#F0FDF4; border-color:#BBF7D0; color:#15803d; }

@keyframes pulse-border {
    0%,100% { box-shadow: 0 0 0 0 rgba(59,130,246,0); }
    50%      { box-shadow: 0 0 0 3px rgba(59,130,246,0.18); }
}

/* ── Stat boxes ──────────────────────────────────────────────────────────── */
.stat-box {
    background: #ffffff;
    border: 1px solid #dde5f0;
    border-radius: 10px;
    padding: 16px 20px;
    text-align: center;
    box-shadow: 0 1px 3px rgba(30,58,95,0.05);
}
.stat-val {
    font-family: 'DM Serif Display', serif;
    font-size: 28px;
    color: #1E3A5F;
    line-height: 1;
}
.stat-lbl {
    font-size: 11px;
    color: #94a3b8;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin-top: 5px;
}

/* ── Tabs ────────────────────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 2px solid #e2e8f0;
    gap: 0;
    padding: 0;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: #64748b !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    font-size: 14px !important;
    padding: 10px 24px !important;
    border-radius: 0 !important;
    border-bottom: 2px solid transparent !important;
    margin-bottom: -2px !important;
}
.stTabs [aria-selected="true"] {
    color: #1E3A5F !important;
    font-weight: 700 !important;
    border-bottom: 2px solid #1E3A5F !important;
}
.stTabs [data-baseweb="tab-panel"] {
    padding: 0 !important;
}

/* ── Input fields ────────────────────────────────────────────────────────── */
.stTextInput > div > div > input,
.stTextArea  > div > div > textarea,
.stSelectbox > div > div {
    background: #ffffff !important;
    border: 1px solid #dde5f0 !important;
    border-radius: 8px !important;
    color: #1a2740 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 14px !important;
}
.stTextInput > div > div > input:focus,
.stTextArea  > div > div > textarea:focus {
    border-color: #1E3A5F !important;
    box-shadow: 0 0 0 3px rgba(30,58,95,0.08) !important;
}
label, .stTextInput label, .stTextArea label,
.stSelectbox label, .stRadio label {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    color: #475569 !important;
    letter-spacing: 0.2px !important;
}

/* ── Buttons ─────────────────────────────────────────────────────────────── */
.stButton > button {
    background: #1E3A5F !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 9px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    padding: 12px 28px !important;
    letter-spacing: 0.2px !important;
    transition: all 0.18s ease !important;
    box-shadow: 0 2px 6px rgba(30,58,95,0.25) !important;
}
.stButton > button:hover {
    background: #163051 !important;
    box-shadow: 0 4px 14px rgba(30,58,95,0.35) !important;
    transform: translateY(-1px) !important;
}
.stButton > button:disabled {
    background: #cbd5e1 !important;
    box-shadow: none !important;
    transform: none !important;
    cursor: not-allowed !important;
}

/* ── Download button ─────────────────────────────────────────────────────── */
.stDownloadButton > button {
    background: #ffffff !important;
    color: #1E3A5F !important;
    border: 1.5px solid #1E3A5F !important;
    border-radius: 9px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
}
.stDownloadButton > button:hover {
    background: #EFF6FF !important;
}

/* ── File uploader ───────────────────────────────────────────────────────── */
[data-testid="stFileUploader"] {
    background: #F8FAFC !important;
    border: 2px dashed #CBD5E1 !important;
    border-radius: 12px !important;
}

/* ── Metric override ─────────────────────────────────────────────────────── */
[data-testid="metric-container"] {
    background: #ffffff !important;
    border: 1px solid #dde5f0 !important;
    border-radius: 10px !important;
    padding: 14px 18px !important;
}
[data-testid="stMetricValue"] {
    font-family: 'DM Serif Display', serif !important;
    color: #1E3A5F !important;
}
[data-testid="stMetricLabel"] {
    color: #94a3b8 !important;
    font-size: 11px !important;
    letter-spacing: 1px !important;
    text-transform: uppercase !important;
}

/* ── Radio ───────────────────────────────────────────────────────────────── */
.stRadio > div {
    background: #F8FAFC;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 6px 12px;
    gap: 8px;
}

/* ── Expander ────────────────────────────────────────────────────────────── */
[data-testid="stExpander"] {
    background: #ffffff !important;
    border: 1px solid #dde5f0 !important;
    border-radius: 10px !important;
}
[data-testid="stExpander"] summary {
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    color: #1E3A5F !important;
    font-size: 14px !important;
}

/* ── Success / info ──────────────────────────────────────────────────────── */
.stSuccess { background: #F0FDF4 !important; border-color: #BBF7D0 !important; color: #166534 !important; border-radius: 8px !important; }
.stInfo    { background: #EFF6FF !important; border-color: #BFDBFE !important; color: #1E40AF !important; border-radius: 8px !important; }

/* ── Specialist report section ───────────────────────────────────────────── */
.spec-header {
    display: flex; align-items: center; justify-content: space-between;
    margin-bottom: 12px;
}
.spec-name {
    font-weight: 700; font-size: 15px; color: #1E3A5F;
    display: flex; align-items: center; gap: 8px;
}
.spec-icon {
    width: 30px; height: 30px; border-radius: 8px;
    background: #EFF6FF; display: inline-flex;
    align-items: center; justify-content: center;
    font-size: 15px;
}
.spec-reason { font-size: 12px; color: #64748b; margin-top: 4px; font-style: italic; }

/* ── Sidebar nav items ───────────────────────────────────────────────────── */
.nav-header {
    padding: 28px 24px 12px;
    border-bottom: 1px solid rgba(255,255,255,0.08);
    margin-bottom: 8px;
}
.nav-brand {
    font-family: 'DM Serif Display', serif;
    font-size: 20px; color: #ffffff !important;
}
.nav-brand-sub {
    font-size: 10px; color: rgba(255,255,255,0.4) !important;
    letter-spacing: 2px; text-transform: uppercase; margin-top: 2px;
}
.nav-section {
    padding: 16px 24px 8px;
}
.nav-section-title {
    font-size: 10px !important;
    font-weight: 700 !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
    color: rgba(255,255,255,0.3) !important;
    margin-bottom: 10px !important;
}
.nav-item {
    display: flex; align-items: center; gap: 10px;
    padding: 8px 12px; border-radius: 8px; margin-bottom: 2px;
    font-size: 13px; color: rgba(255,255,255,0.7) !important;
    cursor: pointer; transition: background 0.15s;
}
.nav-item:hover { background: rgba(255,255,255,0.06); }
.nav-item-dot {
    width: 6px; height: 6px; border-radius: 50%;
    background: #0EA5E9; flex-shrink: 0;
}
.nav-divider { border-top: 1px solid rgba(255,255,255,0.07); margin: 12px 24px; }
.nav-disclaimer {
    padding: 16px 24px;
    font-size: 11px !important;
    color: rgba(255,255,255,0.25) !important;
    line-height: 1.5 !important;
}

/* ── Progress / spinner ──────────────────────────────────────────────────── */
.stSpinner > div { border-top-color: #1E3A5F !important; }

/* ── Selectbox ───────────────────────────────────────────────────────────── */
.stSelectbox [data-baseweb="select"] > div {
    background: #ffffff !important;
    border-color: #dde5f0 !important;
}
</style>
""", unsafe_allow_html=True)

# ── Session State ─────────────────────────────────────────────────────────────
if "diagnosis_done" not in st.session_state: st.session_state.diagnosis_done = False
if "last_result"    not in st.session_state: st.session_state.last_result    = None

# ── Helpers ───────────────────────────────────────────────────────────────────
LOGS_DIR = "logs"
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs("results", exist_ok=True)

SPECIALIST_ICONS = {
    "Cardiologist": "🫀", "Psychologist": "🧠", "Pulmonologist": "🫁",
    "Neurologist": "⚡", "Endocrinologist": "⚗️", "Gastroenterologist": "🔬",
    "Dermatologist": "🩺", "Hematologist": "🩸",
}

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

def risk_pill(risk: str) -> str:
    dot = {"High":"🔴","Moderate":"🟡","Low":"🟢","Unknown":"⚪"}.get(risk,"⚪")
    cls = {"High":"risk-high","Moderate":"risk-mod","Low":"risk-low"}.get(risk,"risk-unk")
    return f'<span class="risk-pill {cls}">{dot} {risk} Risk</span>'

def conf_pill(conf: str) -> str:
    cls = {"High":"conf-high","Medium":"conf-medium","Low":"conf-low"}.get(conf,"conf-low")
    return f'<span class="conf-pill {cls}">{conf}</span>'

def conf_bar_html(conf: str) -> str:
    w   = {"High":"100%","Medium":"62%","Low":"28%"}.get(conf,"28%")
    cls = {"High":"conf-fill-high","Medium":"conf-fill-medium","Low":"conf-fill-low"}.get(conf,"conf-fill-low")
    return f'<div class="conf-track"><div class="{cls}" style="width:{w}"></div></div>'

def run_diagnosis(medical_report: str, source_label: str):
    selected  = select_specialists(medical_report)
    agents    = {name: SPECIALIST_REGISTRY[name](medical_report) for name in selected}
    responses = {}

    status_ph = st.empty()

    def update_status(done_list, running_name, pending_list):
        html = '<div style="display:flex;flex-wrap:wrap;gap:4px;padding:4px 0">'
        for s in done_list:
            icon = SPECIALIST_ICONS.get(s, "🔬")
            html += f'<span class="agent-badge agent-done">{icon} {s} ✓</span>'
        if running_name:
            icon = SPECIALIST_ICONS.get(running_name, "🔬")
            html += f'<span class="agent-badge agent-running">⚙️ {running_name}</span>'
        for s in pending_list:
            icon = SPECIALIST_ICONS.get(s, "🔬")
            html += f'<span class="agent-badge agent-pending">{icon} {s}</span>'
        html += '</div>'
        status_ph.markdown(
            f'<div class="clin-card" style="padding:16px 20px">'
            f'<div class="card-label">Specialist Panel — Running in Parallel</div>'
            f'{html}</div>',
            unsafe_allow_html=True
        )

    done, pending = [], list(selected)
    update_status([], None, pending)

    def get_response(agent_name, agent):
        return agent_name, agent.run()

    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(get_response, name, agent): name for name, agent in agents.items()}
        for future in as_completed(futures):
            agent_name, response = future.result()
            responses[agent_name] = response
            done.append(agent_name)
            if agent_name in pending: pending.remove(agent_name)
            update_status(done, None, pending)

    status_ph.markdown(
        '<div class="clin-card clin-card-teal" style="padding:16px 20px">'
        '<div class="card-label">Multidisciplinary Team</div>'
        '<div style="color:#1E3A5F;font-weight:600;font-size:14px">'
        '🧠 &nbsp;Lead physician synthesizing all specialist findings...</div></div>',
        unsafe_allow_html=True
    )

    team_agent    = MultidisciplinaryTeam(specialist_reports=responses)
    final_summary = team_agent.run()
    status_ph.empty()

    risk   = parse_risk(final_summary)
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

    output_path = f"results/{run_id}_diagnosis.txt"
    full_text   = f"Source: {source_label}\nSpecialists: {', '.join(selected)}\n\n"
    for sp, data in responses.items():
        conf   = data.get("confidence","?")
        reason = data.get("confidence_reason","")
        report = data.get("report","")
        full_text += f"[{sp}] — Confidence: {conf} ({reason})\n{report}\n\n"
    full_text += f"{'='*65}\nFINAL SUMMARY\n{'='*65}\n{final_summary}\n"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(full_text)

    result["output_path"] = output_path
    save_log(result)
    return result

def render_confidence_panel(sp_data: dict):
    cols = st.columns(2)
    for i, (sp, data) in enumerate(sp_data.items()):
        conf   = data.get("confidence","Medium")
        reason = data.get("confidence_reason","")
        icon   = SPECIALIST_ICONS.get(sp,"🔬")
        with cols[i % 2]:
            st.markdown(
                f'<div class="clin-card" style="padding:16px 20px;margin-bottom:10px">'
                f'<div class="spec-header">'
                f'<div class="spec-name"><span class="spec-icon">{icon}</span>{sp}</div>'
                f'{conf_pill(conf)}'
                f'</div>'
                f'{conf_bar_html(conf)}'
                f'<div class="spec-reason">{reason}</div>'
                f'</div>',
                unsafe_allow_html=True
            )

def render_summary(result: dict):
    summary = result["final_summary"]
    risk    = result["risk"]
    specs   = result["specialists"]
    ts      = result["timestamp"]
    sp_data = result.get("specialist_reports", {})

    # ── Top bar ───────────────────────────────────────────────────────────────
    st.markdown(
        f'<div style="display:flex;align-items:center;justify-content:space-between;'
        f'background:#ffffff;border:1px solid #dde5f0;border-radius:12px;'
        f'padding:16px 24px;margin-bottom:20px;box-shadow:0 1px 4px rgba(30,58,95,0.06)">'
        f'<div>'
        f'<div style="font-family:\'DM Serif Display\',serif;font-size:18px;color:#1E3A5F">'
        f'Diagnosis Report</div>'
        f'<div style="font-size:12px;color:#94a3b8;margin-top:2px">{result["source"]}</div>'
        f'</div>'
        f'<div style="display:flex;align-items:center;gap:16px">'
        f'<div style="text-align:right">'
        f'<div style="font-size:11px;color:#94a3b8;letter-spacing:1px;text-transform:uppercase">Generated</div>'
        f'<div style="font-size:13px;color:#475569;margin-top:1px">{ts}</div>'
        f'</div>'
        f'{risk_pill(risk)}'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True
    )

    # ── Stat row ──────────────────────────────────────────────────────────────
    s1, s2, s3, s4 = st.columns(4)
    with s1:
        st.markdown(f'<div class="stat-box"><div class="stat-val">{len(specs)}</div><div class="stat-lbl">Specialists</div></div>', unsafe_allow_html=True)
    with s2:
        high_conf = sum(1 for d in sp_data.values() if isinstance(d,dict) and d.get("confidence")=="High")
        st.markdown(f'<div class="stat-box"><div class="stat-val">{high_conf}</div><div class="stat-lbl">High Confidence</div></div>', unsafe_allow_html=True)
    with s3:
        risk_color = {"High":"#EF4444","Moderate":"#F59E0B","Low":"#10B981"}.get(risk,"#94a3b8")
        st.markdown(f'<div class="stat-box"><div class="stat-val" style="color:{risk_color}">{risk}</div><div class="stat-lbl">Risk Level</div></div>', unsafe_allow_html=True)
    with s4:
        st.markdown(f'<div class="stat-box"><div class="stat-val" style="font-size:16px;padding-top:6px">{result.get("model",OLLAMA_MODEL)}</div><div class="stat-lbl">AI Model</div></div>', unsafe_allow_html=True)

    st.markdown('<hr class="rule">', unsafe_allow_html=True)

    # ── Confidence panel ──────────────────────────────────────────────────────
    if sp_data:
        st.markdown('<div class="card-label" style="margin-bottom:12px">🎯 &nbsp;SPECIALIST CONFIDENCE SCORES</div>', unsafe_allow_html=True)
        render_confidence_panel(sp_data)
        st.markdown('<hr class="rule">', unsafe_allow_html=True)

    # ── Summary sections ──────────────────────────────────────────────────────
    SECTIONS = {
        "PATIENT SUMMARY":        ("clin-card-blue",   "👤", "PATIENT SUMMARY"),
        "TOP 3 DIAGNOSES":        ("clin-card-purple",  "🔬", "TOP 3 DIAGNOSES"),
        "KEY FINDINGS":           ("clin-card-amber",   "📋", "KEY FINDINGS"),
        "RECOMMENDED NEXT STEPS": ("clin-card-green",   "✅", "RECOMMENDED NEXT STEPS"),
        "OVERALL RISK LEVEL":     ("clin-card-red",     "⚠️", "OVERALL RISK LEVEL"),
    }

    lines = summary.split("\n")
    cur_section, cur_lines, rendered = None, [], {}

    def flush(sec, lns, card_cls, icon, label):
        content = "\n".join(l for l in lns if l.strip() and "---" not in l).strip()
        if content:
            st.markdown(
                f'<div class="clin-card {card_cls}">'
                f'<div class="card-label">{icon} &nbsp;{label}</div>'
                f'<div class="card-body">{content.replace(chr(10),"<br>")}</div>'
                f'</div>',
                unsafe_allow_html=True
            )
        rendered[sec] = True

    for line in lines:
        matched = next((k for k in SECTIONS if k in line.upper()), None)
        if matched:
            if cur_section and cur_section in SECTIONS:
                flush(cur_section, cur_lines, *SECTIONS[cur_section])
            cur_section, cur_lines = matched, []
        elif cur_section:
            cur_lines.append(line)

    if cur_section and cur_section in SECTIONS and cur_section not in rendered:
        flush(cur_section, cur_lines, *SECTIONS[cur_section])

    if not rendered:
        st.markdown(
            f'<div class="clin-card"><div class="card-body" style="white-space:pre-wrap">{summary}</div></div>',
            unsafe_allow_html=True
        )

    # ── Specialist drilldown ──────────────────────────────────────────────────
    if sp_data:
        st.markdown('<hr class="rule">', unsafe_allow_html=True)
        st.markdown('<div class="card-label" style="margin-bottom:12px">🩺 &nbsp;SPECIALIST REPORTS — FULL DETAIL</div>', unsafe_allow_html=True)
        for sp, data in sp_data.items():
            icon   = SPECIALIST_ICONS.get(sp,"🔬")
            conf   = data.get("confidence","?")
            report = data.get("report","")
            with st.expander(f"{icon}  {sp}  —  Confidence: {conf}"):
                st.markdown(
                    f'<div style="font-size:13px;color:#334155;line-height:1.8;white-space:pre-wrap">{report}</div>',
                    unsafe_allow_html=True
                )

    # ── Download ──────────────────────────────────────────────────────────────
    st.markdown('<hr class="rule">', unsafe_allow_html=True)
    out_path = result.get("output_path","")
    if out_path and os.path.exists(out_path):
        st.download_button(
            label="⬇️  Download Full Report (.txt)",
            data=open(out_path, encoding="utf-8").read(),
            file_name=f"diagnosis_{result['id']}.txt",
            mime="text/plain",
            use_container_width=True
        )

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        '<div class="nav-header">'
        '<div class="nav-brand">MediAgent</div>'
        '<div class="nav-brand-sub">AI Diagnostic System</div>'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="nav-section">'
        '<div class="nav-section-title">Active Model</div>'
        f'<div style="background:rgba(14,165,233,0.12);border:1px solid rgba(14,165,233,0.25);'
        f'border-radius:8px;padding:10px 14px;font-size:13px;color:#7dd3fc !important;">'
        f'<span style="color:rgba(255,255,255,0.4);font-size:11px">OLLAMA /</span><br>'
        f'<span style="font-weight:600;color:#ffffff !important">{OLLAMA_MODEL}</span>'
        f'</div>'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown('<div class="nav-divider"></div>', unsafe_allow_html=True)

    st.markdown(
        '<div class="nav-section">'
        '<div class="nav-section-title">Specialist Panel</div>',
        unsafe_allow_html=True
    )
    for sp, icon in SPECIALIST_ICONS.items():
        st.markdown(
            f'<div class="nav-item"><span class="nav-item-dot"></span>{icon} &nbsp;{sp}</div>',
            unsafe_allow_html=True
        )
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="nav-divider"></div>', unsafe_allow_html=True)

    st.markdown(
        '<div class="nav-section">'
        '<div class="nav-section-title">Confidence Scale</div>'
        f'<div style="margin-bottom:8px">{conf_pill("High")} &nbsp;<span style="font-size:12px;color:rgba(255,255,255,0.5)">3+ direct indicators</span></div>'
        f'<div style="margin-bottom:8px">{conf_pill("Medium")} &nbsp;<span style="font-size:12px;color:rgba(255,255,255,0.5)">1–2 indirect signs</span></div>'
        f'<div>{conf_pill("Low")} &nbsp;<span style="font-size:12px;color:rgba(255,255,255,0.5)">Precautionary only</span></div>'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown('<div class="nav-divider"></div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="nav-disclaimer">'
        '⚠️ For educational and research use only. Not a substitute for professional medical advice or clinical diagnosis.'
        '</div>',
        unsafe_allow_html=True
    )

# ── Page Header ───────────────────────────────────────────────────────────────
st.markdown(
    f'<div class="page-header">'
    f'<div class="page-header-inner">'
    f'<div class="brand">'
    f'<div class="brand-icon">⚕️</div>'
    f'<div>'
    f'<div class="brand-name">MediAgent</div>'
    f'<div class="brand-sub">AI-Powered Clinical Diagnosis System</div>'
    f'</div>'
    f'</div>'
    f'<div class="header-meta">'
    f'<div class="header-meta-label">Session Date</div>'
    f'<div class="header-meta-value">{datetime.now().strftime("%d %B %Y")}</div>'
    f'</div>'
    f'</div>'
    f'</div>',
    unsafe_allow_html=True
)

# ── Main Content ──────────────────────────────────────────────────────────────
st.markdown('<div class="content-wrap">', unsafe_allow_html=True)

tab1, tab2 = st.tabs(["⚕️  New Diagnosis", "📂  Patient History"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — DIAGNOSE
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div style="height:24px"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">New Diagnosis</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Provide patient data via a report file or enter symptoms manually.</div>', unsafe_allow_html=True)

    input_mode = st.radio(
        "input_mode",
        ["📄  Report File", "✏️  Manual Symptom Entry"],
        horizontal=True,
        label_visibility="collapsed"
    )

    medical_report = None
    source_label   = ""

    # ── File mode ─────────────────────────────────────────────────────────────
    if "File" in input_mode:
        st.markdown('<div class="clin-card" style="margin-top:8px">', unsafe_allow_html=True)
        st.markdown('<div class="card-label">📁 &nbsp;PATIENT REPORT FILE</div>', unsafe_allow_html=True)

        col_up, col_sel = st.columns(2)
        with col_up:
            st.markdown("**Upload file**")
            uploaded = st.file_uploader("Upload .txt", type=["txt"], label_visibility="collapsed")
            if uploaded:
                medical_report = uploaded.read().decode("utf-8", errors="ignore")
                source_label   = f"Uploaded: {uploaded.name}"
                st.success(f"✅  {uploaded.name} loaded successfully")

        with col_sel:
            st.markdown("**Or select existing report**")
            reports_dir  = "Medical Reports"
            report_files = sorted([f for f in os.listdir(reports_dir) if f.endswith(".txt")]) if os.path.isdir(reports_dir) else []
            if report_files:
                chosen = st.selectbox("Report", ["— select a report —"] + report_files, label_visibility="collapsed")
                if chosen != "— select a report —":
                    with open(os.path.join(reports_dir, chosen), "r", encoding="utf-8", errors="ignore") as fh:
                        medical_report = fh.read()
                    source_label = f"File: {chosen}"
                    st.success(f"✅  {chosen} loaded")
            else:
                st.info("No reports found in 'Medical Reports/' folder.")

        if medical_report:
            with st.expander("Preview report content"):
                st.code(medical_report[:900] + ("…" if len(medical_report) > 900 else ""), language=None)
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Manual mode ───────────────────────────────────────────────────────────
    else:
        st.markdown('<div class="clin-card" style="margin-top:8px">', unsafe_allow_html=True)
        st.markdown('<div class="card-label">👤 &nbsp;PATIENT DEMOGRAPHICS</div>', unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)
        name   = c1.text_input("Patient Name",  placeholder="Full name")
        age    = c2.text_input("Age",            placeholder="e.g. 42")
        gender = c3.selectbox("Gender", ["— select —", "Male", "Female", "Other", "Prefer not to say"])

        st.markdown('<hr class="rule">', unsafe_allow_html=True)
        st.markdown('<div class="card-label">🩺 &nbsp;PRESENTING SYMPTOMS</div>', unsafe_allow_html=True)
        symptoms = st.text_area(
            "Symptoms",
            placeholder="Describe all symptoms in detail, including onset, duration, and severity.\nE.g. persistent chest pain (3 weeks, moderate), shortness of breath on exertion, dizziness, anxiety episodes...",
            height=120,
            label_visibility="collapsed"
        )

        st.markdown('<hr class="rule">', unsafe_allow_html=True)
        col_h, col_m = st.columns(2)
        with col_h:
            st.markdown('<div class="card-label">📋 &nbsp;MEDICAL HISTORY</div>', unsafe_allow_html=True)
            history = st.text_area("History", placeholder="Known conditions, past surgeries, family history...", height=90, label_visibility="collapsed")
        with col_m:
            st.markdown('<div class="card-label">💊 &nbsp;CURRENT MEDICATIONS</div>', unsafe_allow_html=True)
            medications = st.text_area("Medications", placeholder="Drug name, dosage, frequency...", height=90, label_visibility="collapsed")

        st.markdown('</div>', unsafe_allow_html=True)

        if symptoms.strip():
            g = gender if gender != "— select —" else "Not specified"
            medical_report = (
                f"Patient Name: {name or 'Unknown'}\nAge: {age or 'Unknown'}\nGender: {g}\n\n"
                f"Presenting Symptoms:\n{symptoms}\n\n"
                f"Medical History:\n{history or 'None provided.'}\n\n"
                f"Current Medications:\n{medications or 'None provided.'}\n"
            )
            source_label = f"Manual Input — {name or 'Unknown Patient'}"

    # ── Run button ────────────────────────────────────────────────────────────
    st.markdown('<div style="height:8px"></div>', unsafe_allow_html=True)
    run_clicked = st.button(
        "⚕️  Run Diagnostic Analysis",
        disabled=(medical_report is None),
        use_container_width=True
    )

    if run_clicked and medical_report:
        st.markdown('<hr class="rule">', unsafe_allow_html=True)
        st.markdown(
            '<div class="clin-card clin-card-blue" style="padding:16px 20px;margin-bottom:16px">'
            '<div class="card-label">⚙️ &nbsp;ANALYSIS IN PROGRESS</div>'
            '<div style="font-size:14px;color:#1E3A5F">Specialist agents are reviewing the patient data in parallel. '
            'This may take 1–3 minutes depending on your hardware.</div>'
            '</div>',
            unsafe_allow_html=True
        )
        with st.spinner("Running multi-agent diagnostic analysis..."):
            result = run_diagnosis(medical_report, source_label)

        st.session_state.last_result    = result
        st.session_state.diagnosis_done = True

        st.markdown('<hr class="rule">', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-title" style="margin-bottom:4px">Diagnosis Complete</div>'
            '<div class="section-sub">All specialists have submitted their findings. Review the summary below.</div>',
            unsafe_allow_html=True
        )
        render_summary(result)

    elif st.session_state.diagnosis_done and st.session_state.last_result:
        st.markdown('<hr class="rule">', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-title" style="margin-bottom:4px">Last Diagnosis</div>'
            '<div class="section-sub">Results from your most recent diagnostic session.</div>',
            unsafe_allow_html=True
        )
        render_summary(st.session_state.last_result)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — HISTORY
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div style="height:24px"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Patient History</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">All past diagnostic sessions are stored locally in the <code>logs/</code> folder.</div>', unsafe_allow_html=True)

    logs = load_logs()
    if not logs:
        st.markdown(
            '<div class="clin-card" style="text-align:center;padding:48px 24px">'
            '<div style="font-size:32px;margin-bottom:12px">📭</div>'
            '<div style="font-family:\'DM Serif Display\',serif;font-size:18px;color:#1E3A5F;margin-bottom:6px">No records yet</div>'
            '<div style="font-size:13px;color:#94a3b8">Run your first diagnostic session to see results here.</div>'
            '</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f'<div style="font-size:13px;color:#64748b;margin-bottom:16px">'
            f'<strong style="color:#1E3A5F">{len(logs)}</strong> session(s) on record</div>',
            unsafe_allow_html=True
        )
        for log in logs:
            risk    = log.get("risk","Unknown")
            source  = log.get("source","")
            ts      = log.get("timestamp","")
            specs   = log.get("specialists",[])
            rcolor  = {"High":"#EF4444","Moderate":"#F59E0B","Low":"#10B981"}.get(risk,"#94a3b8")
            sp_rpts = log.get("specialist_reports",{})

            with st.expander(f"🗓️  {ts}   ·   {source}   ·   {risk} Risk"):
                # Stat row
                h1, h2, h3, h4 = st.columns(4)
                h1.metric("Specialists", len(specs))
                h2.metric("Risk Level",  risk)
                h3.metric("Model",       log.get("model","—"))
                high_c = sum(1 for d in sp_rpts.values() if isinstance(d,dict) and d.get("confidence")=="High")
                h4.metric("High Confidence", high_c)

                # Confidence badges
                if sp_rpts and isinstance(list(sp_rpts.values())[0], dict):
                    st.markdown('<div class="card-label" style="margin-top:16px;margin-bottom:8px">🎯 CONFIDENCE SCORES</div>', unsafe_allow_html=True)
                    badge_html = "".join(
                        f'<span style="display:inline-flex;align-items:center;gap:6px;'
                        f'margin:3px;padding:5px 12px;background:#F8FAFC;border:1px solid #e2e8f0;'
                        f'border-radius:8px;font-size:12px;color:#475569">'
                        f'{SPECIALIST_ICONS.get(sp,"🔬")} {sp} &nbsp;{conf_pill(data.get("confidence","?"))}'
                        f'</span>'
                        for sp, data in sp_rpts.items()
                        if isinstance(data, dict)
                    )
                    st.markdown(badge_html, unsafe_allow_html=True)

                st.markdown('<hr class="rule">', unsafe_allow_html=True)

                # Summary
                summary_text = log.get("final_summary","")
                st.markdown(
                    f'<div class="clin-card" style="border-top:3px solid {rcolor}">'
                    f'<div class="card-label">FINAL SUMMARY</div>'
                    f'<div class="card-body" style="white-space:pre-wrap">{summary_text}</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )

                out_path = log.get("output_path","")
                if out_path and os.path.exists(out_path):
                    with open(out_path, encoding="utf-8") as fh:
                        st.download_button(
                            "⬇️  Download Full Report",
                            data=fh.read(),
                            file_name=f"diagnosis_{log['id']}.txt",
                            mime="text/plain",
                            key=f"dl_{log['id']}"
                        )

st.markdown('</div>', unsafe_allow_html=True)