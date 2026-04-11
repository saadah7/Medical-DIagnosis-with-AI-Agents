import streamlit as st
import os, json, re
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeoutError
from Utils.agents import select_specialists, SPECIALIST_REGISTRY, MultidisciplinaryTeam, OLLAMA_MODEL, check_ollama

st.set_page_config(page_title="MediAgent", page_icon="⚕️", layout="wide", initial_sidebar_state="expanded")

try:
    check_ollama()
except RuntimeError as _ollama_err:
    st.error(str(_ollama_err))
    st.stop()

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Serif+Display&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; }
html, body, .stApp { background: #EEF2F7 !important; font-family: 'DM Sans', sans-serif; color: #1a2740; }
#MainMenu, footer, header, .stDeployButton, [data-testid="stToolbar"] { display: none !important; visibility: hidden !important; }
.main .block-container { padding: 0 !important; max-width: 100% !important; }

/* Sidebar */
[data-testid="stSidebar"] { background: #112240 !important; border-right: none !important; }
[data-testid="stSidebar"] * { color: #8892b0 !important; font-family: 'DM Sans', sans-serif !important; }

/* Page header */
.ph { background: linear-gradient(135deg,#112240 0%,#1a3a6b 100%); padding: 22px 40px; border-bottom: 2px solid #0EA5E9; position: relative; overflow: hidden; }
.ph::after { content:''; position:absolute; inset:0; background: repeating-linear-gradient(90deg,transparent,transparent 60px,rgba(255,255,255,.02) 60px,rgba(255,255,255,.02) 61px),repeating-linear-gradient(0deg,transparent,transparent 60px,rgba(255,255,255,.02) 60px,rgba(255,255,255,.02) 61px); }
.ph-inner { position:relative; display:flex; align-items:center; justify-content:space-between; }
.ph-brand { display:flex; align-items:center; gap:14px; }
.ph-icon { width:46px;height:46px; background:rgba(14,165,233,.15); border:1px solid rgba(14,165,233,.3); border-radius:12px; display:flex; align-items:center; justify-content:center; font-size:22px; }
.ph-title { font-family:'DM Serif Display',serif; font-size:24px; color:#fff; line-height:1; }
.ph-sub { font-size:11px; color:rgba(255,255,255,.4); letter-spacing:2px; text-transform:uppercase; margin-top:3px; }
.ph-date { font-size:12px; color:rgba(255,255,255,.45); text-align:right; letter-spacing:.5px; }

/* Content */
.wrap { padding: 28px 36px; }

/* Section heading */
.sh { font-family:'DM Serif Display',serif; font-size:20px; color:#112240; margin-bottom:2px; }
.ss { font-size:13px; color:#64748b; margin-bottom:20px; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] { background:transparent !important; border-bottom:2px solid #dde4ef; gap:0; padding:0; }
.stTabs [data-baseweb="tab"] { background:transparent !important; color:#64748b !important; font-family:'DM Sans',sans-serif !important; font-weight:500 !important; font-size:14px !important; padding:10px 22px !important; border-radius:0 !important; border-bottom:2px solid transparent !important; margin-bottom:-2px !important; }
.stTabs [aria-selected="true"] { color:#112240 !important; font-weight:700 !important; border-bottom:2px solid #112240 !important; }
.stTabs [data-baseweb="tab-panel"] { padding:0 !important; }

/* Cards */
.card { background:#fff; border:1px solid #dde4ef; border-radius:12px; padding:20px 24px; margin-bottom:14px; box-shadow:0 1px 3px rgba(17,34,64,.05); }
.card-t-navy  { border-top:3px solid #112240; }
.card-t-sky   { border-top:3px solid #0EA5E9; }
.card-t-green { border-top:3px solid #10B981; }
.card-t-amber { border-top:3px solid #F59E0B; }
.card-t-red   { border-top:3px solid #EF4444; }
.card-t-violet{ border-top:3px solid #7C3AED; }

/* Card label */
.cl { font-size:10px; font-weight:700; letter-spacing:2px; text-transform:uppercase; color:#94a3b8; margin-bottom:10px; }

/* Horizontal rule */
.hr { border:none; border-top:1px solid #e8edf5; margin:16px 0; }

/* Risk pill */
.rp { display:inline-flex; align-items:center; gap:5px; padding:5px 13px; border-radius:999px; font-size:12px; font-weight:700; }
.rp-l { background:#D1FAE5; color:#065F46; }
.rp-m { background:#FEF3C7; color:#92400E; }
.rp-h { background:#FEE2E2; color:#991B1B; }
.rp-u { background:#F1F5F9; color:#64748b; }

/* Confidence pill */
.cp { display:inline-block; padding:2px 10px; border-radius:999px; font-size:11px; font-weight:700; }
.cp-h { background:#D1FAE5; color:#065F46; }
.cp-m { background:#FEF3C7; color:#92400E; }
.cp-l { background:#F1F5F9; color:#64748b; }

/* Confidence bar */
.ctr { background:#EFF4FF; border-radius:999px; height:5px; width:100%; margin-top:5px; overflow:hidden; }
.cf-h { background:linear-gradient(90deg,#10B981,#34D399); height:5px; border-radius:999px; }
.cf-m { background:linear-gradient(90deg,#F59E0B,#FBBF24); height:5px; border-radius:999px; }
.cf-l { background:#CBD5E1; height:5px; border-radius:999px; }

/* Agent badges */
.ab { display:inline-flex; align-items:center; gap:5px; padding:5px 11px; border-radius:7px; font-size:12px; font-weight:600; margin:3px; border:1px solid transparent; }
.ab-p { background:#F8FAFC; border-color:#E2E8F0; color:#94a3b8; }
.ab-r { background:#EFF6FF; border-color:#BFDBFE; color:#1d4ed8; animation:pulse 1.4s infinite; }
.ab-d { background:#F0FDF4; border-color:#BBF7D0; color:#15803d; }
@keyframes pulse { 0%,100%{box-shadow:0 0 0 0 rgba(59,130,246,0)} 50%{box-shadow:0 0 0 3px rgba(59,130,246,.15)} }

/* Stat box */
.sb { background:#fff; border:1px solid #dde4ef; border-radius:10px; padding:14px 18px; text-align:center; }
.sv { font-family:'DM Serif Display',serif; font-size:26px; color:#112240; line-height:1; }
.sl { font-size:10px; color:#94a3b8; letter-spacing:1.5px; text-transform:uppercase; margin-top:4px; }

/* Diagnosis block */
.dx-block { background:#F8FAFC; border:1px solid #e2e8f0; border-left:3px solid #0EA5E9; border-radius:8px; padding:14px 18px; margin-bottom:10px; }
.dx-title { font-size:14px; font-weight:700; color:#112240; margin-bottom:6px; }
.dx-row { display:flex; gap:8px; margin-bottom:4px; align-items:flex-start; }
.dx-key { font-size:11px; font-weight:700; letter-spacing:1px; text-transform:uppercase; color:#94a3b8; min-width:80px; padding-top:1px; flex-shrink:0; }
.dx-val { font-size:13px; color:#334155; line-height:1.5; }
.dx-for  { color:#065F46; }
.dx-ag   { color:#92400E; }

/* Finding row */
.fi { display:flex; align-items:flex-start; gap:10px; padding:7px 0; border-bottom:1px solid #f1f5f9; font-size:13px; color:#334155; line-height:1.5; }
.fi:last-child { border-bottom:none; }
.fi-num { width:22px; height:22px; border-radius:50%; background:#EFF6FF; color:#1E40AF; font-size:11px; font-weight:700; display:flex; align-items:center; justify-content:center; flex-shrink:0; margin-top:1px; }
.fi-dot { width:6px; height:6px; border-radius:50%; background:#0EA5E9; flex-shrink:0; margin-top:6px; }

/* Next step row */
.ns { display:flex; align-items:flex-start; gap:10px; padding:7px 0; border-bottom:1px solid #f1f5f9; font-size:13px; color:#334155; line-height:1.5; }
.ns:last-child { border-bottom:none; }
.ns-num { width:22px; height:22px; border-radius:6px; background:#112240; color:#fff; font-size:11px; font-weight:700; display:flex; align-items:center; justify-content:center; flex-shrink:0; margin-top:1px; }

/* Specialist detail */
.sp-section { margin-bottom:10px; }
.sp-section-title { font-size:11px; font-weight:700; letter-spacing:1.5px; text-transform:uppercase; color:#0EA5E9; margin-bottom:6px; padding-bottom:4px; border-bottom:1px solid #EFF6FF; }
.sp-text { font-size:13px; color:#334155; line-height:1.65; white-space:pre-wrap; }

/* Buttons */
.stButton > button { background:#112240 !important; color:#fff !important; border:none !important; border-radius:9px !important; font-family:'DM Sans',sans-serif !important; font-weight:600 !important; font-size:14px !important; padding:11px 24px !important; box-shadow:0 2px 6px rgba(17,34,64,.2) !important; transition:all .15s !important; }
.stButton > button:hover { background:#0d1b33 !important; box-shadow:0 4px 12px rgba(17,34,64,.3) !important; transform:translateY(-1px) !important; }
.stButton > button:disabled { background:#cbd5e1 !important; box-shadow:none !important; transform:none !important; }
.stDownloadButton > button { background:#fff !important; color:#112240 !important; border:1.5px solid #112240 !important; border-radius:9px !important; font-weight:600 !important; }
.stDownloadButton > button:hover { background:#EFF6FF !important; }

/* Inputs */
.stTextInput > div > div > input, .stTextArea > div > div > textarea { background:#fff !important; border:1px solid #dde4ef !important; border-radius:8px !important; color:#1a2740 !important; font-family:'DM Sans',sans-serif !important; font-size:14px !important; }
.stTextInput > div > div > input:focus, .stTextArea > div > div > textarea:focus { border-color:#112240 !important; box-shadow:0 0 0 3px rgba(17,34,64,.07) !important; }
label, .stTextInput label, .stTextArea label, .stSelectbox label { font-family:'DM Sans',sans-serif !important; font-size:12px !important; font-weight:600 !important; color:#475569 !important; letter-spacing:.3px !important; text-transform:uppercase !important; }
.stSelectbox [data-baseweb="select"] > div { background:#fff !important; border-color:#dde4ef !important; }
.stRadio > div { background:#F8FAFC; border:1px solid #e8edf5; border-radius:9px; padding:5px 10px; }
[data-testid="stRadioGroup"] label { color:#1a2740 !important; font-weight:600 !important; font-size:13px !important; }
[data-testid="stRadioGroup"] label p { color:#1a2740 !important; }
[data-testid="stFileUploader"] { background:#F8FAFC !important; border:2px dashed #CBD5E1 !important; border-radius:10px !important; }
[data-testid="metric-container"] { background:#fff !important; border:1px solid #dde4ef !important; border-radius:10px !important; padding:14px 18px !important; }
[data-testid="stMetricValue"] { font-family:'DM Serif Display',serif !important; color:#112240 !important; }
[data-testid="stMetricLabel"] { color:#94a3b8 !important; font-size:10px !important; letter-spacing:1px !important; text-transform:uppercase !important; }
[data-testid="stExpander"] { background:#fff !important; border:1px solid #dde4ef !important; border-radius:10px !important; }
[data-testid="stExpander"] summary { font-family:'DM Sans',sans-serif !important; font-weight:600 !important; color:#112240 !important; font-size:14px !important; }
.stSuccess { background:#F0FDF4 !important; border-color:#BBF7D0 !important; color:#166534 !important; border-radius:8px !important; }
.stInfo { background:#EFF6FF !important; border-color:#BFDBFE !important; color:#1E40AF !important; border-radius:8px !important; }

/* Sidebar nav */
.nb { padding:24px 20px 14px; border-bottom:1px solid rgba(255,255,255,.07); }
.nt { font-family:'DM Serif Display',serif; font-size:19px; color:#ccd6f6 !important; }
.ns2 { font-size:10px; color:rgba(255,255,255,.3) !important; letter-spacing:2px; text-transform:uppercase; margin-top:2px; }
.nsec { padding:14px 20px 6px; }
.nst { font-size:10px !important; font-weight:700 !important; letter-spacing:2px !important; text-transform:uppercase !important; color:rgba(255,255,255,.25) !important; margin-bottom:8px !important; }
.ni { display:flex; align-items:center; gap:9px; padding:7px 10px; border-radius:7px; margin-bottom:2px; font-size:13px; color:rgba(255,255,255,.6) !important; }
.ni:hover { background:rgba(255,255,255,.05); }
.nd { border-top:1px solid rgba(255,255,255,.06); margin:10px 20px; }
.ndis { padding:12px 20px; font-size:11px !important; color:rgba(255,255,255,.2) !important; line-height:1.5 !important; }
</style>
""", unsafe_allow_html=True)

# ── State ─────────────────────────────────────────────────────────────────────
for k, v in [("done", False), ("result", None)]:
    if k not in st.session_state: st.session_state[k] = v

LOGS = "logs"; os.makedirs(LOGS, exist_ok=True); os.makedirs("results", exist_ok=True)

ICONS = {"Cardiologist":"🫀","Psychologist":"🧠","Pulmonologist":"🫁",
         "Neurologist":"⚡","Endocrinologist":"⚗️","Gastroenterologist":"🔬",
         "Dermatologist":"🩺","Hematologist":"🩸"}

def save_log(r):
    with open(f"{LOGS}/{r['id']}.json","w",encoding="utf-8") as f: json.dump(r,f,indent=2,ensure_ascii=False)

def load_logs():
    out=[]
    for fn in sorted(os.listdir(LOGS),reverse=True):
        if fn.endswith(".json"):
            with open(f"{LOGS}/{fn}",encoding="utf-8") as f: out.append(json.load(f))
    return out

def parse_risk(s):
    u=s.upper()
    if "OVERALL RISK LEVEL" in u:
        a=u.split("OVERALL RISK LEVEL")[-1]
        for r in ["HIGH","MODERATE","LOW"]:
            if r in a[:100]: return r.capitalize()
    return "Unknown"

def risk_pill(r):
    cls={"High":"rp-h","Moderate":"rp-m","Low":"rp-l"}.get(r,"rp-u")
    dot={"High":"🔴","Moderate":"🟡","Low":"🟢"}.get(r,"⚪")
    return f'<span class="rp {cls}">{dot} {r} Risk</span>'

def conf_pill(c):
    cls={"High":"cp-h","Medium":"cp-m","Low":"cp-l"}.get(c,"cp-l")
    return f'<span class="cp {cls}">{c}</span>'

def conf_bar(c):
    w={"High":"100%","Medium":"62%","Low":"28%"}.get(c,"28%")
    cls={"High":"cf-h","Medium":"cf-m","Low":"cf-l"}.get(c,"cf-l")
    return f'<div class="ctr"><div class="{cls}" style="width:{w}"></div></div>'

def clean_text(t):
    """Final pass to remove any stray markdown before display."""
    t = re.sub(r'\*{1,3}(.*?)\*{1,3}', r'\1', t)
    t = re.sub(r'^#{1,6}\s+', '', t, flags=re.MULTILINE)
    t = re.sub(r'^\s*[-•]\s+', '', t, flags=re.MULTILINE)
    t = re.sub(r'\n{3,}', '\n\n', t)
    return t.strip()


def generate_pdf(result) -> bytes:
    from fpdf import FPDF

    def safe(t):
        return (t or "").encode("latin-1", errors="replace").decode("latin-1")

    summary = result.get("final_summary", "")
    risk    = result.get("risk", "Unknown")
    source  = result.get("source", "")
    ts      = result.get("timestamp", "")
    specs   = result.get("specialists", [])
    sp_data = result.get("specialist_reports", {})

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Header bar
    pdf.set_fill_color(17, 34, 64)
    pdf.rect(0, 0, 210, 28, "F")
    pdf.set_font("Helvetica", "B", 15)
    pdf.set_text_color(255, 255, 255)
    pdf.set_xy(0, 7)
    pdf.cell(0, 8, "MediAgent  |  Diagnosis Report", align="C")
    pdf.set_font("Helvetica", "", 8)
    pdf.set_text_color(160, 185, 210)
    pdf.set_xy(0, 18)
    pdf.cell(0, 5, safe(f"{ts}   |   {source}"), align="C")
    pdf.set_y(34)

    # Risk banner
    rc = {"High": (239, 68, 68), "Moderate": (245, 158, 11), "Low": (16, 185, 129)}.get(risk, (148, 163, 184))
    pdf.set_fill_color(*rc)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(0, 9, f"  Risk: {risk}   |   Specialists consulted: {len(specs)}", fill=True, ln=True)
    pdf.ln(5)

    def section_header(title):
        pdf.set_fill_color(237, 242, 251)
        pdf.set_text_color(17, 34, 64)
        pdf.set_font("Helvetica", "B", 9)
        pdf.cell(0, 7, f"  {title}", fill=True, ln=True)
        pdf.ln(1)
        pdf.set_text_color(51, 65, 85)
        pdf.set_font("Helvetica", "", 9)

    def body(text):
        pdf.multi_cell(0, 5, safe(text))
        pdf.ln(3)

    # Patient Summary
    ps = clean_text(parse_section(summary, "PATIENT SUMMARY", ["TOP 3", "KEY FINDINGS", "NEXT STEPS", "OVERALL RISK"]))
    if ps:
        section_header("PATIENT SUMMARY")
        body(ps)

    # Top 3 Diagnoses
    dx = clean_text(parse_section(summary, "TOP 3 DIAGNOSES", ["KEY FINDINGS", "NEXT STEPS", "OVERALL RISK"]))
    if dx:
        section_header("TOP 3 DIAGNOSES")
        body(dx)

    # Key Findings
    kf = clean_text(parse_section(summary, "KEY FINDINGS", ["NEXT STEPS", "OVERALL RISK"]))
    if kf:
        section_header("KEY FINDINGS")
        for line in kf.split("\n"):
            line = line.strip()
            if line:
                pdf.multi_cell(0, 5, safe(f"  \u2022  {line}"))
        pdf.ln(3)

    # Next Steps
    ns = clean_text(parse_section(summary, "NEXT STEPS", ["OVERALL RISK"]))
    if not ns:
        ns = clean_text(parse_section(summary, "RECOMMENDED NEXT", ["OVERALL RISK"]))
    if ns:
        section_header("NEXT STEPS")
        for i, line in enumerate([l.strip() for l in ns.split("\n") if l.strip()], 1):
            line = re.sub(r'^\d+\.\s*', '', line)
            pdf.multi_cell(0, 5, safe(f"  {i}. {line}"))
        pdf.ln(3)

    # Risk Level
    rl = clean_text(parse_section(summary, "OVERALL RISK LEVEL", []))
    if rl:
        section_header("OVERALL RISK LEVEL")
        body(f"{risk} \u2014 {rl}")

    # Specialist Confidence
    if sp_data:
        section_header("SPECIALIST CONFIDENCE")
        for sp, data in sp_data.items():
            if isinstance(data, dict):
                conf   = data.get("confidence", "?")
                reason = data.get("confidence_reason", "")
                pdf.multi_cell(0, 5, safe(f"  {sp}: {conf} \u2014 {reason}"))
        pdf.ln(3)

    # Individual Specialist Reports
    if sp_data:
        section_header("SPECIALIST REPORTS")
        for sp, data in sp_data.items():
            if isinstance(data, dict) and not data.get("failed"):
                pdf.set_font("Helvetica", "B", 8)
                pdf.set_text_color(14, 90, 160)
                pdf.cell(0, 5, safe(f"[ {sp} ]"), ln=True)
                pdf.set_font("Helvetica", "", 8)
                pdf.set_text_color(51, 65, 85)
                pdf.multi_cell(0, 4, safe(clean_text(data.get("report", ""))))
                pdf.ln(2)

    # Footer
    pdf.set_y(-12)
    pdf.set_font("Helvetica", "I", 7)
    pdf.set_text_color(148, 163, 184)
    pdf.cell(0, 4, "For educational and research use only. Not a substitute for professional medical advice.", align="C")

    return bytes(pdf.output())

def parse_section(text, start_key, end_keys):
    """Extract a named section from plain-text report."""
    lines = text.split('\n')
    inside = False
    result = []
    for line in lines:
        lu = line.upper().strip()
        if start_key in lu:
            inside = True
            continue
        if inside:
            if any(k in lu for k in end_keys) and lu:
                break
            result.append(line)
    return '\n'.join(result).strip()

def render_diagnosis(result):
    summary  = result["final_summary"]
    risk     = result["risk"]
    specs    = result["specialists"]
    ts       = result["timestamp"]
    sp_data  = result.get("specialist_reports", {})

    # Top bar
    st.markdown(
        f'<div style="display:flex;align-items:center;justify-content:space-between;'
        f'background:#fff;border:1px solid #dde4ef;border-radius:12px;padding:16px 22px;'
        f'margin-bottom:18px;box-shadow:0 1px 4px rgba(17,34,64,.05)">'
        f'<div><div style="font-family:\'DM Serif Display\',serif;font-size:17px;color:#112240">Diagnosis Report</div>'
        f'<div style="font-size:12px;color:#94a3b8;margin-top:2px">{result["source"]}</div></div>'
        f'<div style="display:flex;align-items:center;gap:14px">'
        f'<div style="font-size:12px;color:#94a3b8">{ts}</div>'
        f'{risk_pill(risk)}</div></div>',
        unsafe_allow_html=True
    )

    # Stats
    c1,c2,c3,c4 = st.columns(4)
    hc = sum(1 for d in sp_data.values() if isinstance(d,dict) and d.get("confidence")=="High")
    rc = {"High":"#EF4444","Moderate":"#F59E0B","Low":"#10B981"}.get(risk,"#94a3b8")
    for col, val, lbl in [(c1,len(specs),"Specialists"),(c2,hc,"High Confidence"),
                          (c3,risk,"Risk Level"),(c4,result.get("model",OLLAMA_MODEL),"Model")]:
        color = rc if lbl=="Risk Level" else "#112240"
        col.markdown(
            f'<div class="sb"><div class="sv" style="color:{color};font-size:{"22px" if len(str(val))>6 else "26px"}">{val}</div>'
            f'<div class="sl">{lbl}</div></div>',
            unsafe_allow_html=True
        )

    st.markdown('<hr class="hr">', unsafe_allow_html=True)

    # Confidence panel
    if sp_data:
        st.markdown('<div class="cl">🎯 &nbsp;SPECIALIST CONFIDENCE</div>', unsafe_allow_html=True)
        cols = st.columns(2)
        for i,(sp,data) in enumerate(sp_data.items()):
            conf   = data.get("confidence","Medium")
            reason = data.get("confidence_reason","")
            icon   = ICONS.get(sp,"🔬")
            with cols[i%2]:
                st.markdown(
                    f'<div class="card" style="padding:13px 17px;margin-bottom:8px">'
                    f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px">'
                    f'<span style="font-size:13px;font-weight:700;color:#112240">{icon} {sp}</span>'
                    f'{conf_pill(conf)}</div>'
                    f'{conf_bar(conf)}'
                    f'<div style="font-size:12px;color:#64748b;margin-top:5px;font-style:italic">{reason}</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )
        st.markdown('<hr class="hr">', unsafe_allow_html=True)

    # Patient Summary
    ps = parse_section(summary, "PATIENT SUMMARY", ["TOP 3","KEY FINDINGS","NEXT STEPS","OVERALL RISK"])
    ps = clean_text(ps)
    if ps:
        st.markdown(
            f'<div class="card card-t-navy"><div class="cl">👤 &nbsp;PATIENT SUMMARY</div>'
            f'<div style="font-size:14px;color:#1e293b;line-height:1.7">{ps}</div></div>',
            unsafe_allow_html=True
        )

    # Top 3 Diagnoses — parsed into structured blocks
    dx_raw = parse_section(summary, "TOP 3 DIAGNOSES", ["KEY FINDINGS","NEXT STEPS","OVERALL RISK"])
    dx_raw = clean_text(dx_raw)
    if dx_raw:
        st.markdown('<div class="card card-t-violet"><div class="cl">🔬 &nbsp;TOP 3 DIAGNOSES</div>', unsafe_allow_html=True)
        # Split into individual diagnosis blocks by numbered prefix
        blocks = re.split(r'(?=^\d+\.)', dx_raw, flags=re.MULTILINE)
        for block in blocks:
            block = block.strip()
            if not block: continue
            lines = block.split('\n')
            title_line = lines[0].strip()
            # Parse title and likelihood
            likelihood = ""
            lm = re.search(r'—\s*(Most Likely|Possible|Less Likely)', title_line, re.IGNORECASE)
            if lm:
                likelihood = lm.group(1)
                title_line = title_line[:lm.start()].strip()
            title_line = re.sub(r'^\d+\.\s*', '', title_line).strip()
            lk_color = {"Most Likely":"#065F46","Possible":"#92400E","Less Likely":"#64748b"}.get(likelihood,"#64748b")
            lk_bg    = {"Most Likely":"#D1FAE5","Possible":"#FEF3C7","Less Likely":"#F1F5F9"}.get(likelihood,"#F1F5F9")

            rows_html = ""
            for line in lines[1:]:
                line = line.strip()
                if not line: continue
                if line.lower().startswith("flagged by:"):
                    rows_html += f'<div class="dx-row"><span class="dx-key">Flagged</span><span class="dx-val">{line[11:].strip()}</span></div>'
                elif line.lower().startswith("evidence for:"):
                    rows_html += f'<div class="dx-row"><span class="dx-key">For</span><span class="dx-val dx-for">{line[13:].strip()}</span></div>'
                elif line.lower().startswith("evidence against:"):
                    rows_html += f'<div class="dx-row"><span class="dx-key">Against</span><span class="dx-val dx-ag">{line[17:].strip()}</span></div>'
                else:
                    rows_html += f'<div class="dx-row"><span class="dx-key"></span><span class="dx-val">{line}</span></div>'

            st.markdown(
                f'<div class="dx-block">'
                f'<div class="dx-title">{title_line}'
                f'{"&nbsp;&nbsp;<span style=\'font-size:11px;font-weight:700;padding:2px 9px;border-radius:999px;background:"+lk_bg+";color:"+lk_color+"\'>" + likelihood + "</span>" if likelihood else ""}'
                f'</div>'
                f'{rows_html}</div>',
                unsafe_allow_html=True
            )
        st.markdown('</div>', unsafe_allow_html=True)

    # Key Findings
    kf_raw = parse_section(summary, "KEY FINDINGS", ["NEXT STEPS","RECOMMENDED NEXT","OVERALL RISK"])
    kf_raw = clean_text(kf_raw)
    if kf_raw:
        items = [l.strip() for l in kf_raw.split('\n') if l.strip() and not re.match(r'^-+$', l)]
        rows = "".join(
            f'<div class="fi"><span class="fi-dot"></span><span>{item}</span></div>'
            for item in items if item
        )
        st.markdown(
            f'<div class="card card-t-amber"><div class="cl">📋 &nbsp;KEY FINDINGS</div>{rows}</div>',
            unsafe_allow_html=True
        )

    # Next Steps
    ns_raw = parse_section(summary, "NEXT STEPS", ["OVERALL RISK"])
    if not ns_raw:
        ns_raw = parse_section(summary, "RECOMMENDED NEXT", ["OVERALL RISK"])
    ns_raw = clean_text(ns_raw)
    if ns_raw:
        items = [l.strip() for l in ns_raw.split('\n') if l.strip()]
        rows = ""
        for i, item in enumerate(items, 1):
            item = re.sub(r'^\d+\.\s*', '', item).strip()
            if item:
                rows += f'<div class="ns"><span class="ns-num">{i}</span><span>{item}</span></div>'
        st.markdown(
            f'<div class="card card-t-green"><div class="cl">✅ &nbsp;RECOMMENDED NEXT STEPS</div>{rows}</div>',
            unsafe_allow_html=True
        )

    # Risk Level
    rl_raw = parse_section(summary, "OVERALL RISK LEVEL", [])
    rl_raw = clean_text(rl_raw)
    if rl_raw:
        rc2 = {"High":"card-t-red","Moderate":"card-t-amber","Low":"card-t-green"}.get(risk,"card-t-navy")
        st.markdown(
            f'<div class="card {rc2}"><div class="cl">⚠️ &nbsp;OVERALL RISK LEVEL</div>'
            f'<div style="display:flex;align-items:center;gap:14px;margin-top:4px">'
            f'{risk_pill(risk)}'
            f'<span style="font-size:13px;color:#334155">{rl_raw}</span></div></div>',
            unsafe_allow_html=True
        )

    # Specialist detail (expandable)
    if sp_data:
        st.markdown('<hr class="hr">', unsafe_allow_html=True)
        st.markdown('<div class="cl" style="margin-bottom:10px">🩺 &nbsp;SPECIALIST REPORTS</div>', unsafe_allow_html=True)
        for sp, data in sp_data.items():
            icon   = ICONS.get(sp,"🔬")
            conf   = data.get("confidence","?")
            report = clean_text(data.get("report",""))
            with st.expander(f"{icon}  {sp}  —  Confidence: {conf}"):
                # Parse into sections for clean display
                for sec_key, sec_label, sec_color in [
                    ("FINDINGS",            "Findings",            "#0EA5E9"),
                    ("DIFFERENTIAL",        "Differential Diagnosis","#7C3AED"),
                    ("PRIMARY DIAGNOSIS",   "Primary Diagnosis",   "#112240"),
                    ("NEXT STEPS",          "Next Steps",          "#10B981"),
                ]:
                    end_keys = ["FINDINGS","DIFFERENTIAL","PRIMARY","NEXT STEPS","CONFIDENCE"]
                    end_keys2 = [k for k in end_keys if k != sec_key]
                    sec_text = parse_section(report, sec_key, end_keys2)
                    sec_text = clean_text(sec_text)
                    if sec_text:
                        st.markdown(
                            f'<div class="sp-section">'
                            f'<div class="sp-section-title" style="color:{sec_color}">{sec_label}</div>'
                            f'<div class="sp-text">{sec_text}</div>'
                            f'</div>',
                            unsafe_allow_html=True
                        )

    # Download
    st.markdown('<hr class="hr">', unsafe_allow_html=True)
    dl1, dl2 = st.columns(2)
    out_path = result.get("output_path", "")
    if out_path and os.path.exists(out_path):
        dl1.download_button(
            "⬇️  Download Report (.txt)",
            data=open(out_path, encoding="utf-8").read(),
            file_name=f"diagnosis_{result['id']}.txt",
            mime="text/plain",
            use_container_width=True,
        )
    try:
        pdf_bytes = generate_pdf(result)
        dl2.download_button(
            "⬇️  Download Report (.pdf)",
            data=pdf_bytes,
            file_name=f"diagnosis_{result['id']}.pdf",
            mime="application/pdf",
            use_container_width=True,
        )
    except Exception as _pdf_err:
        dl2.warning(f"PDF unavailable: {_pdf_err}")

def run_diagnosis(medical_report, source_label):
    selected  = select_specialists(medical_report)
    agents    = {n: SPECIALIST_REGISTRY[n](medical_report) for n in selected}
    responses = {}
    ph = st.empty()
    total = len(selected)

    CONF_COLOR = {"High": "#10B981", "Medium": "#F59E0B", "Low": "#94a3b8"}

    def step_bar(phase, done_count):
        s1 = '<span style="color:#10B981;font-weight:700;font-size:12px">✓ Input</span>'
        s2_color = "#0EA5E9" if phase == "agents" else "#10B981" if phase == "mdt" else "#94a3b8"
        s2_prefix = "⟳ " if phase == "agents" else "✓ "
        s2 = f'<span style="color:{s2_color};font-weight:700;font-size:12px">{s2_prefix}Specialists ({done_count}/{total})</span>'
        s3_color = "#0EA5E9" if phase == "mdt" else "#94a3b8"
        s3_prefix = "⟳ " if phase == "mdt" else ""
        s3 = f'<span style="color:{s3_color};font-weight:700;font-size:12px">{s3_prefix}Synthesis</span>'
        arr = '<span style="color:#cbd5e1;font-size:12px">→</span>'
        return f'<div style="display:flex;align-items:center;gap:10px;margin-bottom:14px;padding-bottom:10px;border-bottom:1px solid #eef2f7">{s1}{arr}{s2}{arr}{s3}</div>'

    def upd(done_data, pending, phase="agents"):
        bd = ""
        for s, data in done_data.items():
            conf = data.get("confidence", "?")
            col  = CONF_COLOR.get(conf, "#94a3b8")
            bd  += (
                f'<span class="ab ab-d" style="border-color:{col}">'
                f'{ICONS.get(s,"🔬")} {s}&nbsp;'
                f'<span style="color:{col};font-size:10px;font-weight:700">{conf}</span>'
                f'</span>'
            )
        bp = "".join(
            f'<span class="ab ab-r">{ICONS.get(s,"🔬")} {s}</span>'
            for s in pending
        )
        ph.markdown(
            f'<div class="card" style="padding:16px 20px">'
            f'{step_bar(phase, len(done_data))}'
            f'<div style="display:flex;flex-wrap:wrap;gap:4px">{bd}{bp}</div>'
            f'</div>',
            unsafe_allow_html=True
        )

    done_data = {}
    upd({}, list(selected), phase="agents")

    AGENT_TIMEOUT = 120  # seconds per specialist

    with ThreadPoolExecutor() as ex:
        futs = {ex.submit(lambda n=n, a=a: (n, a.run())): n for n, a in agents.items()}
        for fut in as_completed(futs):
            try:
                n, resp = fut.result(timeout=AGENT_TIMEOUT)
            except FuturesTimeoutError:
                n = futs[fut]
                resp = {
                    "report": f"{n} timed out after {AGENT_TIMEOUT}s.",
                    "confidence": "Low",
                    "confidence_reason": "Agent timed out.",
                    "failed": True,
                }
            responses[n] = resp
            done_data[n] = resp
            pending = [s for s in selected if s not in done_data]
            upd(done_data, pending, phase="agents")

    # Only pass successful reports to MDT to avoid polluting the synthesis
    valid_reports = {k: v for k, v in responses.items() if not v.get("failed")}

    upd(done_data, [], phase="mdt")
    mdt    = MultidisciplinaryTeam(specialist_reports=valid_reports)
    fs     = mdt.run()
    ph.empty()
    risk   = parse_risk(fs)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    result = {
        "id": run_id, "timestamp": datetime.now().strftime("%d %b %Y, %I:%M %p"),
        "source": source_label, "specialists": selected,
        "specialist_reports": responses, "final_summary": fs,
        "risk": risk, "model": OLLAMA_MODEL,
    }
    out = f"results/{run_id}_diagnosis.txt"
    txt = f"Source: {source_label}\nSpecialists: {', '.join(selected)}\n\n"
    for sp, d in responses.items():
        txt += f"[{sp}] Confidence: {d.get('confidence','?')} — {d.get('confidence_reason','')}\n{d.get('report','')}\n\n"
    txt += f"{'='*60}\nFINAL SUMMARY\n{'='*60}\n{fs}\n"
    with open(out, "w", encoding="utf-8") as f: f.write(txt)
    result["output_path"] = out
    save_log(result)
    return result

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        '<div class="nb"><div class="nt">MediAgent</div>'
        '<div class="ns2">AI Diagnostic System</div></div>',
        unsafe_allow_html=True
    )
    st.markdown(
        f'<div class="nsec"><div class="nst">Model</div>'
        f'<div style="background:rgba(14,165,233,.1);border:1px solid rgba(14,165,233,.2);'
        f'border-radius:7px;padding:8px 12px">'
        f'<span style="font-size:10px;color:rgba(255,255,255,.3)">OLLAMA /</span> '
        f'<span style="font-size:13px;font-weight:600;color:#7dd3fc !important">{OLLAMA_MODEL}</span>'
        f'</div></div>',
        unsafe_allow_html=True
    )
    st.markdown('<div class="nd"></div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="nsec"><div class="nst">How it works</div>'
        '<div style="display:flex;flex-direction:column;gap:10px;margin-top:6px">'
        '<div style="display:flex;align-items:flex-start;gap:10px">'
        '<span style="min-width:22px;height:22px;border-radius:50%;background:rgba(14,165,233,.2);'
        'border:1px solid rgba(14,165,233,.3);display:flex;align-items:center;justify-content:center;'
        'font-size:11px;font-weight:700;color:#7dd3fc">1</span>'
        '<span style="font-size:12px;color:rgba(255,255,255,.55);line-height:1.5">Upload a report or enter symptoms manually</span>'
        '</div>'
        '<div style="display:flex;align-items:flex-start;gap:10px">'
        '<span style="min-width:22px;height:22px;border-radius:50%;background:rgba(14,165,233,.2);'
        'border:1px solid rgba(14,165,233,.3);display:flex;align-items:center;justify-content:center;'
        'font-size:11px;font-weight:700;color:#7dd3fc">2</span>'
        '<span style="font-size:12px;color:rgba(255,255,255,.55);line-height:1.5">Up to 8 specialist agents analyze in parallel</span>'
        '</div>'
        '<div style="display:flex;align-items:flex-start;gap:10px">'
        '<span style="min-width:22px;height:22px;border-radius:50%;background:rgba(14,165,233,.2);'
        'border:1px solid rgba(14,165,233,.3);display:flex;align-items:center;justify-content:center;'
        'font-size:11px;font-weight:700;color:#7dd3fc">3</span>'
        '<span style="font-size:12px;color:rgba(255,255,255,.55);line-height:1.5">Lead physician synthesizes a unified diagnosis</span>'
        '</div>'
        '</div></div>',
        unsafe_allow_html=True
    )
    st.markdown('<div class="nd"></div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="ndis">⚠️ Educational use only. Not a substitute for professional medical advice.</div>',
        unsafe_allow_html=True
    )

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(
    f'<div class="ph"><div class="ph-inner">'
    f'<div class="ph-brand"><div class="ph-icon">⚕️</div>'
    f'<div><div class="ph-title">MediAgent</div><div class="ph-sub">AI-Powered Clinical Diagnosis System</div></div></div>'
    f'<div class="ph-date">{datetime.now().strftime("%d %B %Y")}</div>'
    f'</div></div>',
    unsafe_allow_html=True
)

st.markdown('<div class="wrap">', unsafe_allow_html=True)
tab1, tab2 = st.tabs(["⚕️  New Diagnosis", "📂  History"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div style="height:16px"></div>', unsafe_allow_html=True)
    mode = st.radio("mode", ["📄  Upload Report", "✏️  Enter Symptoms"], horizontal=True, label_visibility="collapsed")

    medical_report = None
    source_label   = ""

    if "Upload" in mode:
        st.markdown('<div class="card" style="margin-top:8px">', unsafe_allow_html=True)
        cu, cs = st.columns(2)
        with cu:
            st.markdown('<div style="font-size:12px;font-weight:600;color:#475569;margin-bottom:6px">Upload .txt file</div>', unsafe_allow_html=True)
            up = st.file_uploader("Upload .txt", type=["txt"], label_visibility="collapsed")
            if up:
                MAX_BYTES = 50_000
                raw_bytes = up.read()
                if len(raw_bytes) > MAX_BYTES:
                    st.error(f"File too large ({len(raw_bytes)//1024} KB). Max is {MAX_BYTES//1024} KB.")
                else:
                    medical_report = raw_bytes.decode("utf-8", errors="ignore")
                    source_label   = f"Uploaded: {up.name}"
                    st.success(f"Ready: {up.name}")
        with cs:
            st.markdown('<div style="font-size:12px;font-weight:600;color:#475569;margin-bottom:6px">Or pick a sample report</div>', unsafe_allow_html=True)
            rd = "Medical Reports"
            rfs = sorted([f for f in os.listdir(rd) if f.endswith(".txt")]) if os.path.isdir(rd) else []
            if rfs:
                ch = st.selectbox("Report", ["— select —"] + rfs, label_visibility="collapsed")
                if ch != "— select —":
                    with open(os.path.join(rd, ch), "r", encoding="utf-8", errors="ignore") as fh:
                        medical_report = fh.read()
                    source_label = f"File: {ch}"
                    st.success(f"Ready: {ch}")
            else:
                st.info("No reports in 'Medical Reports/' folder.")
        if medical_report:
            with st.expander("Preview report"):
                st.code(medical_report[:800] + ("…" if len(medical_report) > 800 else ""), language=None)
        st.markdown('</div>', unsafe_allow_html=True)

    else:
        st.markdown('<div class="card" style="margin-top:8px">', unsafe_allow_html=True)
        c1,c2,c3 = st.columns(3)
        name   = c1.text_input("Patient Name",  placeholder="Full name")
        age    = c2.text_input("Age",            placeholder="e.g. 42")
        gender = c3.selectbox("Gender", ["— select —","Male","Female","Other","Prefer not to say"])
        symptoms = st.text_area("Symptoms", placeholder="Describe symptoms — onset, duration, severity, location...", height=110)
        st.markdown('<hr class="hr">', unsafe_allow_html=True)
        ch2, cm2 = st.columns(2)
        with ch2:
            history = st.text_area("Medical History", placeholder="Known conditions, past surgeries, family history...", height=80)
        with cm2:
            medications = st.text_area("Current Medications", placeholder="Drug name, dosage, frequency...", height=80)
        st.markdown('</div>', unsafe_allow_html=True)
        if symptoms.strip():
            g = gender if gender != "— select —" else "Not specified"
            safe_name = re.sub(r'[^\w\s-]', '', name).strip() or 'Unknown'
            medical_report = (
                f"Patient Name: {safe_name}\nAge: {age or 'Unknown'}\nGender: {g}\n\n"
                f"Presenting Symptoms:\n{symptoms}\n\nMedical History:\n{history or 'None.'}\n\n"
                f"Current Medications:\n{medications or 'None.'}\n"
            )
            source_label = f"Manual — {safe_name}"

    st.markdown('<div style="height:6px"></div>', unsafe_allow_html=True)
    go = st.button("⚕️  Run Diagnostic Analysis", disabled=(medical_report is None), use_container_width=True)

    if go and medical_report:
        st.markdown('<hr class="hr">', unsafe_allow_html=True)
        with st.spinner(""):
            result = run_diagnosis(medical_report, source_label)
        st.session_state.done   = True
        st.session_state.result = result
        st.markdown('<hr class="hr">', unsafe_allow_html=True)
        st.markdown('<div class="sh">Results</div>', unsafe_allow_html=True)
        render_diagnosis(result)

    elif st.session_state.done and st.session_state.result:
        st.markdown('<hr class="hr">', unsafe_allow_html=True)
        st.markdown('<div class="sh">Last Diagnosis</div>', unsafe_allow_html=True)
        render_diagnosis(st.session_state.result)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div style="height:22px"></div>', unsafe_allow_html=True)
    st.markdown('<div class="sh">Patient History</div><div class="ss">All sessions stored in <code>logs/</code>.</div>', unsafe_allow_html=True)
    logs = load_logs()
    if not logs:
        st.markdown(
            '<div class="card" style="text-align:center;padding:44px 24px">'
            '<div style="font-size:30px;margin-bottom:10px">📭</div>'
            '<div style="font-family:\'DM Serif Display\',serif;font-size:17px;color:#112240;margin-bottom:5px">No records yet</div>'
            '<div style="font-size:13px;color:#94a3b8">Run a diagnostic session to see results here.</div>'
            '</div>', unsafe_allow_html=True
        )
    else:
        st.markdown(f'<div style="font-size:13px;color:#64748b;margin-bottom:14px"><strong style="color:#112240">{len(logs)}</strong> session(s) on record</div>', unsafe_allow_html=True)
        for log in logs:
            risk   = log.get("risk","Unknown")
            rc3    = {"High":"card-t-red","Moderate":"card-t-amber","Low":"card-t-green"}.get(risk,"card-t-navy")
            rcs    = {"High":"#EF4444","Moderate":"#F59E0B","Low":"#10B981"}.get(risk,"#94a3b8")
            sp_rps = log.get("specialist_reports",{})

            with st.expander(f"🗓️  {log.get('timestamp','')}   ·   {log.get('source','')}   ·   {risk} Risk"):
                h1,h2,h3,h4 = st.columns(4)
                hc2 = sum(1 for d in sp_rps.values() if isinstance(d,dict) and d.get("confidence")=="High")
                h1.metric("Specialists",    len(log.get("specialists",[])))
                h2.metric("Risk",           risk)
                h3.metric("Model",          log.get("model","—"))
                h4.metric("High Confidence",hc2)

                if sp_rps and isinstance(list(sp_rps.values())[0], dict):
                    st.markdown('<div class="cl" style="margin-top:14px;margin-bottom:8px">🎯 CONFIDENCE</div>', unsafe_allow_html=True)
                    bh = "".join(
                        f'<span style="display:inline-flex;align-items:center;gap:5px;margin:3px;padding:4px 11px;'
                        f'background:#F8FAFC;border:1px solid #e2e8f0;border-radius:7px;font-size:12px;color:#475569">'
                        f'{ICONS.get(sp,"🔬")} {sp} &nbsp;{conf_pill(d.get("confidence","?"))}</span>'
                        for sp,d in sp_rps.items() if isinstance(d,dict)
                    )
                    st.markdown(bh, unsafe_allow_html=True)

                st.markdown('<hr class="hr">', unsafe_allow_html=True)
                fs2 = clean_text(log.get("final_summary",""))
                st.markdown(
                    f'<div class="card {rc3}"><div class="cl">FINAL SUMMARY</div>'
                    f'<div style="font-size:13px;color:#334155;line-height:1.7;white-space:pre-wrap">{fs2}</div></div>',
                    unsafe_allow_html=True
                )
                op = log.get("output_path","")
                hd1, hd2 = st.columns(2)
                if op and os.path.exists(op):
                    with open(op, encoding="utf-8") as fh:
                        hd1.download_button("⬇️  Download (.txt)", fh.read(), f"diagnosis_{log['id']}.txt", "text/plain", key=f"dl_{log['id']}", use_container_width=True)
                try:
                    hd2.download_button("⬇️  Download (.pdf)", data=generate_pdf(log), file_name=f"diagnosis_{log['id']}.pdf", mime="application/pdf", key=f"pdf_{log['id']}", use_container_width=True)
                except Exception:
                    pass

st.markdown('</div>', unsafe_allow_html=True)