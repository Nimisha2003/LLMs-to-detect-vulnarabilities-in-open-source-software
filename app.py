
import base64
import json
import re
import time

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import torch

from tokenizer import CustomTokenizer
from model import SimpleTransformerClassifier

# Must be the first streamlit command
st.set_page_config(page_title="CODE SENTINEL", layout="wide", page_icon="🛡️")

# -----------------------------
# Modern CSS & Branding
# -----------------------------
def apply_theme():
    st.markdown("""
    <style>
    /* Global Styles */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;600;800&display=swap');
    html, body, [class*="css"] { font-family: 'Plus Jakarta Sans', sans-serif; }

    /* Custom Badges */
    .badge-high { background: #DC143C; color: white; padding: 5px 12px; border-radius: 8px; font-weight: bold; }
    .badge-medium { background: #FF8C00; color: white; padding: 5px 12px; border-radius: 8px; font-weight: bold; }
    .badge-low { background: #8A9A5B; color: white; padding: 5px 12px; border-radius: 8px; font-weight: bold; }

    /* Result Cards */
    .result-card {
        background: #1a1c23;
        padding: 20px;
        border-radius: 12px;
        margin-bottom: 15px;
        border: 1px solid #30363d;
        border-left: 6px solid #30363d;
    }
    .card-high { border-left-color: #DC143C !important; }
    .card-medium { border-left-color: #FF8C00 !important; }
    .card-low { border-left-color: #8A9A5B !important; }

    /* Banner Styling */
    .banner-container { text-align: center; padding: 2rem 0; }
    .banner-title { 
        font-size: 52px; font-weight: 800; color: white; 
        letter-spacing: 2px; margin-bottom: 0;
    }
    .subtext { color: #8b949e; font-size: 18px; margin-top: 5px; }
    
    /* Code blocks */
    pre { background: #0d1117 !important; color: #e6edf3 !important; padding: 15px; border-radius: 8px; border: 1px solid #30363d; }
    
    /* Quick Nav Buttons */
    .stButton>button { border-radius: 8px; transition: all 0.3s; }
    /* Logo zoom animation */
    .logo-anim { display: inline-block; width: 200px; height: auto; transform-origin: center; animation: zoom 3s ease-in-out infinite; will-change: transform; }
    @keyframes zoom {
        0% { transform: scale(1); opacity: 1; }
        50% { transform: scale(1.08); opacity: 0.98; }
        100% { transform: scale(1); opacity: 1; }
    }
    </style>
    """, unsafe_allow_html=True)

apply_theme()

# -----------------------------
# Model & Tokenizer Loader
# -----------------------------
@st.cache_resource
def load_assets():
    try:
        with open("tokenizer.json", "r", encoding="utf-8") as f:
            tok_data = json.load(f)
        tokenizer = CustomTokenizer()
        tokenizer.token_to_id = tok_data["token_to_id"]
        tokenizer.id_to_token = {int(k): v for k, v in tok_data["id_to_token"].items()}
        tokenizer.vocab_size = tok_data["vocab_size"]

        model = SimpleTransformerClassifier(vocab_size=tokenizer.vocab_size)
        model.load_state_dict(torch.load("vuln_model.pth", map_location=torch.device("cpu")))
        model.eval()
        return model, tokenizer
    except Exception as e:
        st.error(f"Asset Load Error: {e}")
        return None, None

model, tokenizer = load_assets()

# -----------------------------
# Scoring Logic (Standardized)
# -----------------------------
def calculate_standard_score(df):
    """
    Calculates a security score based on a weighted penalty system.
    Starting Score: 100
    High Severity: -5 points
    Medium Severity: -2 points
    Low Severity: -0 points
    """
    if df.empty: return 100
    h = (df['Severity'] == "High").sum()
    m = (df['Severity'] == "Medium").sum()
    penalty = (h * 5) + (m * 2)
    return max(0, 100 - penalty)

# -----------------------------
# Vulnerability Logic (Enhanced)
# -----------------------------
def analyze_line(line, lang):
    # Strip comments to avoid false positives
    line_clean = re.sub(r'(#[^\n]*|//.*|/\*.*?\*/|"[^"]*"|\'[^\']*\')', '', line).strip()
    
    # 1. LLVM IR VULNERABILITIES
    if lang == "LLVM IR":
        if re.search(r'\bstrcpy\b|\bgets\s*\(', line_clean):
            return ("High", "Buffer Overflow risk", "Use strncpy(), fgets() with bounds checking")
        if re.search(r'\bunsafe_func\b', line_clean):
            return ("High", "Unsafe function call", "Replace with a safe validated function")
        if "alloca" in line_clean and "i32" in line_clean:
            return ("Medium", "Stack Allocation Monitoring", "Review stack allocations for potential overflow.")
        if "getelementptr" in line_clean:
            return ("Medium", "CWE-129: Array Index Out of Bounds", "Ensure offsets are strictly bounded.")
        return ("Low", "No known issues", "Review memory operations and bounds")

    # 2. PYTHON VULNERABILITIES
    if lang == "Python":
        # --- High Severity ---
        if re.search(r'\beval\s*\(', line_clean):
            return ("High", "Code Injection via eval()", "Replace with safe parsing like ast.literal_eval")
        if re.search(r'\bexec\s*\(', line_clean):
            return ("High", "Use of exec()", "Avoid using exec() entirely to prevent arbitrary code execution")
        if re.search(r'\bos\.system\s*\(|\bsubprocess\.call\s*\(|shell\s*=\s*True', line_clean):
            return ("High", "Command Injection risk", "Use subprocess.run([...], check=True) with argument lists and shell=False")
        if re.search(r'\bpickle\.loads\s*\(|\byaml\.load\s*\(', line_clean) and "SafeLoader" not in line_clean:
            return ("High", "Insecure Deserialization", "Use json.loads() or yaml.safe_load() to prevent RCE")
        if re.search(r'SELECT .* (format|%|f")', line_clean, re.IGNORECASE):
            return ("High", "SQL Injection risk", "Use parameterized queries or ORM methods instead of string formatting")
        if re.search(r'verify\s*=\s*False', line_clean):
            return ("High", "SSL Verification Disabled", "Enable SSL certificate verification (verify=True) to prevent MITM")
        if re.search(r'\b(md5|sha1)\s*\(', line_clean):
            return ("High", "Weak Hash Algorithm (MD5/SHA1)", "Use SHA-256 or bcrypt (hashlib.sha256) for security")
        
        # --- Medium Severity ---
        if re.search(r'\binput\s*\(', line_clean):
            return ("Medium", "Unvalidated user input", "Validate and sanitize user inputs before use")
        if re.search(r'tempfile\.mktemp\s*\(', line_clean):
            return ("Medium", "Insecure Temporary File", "Use tempfile.mkstemp() to avoid race conditions")
        
        return ("Low", "No known issues", "No action needed")
    
    # 3. C/C++ VULNERABILITIES
    if lang in ["C", "C++"]:
        # --- High Severity ---
        if re.search(r'\bgets\s*\(', line_clean):
           return ("High", "CWE-120: Buffer Overflow risk", "Use fgets() with bounds checking instead of gets()")
        if re.search(r'\bstrcpy\s*\(', line_clean):
           return ("High", "Unsafe strcpy()", "Use strncpy() instead with proper size bounds")
        if re.search(r'\bsprintf\s*\(', line_clean):
           return ("High", "Format string vulnerability", "Use snprintf() with explicit length limits")
        if re.search(r'\bsystem\s*\(', line_clean):
           return ("High", "Command Injection risk", "Avoid system(); use execvp with argument arrays and sanitization")
        if re.search(r'\bmemcpy\s*\(', line_clean) and "sizeof" not in line_clean:
           return ("High", "Potential Buffer Overflow in memcpy()", "Validate buffer sizes before copying; ensure destination is large enough")
        if re.search(r'\bstrcat\s*\(', line_clean):
           return ("High", "Unsafe strcat()", "Use strncat() with explicit bounds to prevent buffer overflow")
        if re.search(r'\bopen\s*\(', line_clean) and "O_CREAT" in line and ("0777" in line or "S_IRWXU|S_IRWXG|S_IRWXO" in line):
           return ("High", "Insecure file permissions", "Use restrictive permissions like 0600 (S_IRUSR|S_IWUSR)")
        if re.search(r'\bfree\s*\(', line_clean) and "NULL" not in line:
            return ("High", "CWE-416: Potential Use After Free / Double Free", "Immediately set pointer to NULL after free() to prevent dangling access")

        # --- Medium Severity ---
        if re.search(r'\brand\s*\(', line_clean):
           return ("Medium", "Weak random number generation", "Use secure RNG like arc4random or C++ <random> (mt19937)")
        if re.search(r'\btmpnam\s*\(', line_clean):
           return ("Medium", "Insecure temporary file creation", "Use mkstemp() to prevent file creation race conditions")
        if re.search(r'\batoi\s*\(|\batol\s*\(', line_clean):
           return ("Medium", "Unsafe integer conversion", "Use strtol() with explicit error checking and range validation")
        if re.search(r'\bmalloc\s*\(\s*[a-zA-Z_0-9]*\s*\*\s*[a-zA-Z_0-9]*\s*\)', line_clean):
            return ("Medium", "Integer Overflow in Allocation", "Check for overflow before multiplication or use calloc")

        # --- Low Severity ---
        if re.search(r'\bprintf\s*\(', line_clean) and "%" not in line:
           return ("Low", "Suspicious printf usage", "Always provide a format string to avoid memory corruption")
        if re.search(r'\bassert\s*\(', line_clean):
           return ("Low", "Reliance on assert()", "Do not use assert() for production security checks; use robust error handling")
        if re.search(r'\bstrncpy\s*\(', line_clean) and "sizeof" not in line:
           return ("Low", "Potential misuse of strncpy()", "Always provide correct buffer size and ensure null-termination")

        return ("Low", "No known issues", "No action needed")

    return ("Low", "General Pattern", "No specific vulnerability identified")


def run_analysis(code, lang, progress_bar=None):
    """Run the line-by-line analysis and optionally update a Streamlit progress bar."""
    if not code or not code.strip():
        return pd.DataFrame()

    lines = code.split("\n")
    results = []
    encoded = [tokenizer.encode(l, max_length=512) for l in lines]

    with torch.no_grad():
        total = max(1, len(lines))
        for i, line in enumerate(lines):
            if not line.strip():
                if progress_bar:
                    progress_bar.progress(int((i + 1) / total * 100))
                continue

            h_sev, h_issue, h_fix = analyze_line(line, lang)
            x = torch.tensor([encoded[i]], dtype=torch.long)
            try:
                ml_score = float(model(x).view(-1)[0])
            except Exception:
                ml_score = 0.0

            final_sev, issue, fix = h_sev, h_issue, h_fix
            if ml_score > 0.75 and h_sev == "Low":
                final_sev, issue, fix = "High", "AI: Suspected High-Risk Pattern", "Deep inspection recommended."
            elif ml_score > 0.45 and h_sev == "Low":
                final_sev, issue, fix = "Medium", "AI: Suspicious Code Structure", "Verify operation safety."

            results.append({
                "Line": line, "Severity": final_sev,
                "Issue": issue, "Fix": fix, "ML": ml_score
            })

            if progress_bar:
                progress_bar.progress(int((i + 1) / total * 100))

    return pd.DataFrame(results)

# -----------------------------
# Main Application
# -----------------------------
def get_base64_logo():
    try:
        with open("CODE SENTINEL LOGO.png", "rb") as f:
            return base64.b64encode(f.read()).decode()
    except: return ""

logo_data = get_base64_logo()

st.markdown(f"""
<div class="banner-container">
    {"<img class='logo-anim' src='data:image/png;base64," + logo_data + "' width='200'/>" if logo_data else ""}
    <div class="banner-title">CODE SENTINEL</div>
    <div class="subtext">AI-Driven Vulnerability Detection Engine</div>
</div>
""", unsafe_allow_html=True)

st.sidebar.title("Navigation")
nav = st.sidebar.radio("Module", ["Scanner", "About", "Help"])

# Initialize session state for filtering
if "filter_severity" not in st.session_state:
    st.session_state["filter_severity"] = "All"

if nav == "Scanner":
    t1, t2, t3, t4 = st.tabs(["📤 Input", "🔍 Findings", "📊 Analytics", "📄 Export"])

    with t1:
        target_language = st.selectbox("Select Language", ["Python", "LLVM IR", "C", "C++"])
        uploaded_file = st.file_uploader("Upload a code file", type=["py", "ll", "c", "cpp"])        

        if uploaded_file is not None:
            try:
                code = uploaded_file.read().decode("utf-8")
            except Exception:
                code = uploaded_file.getvalue().decode("utf-8") if hasattr(uploaded_file, 'getvalue') else ""
        else:
            code = st.text_area(f"Paste your {target_language} code here:", height=300)

        lang = target_language

        if st.button("RUN ANALYSIS", use_container_width=True):
            if not code.strip():
                st.warning("Please enter code.")
            else:
                with st.spinner("Analyzing code architecture..."):
                    progress = st.progress(0)
                    df_results = run_analysis(code, lang, progress_bar=progress)
                    st.session_state["results"] = df_results
                    st.session_state["filter_severity"] = "All"
                    st.session_state["last_code"] = code
                    st.session_state["last_lang"] = lang
                    st.success("Analysis Complete!")

    with t2:
        # Allow re-running analysis from Findings tab using last uploaded/pasted code
        code_for_analysis = st.session_state.get("last_code", "")
        lang_for_analysis = st.session_state.get("last_lang", "Python")

        if st.button("Analyze Code", key="analyze_t2"):
            if not code_for_analysis.strip():
                st.warning("Please enter or upload some code to analyze.")
            else:
                with st.spinner("AI is scanning your code..."):
                    progress = st.progress(0)
                    df_new = run_analysis(code_for_analysis, lang_for_analysis, progress_bar=progress)
                    st.session_state["results"] = df_new
                    st.session_state["filter_severity"] = "All"
                    st.success("Analysis Complete!")

        df = st.session_state.get("results", pd.DataFrame())
        if not df.empty:
            active_filter = st.session_state["filter_severity"]
            
            # --- Quick Navigation (Moved to Findings Tab) ---
            h = (df['Severity'] == "High").sum()
            m = (df['Severity'] == "Medium").sum()
            
            st.markdown("### Quick Navigation")
            c_nav1, c_nav2, c_nav3 = st.columns(3)
            with c_nav1:
                if st.button(f"🔴 View {h} High Risks", use_container_width=True, key="nav_high"):
                    st.session_state["filter_severity"] = "High"
                    st.rerun()
            with c_nav2:
                if st.button(f"🟠 View {m} Medium Risks", use_container_width=True, key="nav_medium"):
                    st.session_state["filter_severity"] = "Medium"
                    st.rerun()
            with c_nav3:
                if st.button("🟢 View All Findings", use_container_width=True, key="nav_all"):
                    st.session_state["filter_severity"] = "All"
                    st.rerun()
            
            st.divider()

            # Filter Logic
            filtered_df = df if active_filter == "All" else df[df['Severity'] == active_filter]
            
            st.write(f"### Results for: {active_filter}")

            for _, row in filtered_df.iterrows():
                sev = row['Severity']
                s_lower = sev.lower()
                st.markdown(f"""
                <div class="result-card card-{s_lower}">
                    <span class="badge-{s_lower}">{sev}</span>
                    <h4 style="margin-top:15px;">{row['Issue']}</h4>
                    <pre><code>{row['Line']}</code></pre>
                    <p style="color:#e0e0e0; background: #2c2f38; padding: 12px; border-radius: 6px; border: 1px dashed #FF8C00; margin-top: 10px;">
                        <b>🛡️ Mitigation Strategy:</b><br>{row['Fix']}
                    </p>
                    <small style="opacity:0.5;">Transformer Confidence: {row['ML']:.2f}</small>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("Run a scan to see findings.")

    with t3:
        df = st.session_state.get("results", pd.DataFrame())
        if not df.empty:
            h = (df['Severity'] == "High").sum()
            m = (df['Severity'] == "Medium").sum()
            l = (df['Severity'] == "Low").sum()

            overall_score = calculate_standard_score(df)
            
            st.markdown("### Executive Summary")
            
            c1, c2 = st.columns(2)
            with c1:
                # Determine color theme based on score
                if overall_score < 50:
                    main_color = "#DC143C"
                elif overall_score < 80:
                    main_color = "#FF8C00"
                else:
                    main_color = "#8A9A5B"

                risk_level = "High" if overall_score < 50 else ("Medium" if overall_score < 80 else "Low")

                fig = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=overall_score,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': f"Vulnerability Health Index (Risk: {risk_level})", 'font': {'size': 24, 'color': 'white'}},
                    delta={'reference': 70, 'increasing': {'color': "#8A9A5B"}},
                    gauge={
                        'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "white"},
                        'bar': {'color': main_color, 'thickness': 0.25},
                        'bgcolor': "rgba(0,0,0,0)",
                        'borderwidth': 2,
                        'bordercolor': "gray",
                        'steps': [
                            {'range': [0, 50], 'color': "rgba(220, 20, 60, 0.1)"},
                            {'range': [50, 80], 'color': "rgba(255, 140, 0, 0.1)"},
                            {'range': [80, 100], 'color': "rgba(138, 154, 91, 0.1)"}
                        ],
                        'threshold': {
                            'line': {'color': "white", 'width': 4},
                            'thickness': 0.75,
                            'value': overall_score
                        }
                    }
                ))

                fig.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font={'color': "white", 'family': "Arial"},
                    height=450,
                    margin=dict(l=50, r=50, t=80, b=20)
                )

                st.plotly_chart(fig, use_container_width=True)



            with c2:
                pie = px.pie(
                    values=[h, m, l], 
                    names=['High', 'Medium', 'Low'],
                    color=['High', 'Medium', 'Low'],
                    color_discrete_map={'High': '#DC143C', 'Medium': '#FF8C00', 'Low': '#8A9A5B'},
                    title="Risk Distribution",
                    hole=0.4
                )
                st.plotly_chart(pie, use_container_width=True)
        else:
            st.info("No data available. Run analysis first.")

    with t4:
        df = st.session_state.get("results", pd.DataFrame())
        if not df.empty:
            st.markdown("### Export Security Audit")
            st.download_button("Download CSV Report", df.to_csv(index=False), "sentinel_audit.csv", use_container_width=True)
        else:
            st.info("No report generated.")

# Footer
st.markdown("<br><p style='text-align:center; color:gray;'>Code Sentinel AI — Securing Software through Deep Intelligence</p>", unsafe_allow_html=True)
st.markdown("<div style='text-align:right; color:#888;'>Built by Nimisha, Sanjay, Reethi, & Thana</div>", unsafe_allow_html=True)