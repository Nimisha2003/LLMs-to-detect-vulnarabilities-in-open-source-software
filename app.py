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

st.set_page_config(page_title="CODE SENTINEL", layout="wide", page_icon="🛡️")

# -----------------------------
# Utilities
# -----------------------------
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

logo_base64 = get_base64_of_bin_file("CODE SENTINEL LOGO.png")

st.markdown(f"""
<style>
.banner-container {{
  text-align: center;
  margin-bottom: 25px;
}}
.banner-container img {{
  width: 230px;
  margin-bottom: 15px;
  animation: pulse 3s infinite alternate;
}}
@keyframes pulse {{
  from {{ transform: scale(1); }}
  to   {{ transform: scale(1.08); }}
}}
.banner-title {{
  font-size: 48px;
  font-weight: 800;
  color: white;
  letter-spacing: 2px;
  text-transform: uppercase;
  text-shadow: 2px 2px 6px rgba(0,0,0,0.5);
}}
.subtext {{
  font-size: 18px;
  color: #ddd;
  margin-top: 8px;
}}
.badge-high {{
  background: crimson; color: white; padding: 4px 8px; border-radius: 6px;
}}
.badge-low {{
  background: seagreen; color: white; padding: 4px 8px; border-radius: 6px;
}}
.result-card {{
  background: #1f2430;
  padding: 12px;
  border-radius: 10px;
  margin-bottom: 10px;
}}
pre {{
  background:#2d2d3f; padding:10px; border-radius: 8px; overflow-x:auto;
}}
footer {{ color: gray; }}
</style>
<div class="banner-container">
  <img src="data:image/png;base64,{logo_base64}">
  <div class="banner-title">CODE SENTINEL</div>
  <div class="subtext">AI that defends your code from vulnerabilities</div>
</div>
""", unsafe_allow_html=True)

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("Navigation")
nav_choice = st.sidebar.radio("Go to:", ["Main", "About", "Help"])

if nav_choice == "About":
    st.sidebar.subheader("About CodeSentinel")
    st.sidebar.write("""
CodeSentinel is an AI-powered tool that scans code for vulnerabilities
in multiple programming languages, helping developers fix issues quickly.
""")
    st.sidebar.markdown("*Developers:* NIMISHA NORBURT, SANJAY SHAJU, REETHI XAVIER, THANA FAIZAL M M")

elif nav_choice == "Help":
    st.sidebar.subheader("Help")
    st.sidebar.write("""
1. Select your programming language.
2. Paste or upload your code.
3. Click 'Analyze Code' to scan for vulnerabilities.
4. Review results, charts, and suggested fixes.
""")
    st.sidebar.info("For support, contact_support@codesentinel.ai")

# -----------------------------
# Load tokenizer & model once
# -----------------------------
@st.cache_resource
def load_model_and_tokenizer():
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

model, tokenizer = load_model_and_tokenizer()

# -----------------------------
# Heuristic explanations
# -----------------------------
def explain_and_mitigate(line, lang):
    # strip comments and strings to reduce false positives
    line_clean = re.sub(r'(#[^\n]*|"[^"]*"|\'[^\']*\')', '', line).strip()

    if lang == "LLVM IR":
        if re.search(r'\bstrcpy\b|\bgets\s*\(', line_clean):
            return ("High", "Buffer Overflow risk", "Use strncpy(), fgets() with bounds checking")
        if re.search(r'\bunsafe_func\b', line_clean):
            return ("High", "Unsafe function call", "Replace with a safe validated function")
        return ("Low", "No known issues", "Review memory operations and bounds")

    if lang == "Python":
        if re.search(r'\beval\s*\(', line_clean):
            return ("High", "Code Injection via eval()", "Replace with safe parsing like ast.literal_eval")
        if re.search(r'\bexec\s*\(', line_clean):
            return ("High", "Use of exec()", "Avoid using exec() entirely")
        if re.search(r'\bos\.system\s*\(', line_clean) or re.search(r'\bsubprocess\.call\s*\(', line_clean):
            return ("High", "Command Injection risk", "Use subprocess.run([...], check=True) with sanitization")
        if re.search(r'\bpickle\.loads\s*\(|\byaml\.load\s*\(', line_clean):
            return ("High", "Insecure Deserialization", "Use json.loads or yaml.safe_load")
        if re.search(r'\binput\s*\(', line_clean):
            return ("Medium", "Unvalidated user input", "Validate and sanitize user inputs before use")
        if re.search(r'SELECT .* (format|%|f")', line_clean):
            return ("High", "SQL Injection risk", "Use parameterized queries or ORM methods")
        if re.search(r'verify\s*=\s*False', line_clean):
            return ("High", "SSL Verification Disabled", "Enable SSL certificate verification")
        if re.search(r'\b(md5|sha1)\s*\(', line_clean):
            return ("High", "Weak Hash Algorithm (MD5/SHA1)", "Use SHA-256 or bcrypt instead")
        return ("Low", "No known issues", "No action needed")
    
    if lang in ["C", "C++"]:
    # High severity vulnerabilities
        if re.search(r'\bgets\s*\(', line_clean):
           return ("High", "Buffer Overflow risk", "Use fgets() with bounds checking")
        if re.search(r'\bstrcpy\s*\(', line_clean):
           return ("High", "Unsafe strcpy()", "Use strncpy() instead with proper bounds")
        if re.search(r'\bsprintf\s*\(', line_clean):
           return ("High", "Format string vulnerability", "Use snprintf() with length limits")
        if re.search(r'\bsystem\s*\(', line_clean):
           return ("High", "Command Injection risk", "Avoid system(); use execvp with sanitization")
        if re.search(r'\bmemcpy\s*\(', line_clean):
           return ("High", "Potential Buffer Overflow in memcpy()", "Validate buffer sizes before copying")
        if re.search(r'\bstrcat\s*\(', line_clean):
           return ("High", "Unsafe strcat()", "Use strncat() with explicit bounds")
        if re.search(r'\bopen\s*\(', line_clean) and "O_CREAT" in line_clean and "0777" in line_clean:
           return ("High", "Insecure file permissions", "Use restrictive permissions like 0600 or 0640")

    # Medium severity vulnerabilities
        if re.search(r'\brand\s*\(', line_clean):
           return ("Medium", "Weak random number generation", "Use cryptographically secure RNG like arc4random or C++ <random>")
        if re.search(r'\btmpnam\s*\(', line_clean):
           return ("Medium", "Insecure temporary file creation", "Use mkstemp() instead")
        if re.search(r'\batoi\s*\(', line_clean):
           return ("Medium", "Unsafe integer conversion", "Use strtol() with error checking")
        if re.search(r'\bfree\s*\(', line_clean) and "NULL" not in line_clean:
           return ("Medium", "Potential double free", "Ensure pointer is set to NULL after free()")

    # Low severity / code quality issues
        if re.search(r'\bprintf\s*\(', line_clean) and "%" not in line_clean:
           return ("Low", "Suspicious printf usage", "Validate format strings to avoid misuse")
        if re.search(r'\bassert\s*\(', line_clean):
           return ("Low", "Reliance on assert()", "Do not use assert() for production security checks")
        if re.search(r'\bstrncpy\s*\(', line_clean) and "sizeof" not in line_clean:
           return ("Low", "Potential misuse of strncpy()", "Always provide correct buffer size")

        return ("Low", "No known issues", "No action needed")

    return ("Low", "Unknown language", "No action needed")

# -----------------------------
# Main
# -----------------------------
if nav_choice == "Main":
    tab1, tab2, tab3, tab4 = st.tabs(["Code Input", "Results", "Visualizations", "Report"])

    with tab1:
        target_language = st.selectbox("Select Language", ["Python", "LLVM IR", "C", "C++"])
        uploaded_file = st.file_uploader("Upload a code file", type=["py", "ll", "c", "cpp"])

        if uploaded_file is not None:
            code_input = uploaded_file.read().decode("utf-8")
        else:
            code_input = st.text_area(f"Paste your {target_language} code here:", height=250)

        if st.button("Analyze Code"):
            if not code_input.strip():
                st.warning("Please enter or upload some code to analyze.")
            else:
                with st.spinner("AI is scanning your code..."):
                    progress = st.progress(0)
                    for pct in range(100):
                        time.sleep(0.01)
                        progress.progress(pct + 1)
    
    with tab2:

                # Batched inference with truncation to avoid OOM
                lines = code_input.strip().split("\n")
                encoded = [tokenizer.encode(line, max_length=512) for line in lines]

                batch_size = 32
                probs = []
                with torch.no_grad():
                    for i in range(0, len(encoded), batch_size):
                        batch = encoded[i:i + batch_size]
                        if not batch:
                            continue
                        max_len = max(len(seq) for seq in batch)
                        padded = [seq + [0] * (max_len - len(seq)) for seq in batch]
                        x = torch.tensor(padded, dtype=torch.long)
                        logits = model(x).view(-1)
                        batch_probs = logits.tolist()  # already sigmoid in model
                        probs.extend(batch_probs)

                results, high_count, medium_count, low_count = [], 0, 0, 0
                for i, line in enumerate(lines):
                    severity, issue, fix = explain_and_mitigate(line, target_language)
                    ml_pred = "High" if probs[i] >= 0.5 else "Low"

                    # Combine ML prediction with heuristic severity
                    if severity == "High" or ml_pred == "High":
                        final_severity = "High"
                    elif severity == "Medium":
                        final_severity = "Medium"
                    else:
                        final_severity = "Low"

                    if final_severity == "High":
                        high_count += 1
                    elif final_severity == "Medium":
                        medium_count += 1
                    else:
                        low_count += 1

                    results.append({
                        "Line": line,
                        "Severity": final_severity,
                        "Issue": issue,
                        "Suggested Fix": fix,
                        "ML Score": round(probs[i], 3),
                    })

                # Summary message
                if high_count > 0:
                    st.error(f"{high_count} High Severity Vulnerabilities Found!")
                elif medium_count > 0:
                    st.warning(f"{medium_count} Medium Severity Vulnerabilities Found!")
                else:
                    st.success("No High or Medium Severity Vulnerabilities Found")

                # Results cards
                for res in results:
                    badge_class = (
                        "badge-high" if res['Severity'] == 'High'
                        else "badge-low" if res['Severity'] == 'Low'
                        else "badge-medium"
                    )
                    st.markdown(f"""<div class="result-card"><h4><span class="{badge_class}">{res['Severity']}</span> | {res['Issue']}</h4><pre>{res['Line']}</pre><b>Suggested Fix:</b> {res['Suggested Fix']}<br/><small>ML Score: {res['ML Score']}</small></div>""", unsafe_allow_html=True)

                st.session_state["results_df"] = pd.DataFrame(results)

    with tab3:
        df = st.session_state.get("results_df", pd.DataFrame(columns=["Severity"]))
        high_count = (df["Severity"] == "High").sum() if not df.empty else 0
        medium_count = (df["Severity"] == "Medium").sum() if not df.empty else 0
        low_count = (df["Severity"] == "Low").sum() if not df.empty else 0
        score = float(df["ML Score"].mean()) * 100.0 if not df.empty and "ML Score" in df.columns else 0.0

        st.subheader("Vulnerability Distribution")
        fig = px.pie(
            names=["High", "Medium", "Low"],
            values=[high_count, medium_count, low_count],
            color=["High", "Medium", "Low"],
            color_discrete_map={"High": "crimson", "Medium": "orange", "Low": "seagreen"}
        )
        st.plotly_chart(fig, use_container_width=True)

        # Gauge color reflects highest severity present
        color = "crimson" if high_count else ("orange" if medium_count else "green")
        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=score,
            title={'text': "Overall Security Score"},
            gauge={'axis': {'range': [0, 100]},
                   'bar': {'color': color}}
        ))
        st.plotly_chart(gauge, use_container_width=True)

    with tab4:
        df = st.session_state.get("results_df", pd.DataFrame())
        st.subheader("Download Report")
        if df.empty:
            st.info("Run an analysis to generate a report.")
        else:
            report_text = df.to_csv(index=False)
            st.download_button("Download CSV", report_text, file_name="vulnerability_report.csv")

    st.markdown("<footer style='text-align:center; margin-top: 30px;'>Powered by CodeSentinel AI — Securing Code, Saving Time</footer>", unsafe_allow_html=True)
    st.markdown("<div class='devs' style='text-align:center; color:#bbb; margin-top: 5px;'>Developed by: <b>Nimisha Norburt</b>, <b>Sanjay Shaju</b>, <b>Reethi Xavier</b>, <b>Thana Faizal M M</b></div>", unsafe_allow_html=True)