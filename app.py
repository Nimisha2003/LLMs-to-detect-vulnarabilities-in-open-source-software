import streamlit as st
import torch
import pandas as pd
import time
import plotly.express as px
import plotly.graph_objects as go
from tokenizer import CustomTokenizer
from model import SimpleTransformerClassifier
import base64
import json
import re

st.set_page_config(page_title="🛡 CODE SENTINEL", layout="wide", page_icon="🛡")

def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

logo_path = "CODE SENTINEL LOGO.png"
logo_base64 = get_base64_of_bin_file(logo_path)

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
            to {{ transform: scale(1.08); }}
        }}
        .banner-title {{
            font-size: 65px;
            font-weight: 900;
            color: white;
            letter-spacing: 3px;
            text-transform: uppercase;
            text-shadow: 2px 2px 6px rgba(0,0,0,0.5);
        }}
        .subtext {{
            font-size: 22px;
            color: #ddd;
            margin-top: 10px;
        }}
    </style>
    <div class="banner-container">
        <img src="data:image/png;base64,{logo_base64}">
        <div class="banner-title">CODE SENTINEL</div>
        <div class="subtext">💡 AI that defends your code from vulnerabilities</div>
    </div>
""", unsafe_allow_html=True)

st.sidebar.title("⚙ Navigation")
nav_choice = st.sidebar.radio("Go to:", ["Main", "About", "Help"])

if nav_choice == "About":
    st.sidebar.subheader("ℹ About CodeSentinel")
    st.sidebar.write("""
        *CodeSentinel* is an AI-powered tool that scans code for vulnerabilities
        in multiple programming languages, helping developers fix issues quickly.
    """)
    st.sidebar.markdown("*Developers:* SANJAY SHAJU, NIMISHA NORBURT, REETHI XAVIER, THANA FAIZAL")

elif nav_choice == "Help":
    st.sidebar.subheader("🆘 Help")
    st.sidebar.write("""
        1. Select your programming language.
        2. Paste or upload your code.
        3. Click 'Analyze Code' to scan for vulnerabilities.
        4. Review results, charts, and suggested fixes.
    """)
    st.sidebar.info("For support, contact support@codesentinel.ai")

if nav_choice == "Main":
    tab1, tab2, tab3, tab4 = st.tabs(["📝 Code Input", "📋 Results", "📊 Visualizations", "📄 Report"])

    with tab1:
        target_language = st.selectbox("🌍 Select Language", ["LLVM IR", "Python"])
        uploaded_file = st.file_uploader("📂 Upload a code file", type=["py", "ll"])
        if uploaded_file is not None:
            code_input = uploaded_file.read().decode("utf-8")
        else:
            code_input = st.text_area(f"✍ Paste your {target_language} code here:", height=250)

    with open("tokenizer.json", "r") as f:
        tok_data = json.load(f)

    tokenizer = CustomTokenizer()
    tokenizer.token_to_id = tok_data["token_to_id"]
    tokenizer.id_to_token = {int(k): v for k, v in tok_data["id_to_token"].items()}
    tokenizer.vocab_size = tok_data["vocab_size"]

    model = SimpleTransformerClassifier(vocab_size=tokenizer.vocab_size)
    model.load_state_dict(torch.load("vuln_model.pth", map_location=torch.device("cpu")))
    model.eval()

    # -----------------------
    # Updated Vulnerability Detection Function
    # -----------------------
    def explain_and_mitigate(line, lang):
        line_clean = re.sub(r'(#[^\n]*|".*?"|\'.*?\')', '', line).strip()

        if lang == "LLVM IR":
            if re.search(r'\bstrcpy\b|\bgets\s*\(', line_clean):
                return ("High", "🚨 Buffer Overflow risk", "🔧 Use strncpy(), fgets() with bounds checking")
            if re.search(r'\bunsafe_func\b', line_clean):
                return ("High", "🚨 Unsafe function call", "🔧 Replace with a safe validated function")

        elif lang == "Python":
            if re.search(r'\beval\s*\(', line_clean):
                return ("High", "🚨 Code Injection via eval()", "🔧 Replace with safe parsing like ast.literal_eval")
            if re.search(r'\bexec\s*\(', line_clean):
                return ("High", "🚨 Use of exec()", "🔧 Avoid using exec() entirely")
            if re.search(r'\bos\.system\s*\(', line_clean) or re.search(r'\bsubprocess\.call\s*\(', line_clean):
                return ("High", "🚨 Command Injection risk", "🔧 Use subprocess.run([...], check=True) with sanitization")
            if re.search(r'\bpickle\.loads\s*\(', line_clean) or re.search(r'\byaml\.load\s*\(', line_clean):
                return ("High", "🚨 Insecure Deserialization", "🔧 Use json.loads or yaml.safe_load")
            if re.search(r'\binput\s*\(', line_clean):
                return ("Medium", "⚠️ Insecure direct input usage", "🔧 Validate and sanitize user inputs before use")
            if re.search(r'SELECT.*\b(format|%|f")', line_clean):
                return ("High", "🚨 SQL Injection risk", "🔧 Use parameterized queries or ORM methods")
            if re.search(r'verify\s*=\s*False', line_clean):
                return ("High", "🚨 SSL Verification Disabled", "🔧 Enable SSL certificate verification")
            if re.search(r'\b(md5|sha1)\s*\(', line_clean):
                return ("High", "🚨 Weak Hash Algorithm (MD5/SHA1)", "🔧 Use SHA-256 or bcrypt instead")

        return ("Low", "✅ No known issues", "No action needed")

    if st.button("🔍 Analyze Code"):
        if not code_input.strip():
            st.warning("⚠ Please enter or upload some code to analyze.")
        else:
            with st.spinner("🧠 AI is scanning your code..."):
                progress = st.progress(0)
                for pct in range(100):
                    time.sleep(0.01)
                    progress.progress(pct+1)

                lines = code_input.strip().split("\n")
                encoded = [tokenizer.encode(line) for line in lines]
                max_len = max(len(seq) for seq in encoded)
                padded = [seq + [0] * (max_len - len(seq)) for seq in encoded]
                x = torch.tensor(padded)

                with torch.no_grad():
                    pred = model(x)
                    score = pred.mean().item()

                results, high_count, low_count = [], 0, 0
                for i, line in enumerate(lines):
                    severity, issue, fix = explain_and_mitigate(line, target_language)
                    if severity == "High":
                        high_count += 1
                    else:
                        low_count += 1
                    badge = f'<span class="badge-high">{severity}</span>' if severity == "High" else f'<span class="badge-low">{severity}</span>'
                    results.append({
                        "Line": line,
                        "Severity": badge,
                        "Issue": issue,
                        "Suggested Fix": fix
                    })

                df = pd.DataFrame(results)

                with tab2:
                    if high_count > 0:
                        st.error(f"🚨 {high_count} High Severity Vulnerabilities Found!")
                    else:
                        st.success("🎉 No Vulnerabilities Found")

                    st.write(f"*Vulnerability Score:* {score:.2f}")

                    for i, res in enumerate(results):
                        st.markdown(f"""
                            <div class="result-card" style="animation-delay:{i*0.3}s">
                                <h4>{res['Severity']} | {res['Issue']}</h4>
                                <pre style="background:#2d2d3f; padding:10px; border-radius:8px;">{res['Line']}</pre>
                                <b>Suggested Fix:</b> {res['Suggested Fix']}
                            </div>
                        """, unsafe_allow_html=True)

                with tab3:
                    fig = px.pie(
                        names=["High", "Low"],
                        values=[high_count, low_count],
                        color=["High", "Low"],
                        color_discrete_map={"High": "crimson", "Low": "seagreen"}
                    )
                    st.subheader("📊 Vulnerability Distribution")
                    st.plotly_chart(fig, use_container_width=True)

                    gauge = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=score*100,
                        title={'text': "Overall Security Score"},
                        gauge={'axis': {'range': [0, 100]},
                               'bar': {'color': "orange" if high_count else "green"}}
                    ))
                    st.plotly_chart(gauge, use_container_width=True)

                with tab4:
                    st.subheader("📄 Download Report")
                    report_text = df.to_csv(index=False)
                    st.download_button("⬇ Download CSV", report_text, file_name="vulnerability_report.csv")

st.markdown("<footer style='text-align:center; margin-top:30px; color:gray;'>🔐 Powered by CodeSentinel AI | Securing Code, Saving Time</footer>", unsafe_allow_html=True)
st.markdown('<div class="devs" style="text-align:center; color:#bbb; margin-top:5px;">👨‍💻 Developed by: <b>Sanjay Shaju</b>, <b>Nimisha Norburt</b>, <b>Reethi Xavier</b>, <b>Thana Faizal</b></div>', unsafe_allow_html=True)
