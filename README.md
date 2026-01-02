How to run the project (quick start)

Create and activate a virtual environment (PowerShell):

python -m venv .venv ..venv\Scripts\Activate.ps1

Install dependencies (if a requirements.txt exists):

pip install -r requirements.txt

If there is no requirements.txt, at minimum install Streamlit to run the app:

pip install streamlit

Run the Streamlit app:

streamlit run app.py

Notes on running

If the project requires additional system packages (CUDA, PyTorch variants, etc.), consult the repository README or train.py / model.py for dependency hints.
On Windows Command Prompt use .venv\\Scripts\\activate.bat to activate the venv.
