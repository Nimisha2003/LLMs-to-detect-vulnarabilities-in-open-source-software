# Code Sentinel — README

Quick run and Git LFS notes

- This repository uses Git LFS for large model and dataset files. After cloning, install and enable Git LFS:

  git lfs install

- Pull LFS objects for the current branch (required to get `tokenizer.json`, `vuln_model.pth`, and `merged_dataset.jsonl`):

  git fetch origin
  git reset --hard origin/main
  git lfs pull origin main

- If you already have local changes you want to keep, back them up (create a branch or use `git stash`) before performing the `reset --hard`.

How to run (Windows PowerShell)

1. Create and activate a virtual environment:

   python -m venv .venv
   .\.venv\Scripts\Activate.ps1

2. Install dependencies (if `requirements.txt` exists):

   pip install -r requirements.txt

   If no requirements file is present, install at minimum:

   pip install streamlit pandas plotly torch

3. Run the Streamlit app:

   streamlit run app.py

Notes on running

- If the project requires additional system packages (CUDA, PyTorch variants, etc.), consult `train.py` / `model.py` for dependency hints.
- On Windows Command Prompt use `.venv\\Scripts\\activate.bat` to activate the venv.

Notes
- The repo history was migrated to Git LFS and force-pushed; collaborators should re-clone or follow the LFS pull steps above.
- `tokenizer.json` and `vuln_model.pth` must be present (non-pointer files) for the app to load correctly.
