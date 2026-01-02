Git LFS Notice
===============

On 2026-01-02 the repository history was migrated to use Git LFS for large files
and the rewritten `main` branch was force-pushed to `origin/main`.

Files moved to LFS:
- `merged_dataset.jsonl`
- `vuln_model.pth`
- `tokenizer.json`

What to do as a collaborator
- If you have no local changes, re-clone the repository:

  `git clone <repo-url>`

- If you have local work you want to keep, back it up (create a branch or patch), then:

  `git lfs install`
  `git fetch origin`
  `git reset --hard origin/main`

Notes
- History was rewritten and force-pushed; local branches that reference old commits
  may need manual reconciliation.
- Make sure `git lfs` is installed before interacting with the repo to avoid large
  file download issues.

If you want, I can add this note to the main `README.md` instead — tell me which you prefer.

How to run the project (quick start)
- Create and activate a virtual environment (PowerShell):

  python -m venv .venv
  .\.venv\Scripts\Activate.ps1

- Install dependencies (if a `requirements.txt` exists):

  pip install -r requirements.txt

  If there is no `requirements.txt`, at minimum install Streamlit to run the app:

  pip install streamlit

- Run the Streamlit app:

  streamlit run app.py

Notes on running
- If the project requires additional system packages (CUDA, PyTorch variants, etc.),
  consult the repository README or `train.py` / `model.py` for dependency hints.
- On Windows Command Prompt use `.venv\\Scripts\\activate.bat` to activate the venv.
