# ⚙️ Setup Guide

This guide walks you through setting up the project on your local machine from scratch.

---

## Prerequisites
- Python 3.10 or above — [Download here](https://www.python.org/downloads/)
- Git — [Download here](https://git-scm.com/)
- A code editor — VS Code recommended — [Download here](https://code.visualstudio.com/)

---

## Step 1 — Clone the Repo
```bash
git clone https://github.com/YOUR_USERNAME/multi-disease-risk-prediction.git
cd multi-disease-risk-prediction
```

## Step 2 — Create Virtual Environment
```bash
python -m venv venv
```
Activate it:
- **Windows:** `venv\Scripts\activate`
- **Mac/Linux:** `source venv/bin/activate`

You should see `(venv)` in your terminal now.

## Step 3 — Install All Packages
```bash
pip install -r requirements.txt
```

## Step 4 — Add the Datasets
Follow instructions in `docs/DATASETS.md` to download and place the CSV files correctly.

## Step 5 — Run a Notebook (for data exploration)
```bash
jupyter notebook
```
Open any notebook from the `notebooks/` folder.

## Step 6 — Run the Dashboard
```bash
streamlit run src/dashboard/app.py
```
This will open the app in your browser at `http://localhost:8501`

---

## Git Workflow for the Team

### Before starting work each day:
```bash
git pull origin main
```

### After finishing your work:
```bash
git add .
git commit -m "your message describing what you did"
git push origin main
```

### If you're working on a separate feature (recommended):
```bash
# Create your own branch
git checkout -b feature/your-name-task

# After work, push your branch
git push origin feature/your-name-task

# Then open a Pull Request on GitHub to merge into main
```

---

## Common Issues

**"Module not found" error** — Make sure your virtual environment is activated before running anything.

**Streamlit not opening** — Try `http://localhost:8501` manually in your browser.

**SHAP installation issues on Windows** — Run `pip install shap --no-binary shap`
