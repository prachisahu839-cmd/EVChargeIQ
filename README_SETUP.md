# EVChargeIQ setup

## Current submission scope
Use the existing ACN-Data Caltech and JPL datasets. Live BEE station integration is future scope.

## Run the existing Streamlit dashboard
1. Keep your original project folder and make a backup copy before replacing files.
2. Copy `.gitignore` and `requirements.txt` from this patch into the project root. Replace the existing `.gitignore` because the uploaded one contains PowerShell command text rather than valid ignore rules.
3. Open PowerShell in the project root.
4. Run:

```powershell
py --version
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

If PowerShell blocks activation, run this once in the same terminal and activate again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

If `.venv` already exists, do not create it again; activate it and install requirements.

## Optional: rerun model/data scripts
Only if you need to rerun ML/DL training or preprocessing scripts, install the larger optional set:

```powershell
pip install -r requirements-training.txt
```

This can take considerably longer, especially TensorFlow. It is not required to run the current dashboard with saved outputs.

## Security
Do not upload or commit `.env`. If an API token in a shared archive was real, revoke/rotate it at its provider and put the replacement only in your local `.env` file.
