echo Same setup as run_sts_cockpit_windows.bat, but starts the web UI (ui/) instead of the command line program
echo Open http://127.0.0.1:8765 once it says it is running. Ctrl+C stops it
python -m venv sts_cockpit_venv
source sts_cockpit_venv/Scripts/activate
python -m pip install -r requirements.txt
python ui/server/app.py
deactivate
