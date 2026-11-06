#!/usr/bin/env bash

# Same setup as run_sts_cockpit_mac.sh, but starts the web UI (ui/) instead of the command line program.
# Open http://127.0.0.1:8765 once it says it is running. Ctrl+C stops it.
python3 -m venv sts_cockpit_venv
source sts_cockpit_venv/bin/activate
python3 -m pip install -r requirements.txt
python3 ui/server/app.py
deactivate
