#!/usr/bin/env bash

# Create virtual environment
python3 -m venv sts_cockpit_venv

# Active the environment to install packages
source sts_cockpit_venv/bin/activate

# Install modules to the environment
# NOTE: this contains the cpu version of pytorch to avoid clashes with cuda versions
## Assumes that if a user is leveraging the run functionality then they're not training from scratch
python3 -m pip install -r requirements.txt

# Boots into sts_cockpit
python3 scripts/sts_cockpit.py

# deactivates the environment
deactivate