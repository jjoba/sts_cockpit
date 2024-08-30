echo Create virtual environment
python -m venv sts_cockpit_venv

echo Active the environment to install packages
source sts_cockpit_venv/Scripts/activate

echo Install modules to the environment
echo NOTE: this contains the cpu version of pytorch to avoid clashes with cuda versions
echo Assumes that if a user is leveraging the run functionality then they're not training from scratch
python -m pip install -r requirements.txt

echo Boots into sts_cockpit
python scripts/sts_cockpit.py

echo deactivates the environment
deactivate