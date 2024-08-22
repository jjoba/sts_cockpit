# Slay the Spire Cockpit

## Purpose
Create a data and AI/ML tool as a companion to support players during runs. There are many analyses that already exist which look at card or relic win rates but are limited in that they look at each item in a silo. STS Cockpit works differently by dynamically updating the expected win rate calculation based on the cards already in the players deck and the accompanying relics (events, health, etc. to come in future updates).

<b>Scope:</b> card and relic selection at the end of combat or during events/shops. Upgrading and card removal is also supported. During combat decision-making is not and will not be supported unless the requisite data becomes available.

## How It Works
STS Cockpit provides a Command Line Interface (CLI) to manage adding/removing/upgrading cards from the deck as well as simulating potential choices (e.g. which of these three cards, if any, should I add to my deck?). When a given action or simulation is taken by the user STS Cockpit provides expected win rates for each choice (this is the number output to the terminal. 1 = 100% chance of victory). This is always compared to a baseline of doing nothing where a higher win rate than the baseline would suggest an adventageous choice.

Supporting the calculations is a four layer Feed Foward Neural Network with drop out regularization (~2.2M Parameters) trained to predict whether or not a given run will win. The model has "seen" a total of 77M runs, but only 1.5M runs were used for finetuning. These runs are all at A20 difficulty after watcher was released and do not contain daily runs, beta test, or endless mode.

All data comes from [this data dump](https://www.reddit.com/r/slaythespire/comments/jt5y1w/77_million_runs_an_sts_metrics_dump/)
For more detailed information on the fields and data structure see data_dump.md in the main git directory.

## Hardware Requirements
### Must Have
- Storage: at least 233 GB of free space. This accomodates the compressed raw data, uncompressed training data, and pytorch model objects
- RAM: at least 16 GB. Technically it will still run with slightly less, but will trigger paging and slow down dramatically

### Nice to Have:
- Storage: 293 GB of free space to accomodate an extra copy of the data compressed data files just in case something happes you don't have to wait for them to download again. Save these in a different directory than the compressed files being used.
- GPU: this will speed up model training SIGNIFICANTLY. For benchmarking purposes it takes ~50-60 hours on only a 2022 laptop i9 w/16 cores, ~15 hours on a 2024 M3 Pro, and a yet to be benchmarked faster speed on a 3090. If run time becomes an issue consider going into the train_model.py script and reduce the number of epochs for pre training alpha and pre training beta -- optionally, you could eliminate the pre train alpha setp entirely. Training runs on PyTorch and automatically supports CPU, GPU (Cuda), and MPS (Apple).
- More CPU Cores: data processing runs in parallel on all available cores and takes ~ 1 hour on an 12 Core M3 Pro. If going above 12 cores and only have 16 GB of RAM monitor memory usage. Processing will be net slower if the computer needs to create paging files. CPU usage can be changed by setting the num_cores variable in process_data.py to a fixed value.
- SSD: there's quite a bit of file IO that takes place (52K files to read in and ~75K written out before reading back in again and appending to the final csvs used for training). Consequently, an HDD will slow down data processing immensly. No meaningful impact seen to model training time.

## Building STS Cockpit
1. Clone git repo to local drive
2. Create the following file structure inside of sts_cockpit
- data
    - raw
    - master_lists
    - training_data
        - finetuning
            - train
            - test
        - pre_training_alpha
            - train
            - test
        - pret_training_beta
            - train
            - test
    - processing_failures
    - optionally create an additional directory for backup copy of raw data here (I use raw_copy DO NOT DELETE)
- scripts (this will already exist)
- model_objects
    - dev
        - a sub directory matching the model version seen in train_model.py
    - production
3. Copy data <u>as is</u> from google drive ("this data dump" above) into raw
4. Build data
    1. run create_master_lists.py
    2. run process_data.py
5. Train Model: run train_model.py
6. Use! run sts_cockpit.py

A couple notes:
1. All scripts, inclduing sts cockpit, expect to be run in the sts_cockpit directory (all pathing set to this)
2. In the future steps 4 & 5 can be handled within the STS Cockpit CLI itself

## Future Features
In no particular order, below are some development ideas
- Add support for relics (currently in progress)
- Incorporate events
- Incorporate act bosses (e.g. when starting act one input the boss to guide card choices)
- Add path choices
- Update model to be a sequence-based transformer to alleviate issues with cards being chosed too early in the run
- Update with more recent data (no idea how to get this, but would love it)
- Experiment with other FNN model architectures
- Add support for health and gold by floor
- Add shop simulation which includes gold amounts to provide best choice
- Clean up the CLI to make it more user friendly (who knows... maybe even make a GUI one day)
- Add data processing and model building functionality directly to sts_cockpit script to streamline experience