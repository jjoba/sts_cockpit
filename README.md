# Slay the Spire Cockpit

## Purpose
Create a data and AI/ML tool as a companion to support players during runs. There are many analyses that already exist which look at card or relic win rates but are limited in that they look at each item in a silo. STS Cockpit works differently by dynamically updating the expected win rate calculation based on the cards already in the players deck and the accompanying relics (events, health, etc. to come in future updates).

<b>Scope:</b> card and relic selection at the end of combat or during events/shops. Upgrading and card removal is also supported. During combat decision-making is not and will not be supported unless the requisite data becomes available.

## How It Works
STS Cockpit provides a Command Line Interface (CLI) to manage adding/removing/upgrading cards from the deck as well as simulating potential choices (e.g. which of these three cards, if any, should I add to my deck?). When a given action or simulation is taken by the user STS Cockpit provides expected win rates for each choice (this is the number output to the terminal. 1 = 100% chance of victory). This is always compared to a baseline of doing nothing where a higher win rate than the baseline would suggest an adventageous choice.

Supporting the calculations is a four layer Feed Foward Neural Network with drop out regularization (~2.2M Parameters) trained to predict whether or not a given run will win. The model has "seen" a total of 77M runs, but only 1.5M runs were used for finetuning. These runs are all at A20 difficulty after watcher was released and do not contain daily runs, beta test, or endless mode.

All data comes from [this data dump](https://www.reddit.com/r/slaythespire/comments/jt5y1w/77_million_runs_an_sts_metrics_dump/)
For more detailed information on the fields and data structure see data_dump.md in the main git directory.

## How to Setup & Run
1. Install supporting programs
    - [Python](https://www.python.org/downloads/)
        - Tested on version 3.12.5, any 3.12.X version should work
        - Check the box to add Python to PATH
    - [Git](https://git-scm.com/downloads)
    - Restart computer after this step
2. Clone git hub repo
    - Navitgage to folder where you want this to be installed
    - Open a terminal (optionally open git bash or git gui, in Windows 11 this can be found by right-click -> show more options)
    - Clone repository using either steps one or two
        1. Typing in the terminal type `git clone https://github.com/jjoba/sts_cockpit.git`
        2. Using git gui choose clone existing repository and give it the url above
    - Note: it is possible to skip the above steps and just download the repo as a zip archive (do this by clicking the <> code button on this page and choosing download as zip), but without the git configuration it's not as straightforward to download new versions, thus is not the recommended option
3. Run the program
    - Windows: double click the run_sts_cockpit_windows.bat file
    - Mac: two options
        1. Open terminal and navigate to sts_cockpit directory, then enter  `sh run_sts_cockpit_mac`
        2. Set run_sts_cockpit_mac.sh as an executable file then double click to run

<b>Additional setup notes:</b>
- The batch and shell scripts automatically manage virtual environment creation, module installation, and booting the CLI in the backround
- For windows and linux, there is an option for pytorch to run on a discrete GPU using Cuda, the version of Pytorch installed with the "run..." scripts is CPU only. This is because 1. Managing various gpu and cuda configurations is quite complex and 2. The cockpit program uses pre-trained models in inference mode which is not meaningfully slower on a CPU. If you're interested in training the models from scratch and you have an Nvidia GPU I'd recommend modifying requirements.txt to match you cuda configuration. This program has been tested on cuda 12.4, but does not work well on cuda 12.6.

## Interacting With STS Cockpit
- On menu screens or prompts with numbered options, the program expects the number followed by enter. For example, if you saw "1. Start a new run" you'd input "1" followed by enter to select this choice
- All card and relic names should be in lowercase with underscores as spaces (e.g. infinite_blades). If that doesn't work try again without the underscores as naming conventions in the data are inconsistent (e.g. justlucky). Watcher tends to have everything smushed into a single word. Eventually, I'll conform all items to follow this format.
- Upgraded cards have an "_u" at the end (e.g. infinite_blades_u)
    - Note: Searing Blow is not handled as it is in the game. Currently, all number of upgrades are grouped into "searing_blow_u" (e.g. an 8 upgrade searing blow is equivalent to a 20 times upgraded one as far as the model is concerned). As a result, searing_blow_u is overvalued and in game upgrades to the card cannot be reflected in STS Cockpit. This will be fixed in a future release, but for now, use this card with caution.
- If you've double checked the spelling and are still getting issues with unknown cards you'll just have to ignore that card for simulations and cannot add it to the deck. This is a known issue as the dataset contains old card names and all have not been fixed yet. If this happens please create an Issue in the github repo including both what you typed in and the card you expected to appear.
- After selecting to start a new run you'll be asked to pick a model
    - A pop up window will appear to select a file. On windows it's often hidden and you'll have to alt-tab into it (look for the feather icon).
    - Navigate to sts_cockpit/model_objects/production. Any of the models here can be used and perform comparably with fnn_v15_small having the slightest edge (see /analyses for more info on model performance)
- Expected character names are ironclad, the_silent, defect, watcher
- Key menu concepts:
    - Simulate: this allows the user to ask "what if" questions w/out modifying their deck in order to make a choice. For example, "which card should I add to my deck?"
    - Update Deck / Relics: this makes permanent changes to your deck and relic set within cockpit. Changes here will be reflected in subsequent simulations. If you input the wrong thing (e.g. accidentally add a_thousand_cuts) it can simply be reversed with no harm done (select remove and pull out a_thousand_cuts). You can manually use this tool to simulate custom deck combinations.
- Understanding the outut:
    - The underlying model is trained to predict the probability of beating the heart on A20 given a certain deck and relic combo.
    - The numbers shown reflect this probability in decimal format (0 = 0% chance of victory, 1 = 100% chance of victory)
    - At the start these numbers will be tiny because the likelihood of beating the heart with only the starting deck and relic is extremely low
    - In simulations it will show a column of numbers followed by a list of cards (working to clean this up in a future update).
        - The first entry will be the current deck and should be considered a baseline (for example if deciding whether to add a card to the deck, but all probabilities are lower than current deck, the model's recommendation would be to skip)
        - The list of win probabilities follow the order of cards shown above. The highest numerical value is the option recommended by the model
        - The model is decent at identifying winning deck/relic combos, but tends to evaluate options at a near 0% chance of winning then suddenly jumping to near 100% when a key card or two is added. As a result, sometimes it struggles to provide guidance on how to build towards a winning combo vs. just telling you when you have one. Very much lean into simulated results as suggestions and use your own expertise for final selections. Plannning to address this shortcoming in future model iterations


## Hardware Requirements To Build
### Must Have
- Storage: at least 233 GB of free space. This accomodates the compressed raw data, uncompressed training data, and pytorch model objects
- RAM: at least 16 GB. Technically it will still run with slightly less, but will trigger paging and slow down dramatically

### Nice to Have:
- Storage: 293 GB of free space to accomodate an extra copy of the data compressed data files just in case something happes you don't have to wait for them to download again. Save these in a different directory than the compressed files being used.
- GPU: this will speed up model training SIGNIFICANTLY. For benchmarking purposes it takes ~50-60 hours on only a 2022 laptop i9 w/16 cores, ~15 hours on a 2024 M3 Pro, and a yet to be benchmarked faster speed on a 3090. If run time becomes an issue consider going into the train_model.py script and reduce the number of epochs for pre training alpha and pre training beta -- optionally, you could eliminate the pre train alpha setp entirely. Training runs on PyTorch and automatically supports CPU, GPU (Cuda), and MPS (Apple).
- More CPU Cores: data processing runs in parallel on all available cores and takes ~ 1 hour on an 12 Core M3 Pro. If going above 12 cores and only have 16 GB of RAM monitor memory usage. Processing will be net slower if the computer needs to create paging files. CPU usage can be changed by setting the num_cores variable in process_data.py to a fixed value.
- SSD: there's quite a bit of file IO that takes place (52K files to read in and ~75K written out before reading back in again and appending to the final csvs used for training). Consequently, an HDD will slow down data processing immensly. No meaningful impact seen to model training time.

## Building STS Cockpit
Note: building from scratch is only needed if interested in training or tweaking the underlying models. Otherwise, use the trained models and CLI.

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
- Properly handle searing blow (currently all quantity of upgrades are treated as a single upgrade, overvaluing the card)
- Create full list of curses
- Standardize naming (some items have space with underscores, some don't)
- Identify the names of the last few missing cards
- Fuzzy matching to support typos in UI

## Test Set Model Performance
Model Version: FNN 15 Small
- Accuracy: 97.99%
    - Test set win rate: 4% (i.e. anything about 96% shows predictive lift)
- Binary Cross Entropy Loss: 0.0437
- Brier Score: 0.0139
- Recall: 68%