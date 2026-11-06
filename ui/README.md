# STS Cockpit UI

A browser version of the cockpit that runs on the real models. Everything the CLI does (add, remove and upgrade cards, relics, floor, card and relic simulations, shop) plus a model picker, a model directory with held-out metrics, a visual deck, and relic planning for Empty Cage, Peace Pipe, Astrolabe and Pandora's Box.

## Run it

From the project root:

```bash
python ui/server/app.py
```

then open http://127.0.0.1:8765 (or use `run_sts_cockpit_ui_mac.sh` / `run_sts_cockpit_ui_windows.bat`, which set up the virtual environment first). It needs no packages beyond `requirements.txt`. Pass `port=9000` to change the port.

## Layout

| Path | What it is |
|---|---|
| `server/app.py` | Small standard library HTTP server. Serves `web/` and two endpoints: `GET /api/catalog`, `POST /api/predict` |
| `server/engine.py` | Loads the models, builds the 929 (or 920) input columns, runs batches. Output matches the saved TorchScript to 6 digits |
| `server/build_catalog.py` | Writes `server/data/cards.json` and `relics.json` (display name, model column, owner, type) and checks them against the model columns |
| `server/build_legacy_layouts.py` | Rebuilds the input layout of v14 to v16 from git history and picks the best fitting one per model (`data/legacy_layouts.json`) |
| `server/build_metrics.py` | Scores every v21+ model on one test set (`data/training_data_v24`, modded runs removed) and writes `server/data/metrics.json`, plus `start_run_state.json` (usual max HP at the start of a run) |
| `web/index.html` | The app (one file, no build step) |
| `mockups/index.html` | The original clickable mockup with fake numbers, kept for reference |

Rebuild the data files after changing the card or relic lists, or after benching a new model:

```bash
python ui/server/build_catalog.py
python ui/server/build_metrics.py
python ui/server/build_legacy_layouts.py   # only if the old models or their data change
```

## How models are run

- Every v20 and later model takes the 929 columns of `starting_decks.csv` (cards with a `_u` column for upgraded copies, relics, floor, four one-hot characters). The cumulative upgrade models (v22 `*_cum`) fold upgraded copies onto the plain column inside the network, so the input is the same split encoding for every model.
- v24 and later models also take run state after those 929 columns: HP as a percentage of max HP, max HP and gold in units of 25 (`scripts/run_state.py`). Which of them a model takes is in `experiments.csv` (`extra_features`); the integers go in as the training files hold them and each network rescales them with its own `input_scale` buffer. The Run page has HP, max HP and gold fields for these models, and the shop scores a basket with the gold you would have left after paying. A model that takes them but gets none falls back on the training data's neutral values, which costs it about 0.0008 log loss.
- A head with weight 0 in `head_weights` was never trained (for example the act 1 head of `fnn_v25_ft_mt_hp_gold_no_act1`), so it is not shown. Single target `*_st` models only have a Heart head.
- v17 to v19 take the same columns without the nine extra Searing Blow level columns (920).
- v14 to v16 take 863 columns (679 cards, 180 relics, the four characters, no floor). They were trained before per-floor rows on final decks, with the label "the run won" (any victory), so the UI calls their output a win probability, not a Heart kill, and the floor has no effect on them. Their column layout is rebuilt from git history by `build_legacy_layouts.py`: the card lists were regenerated and renamed to one word names on 2024-09-08, which moved 68 columns, and the script scores each model under each candidate list on held-out rows from floor 40 on and keeps the better fit (v14, v15, v15_small and v15_large fit the August underscore list by about 0.03 AUC, v16 fits the September one word list by about 0.02, matching the dates of the files). Old names such as `underhanded_strike` are mapped to the current columns. Their outputs match the saved TorchScript to about 1e-7 on real rows.
- The forward pass is rebuilt from each model's weights, as `scripts/interaction_bench.py` does. The saved TorchScript of the cumulative models does not run reliably across devices.
- v21 and later have five heads (act 1, act 2, act 3 boss, act 3 second boss at A20, Heart kill). Single target models (`*_st`) trained only the Heart head, so only Heart is shown for them. Earlier models have the Heart head only.
- Every v21 and later model is scored on the same cleaned test set, so their metrics are comparable; v21 and v22 differ slightly from older write-ups, which used the test set before modded runs were removed. The old models have no metrics (a different split and label). The model directory hides experiment variants (batch size, learning rate, flip sampling, head tests) until you tick the box; the main recipes are listed in `FEATURED` in `server/app.py`.
- The default model is the highest version number with the best AUC among models that have held-out metrics.

## Planned

- The Event tab: type an event name (37 events from Act 1 to 3, in `ui/server/data/events.json`, written by `python ui/server/build_events.py`) and its options appear, each scored with the blend and the Heart change. An option's effects (gold, HP, max HP, cards, curses, relics, removals, upgrades, transforms, random results, card rewards) are applied to the deck and scored together; choices are made by the model and random results are averaged over 60 seeded draws. Options needing a relic (Golden Idol, Red Mask) are greyed out without it. Fights, potions, mini games and other things the model cannot predict are called out in amber, and events with nothing to predict say so. Values are entered by hand from the game (Ascension 15+ numbers), so check them; add or fix an event in `build_events.py` (the effect keys are listed at the top of that file). Do it applies an option to the run and asks for random results.
- The Campfire tab offers only what the relics in the inventory allow: Rest (30% of max HP, +15 with Regal Pillow, blocked by Coffee Dripper; the heal is scored only by models that read HP; with Dream Catcher the expected best-of-3 card reward is added: every pool card is scored after the heal and weighted by the chance it is the best of the cards offered or a skip: 3 cards, 2 fewer with Busted Crown (so 1), +1 with Question Card; rarity ignored), Smith (best upgrade, blocked by Fusion Hammer), Toke (Peace Pipe, best card to remove), Dig (Shovel, average over a sample of relics) and Lift (Girya, listed but not scored because Strength is not an input).
- Saves: the 💾 Saves button on the Run page stores the character, deck, relics, floor, HP, gold, shop stock and model under a name, and lists, loads, overwrites and deletes saves. Files go to `ui/server/data/saves/` (gitignored, one JSON each), via `GET/POST /api/saves` and `POST /api/saves/delete`.
- The shop stock accepts upgraded cards (an "upgraded" tick box on each card row, or type "Carnage+"; the shop sells them when you hold an egg relic) and the card removal slot suggests the card whose removal scores best (marked "best" in the list, with a Use button after you pick another one).
- Remove and Transform take a number of cards. Remove tries every combination (a greedy pick above 1,500 combinations) and recommends the best set; Transform picks greedily and scores the set with Monte Carlo draws (stops at ±0.1 pts, 1,000 draws or 2 s). Both have a button that applies the set; Transform asks what you received.
- The What if panel has a Transform tab: each card that can be transformed is ranked by the average result of turning one copy into a random pool card (the same fixed sample of up to 60 cards as the relic planner), optionally upgraded. Transform then asks which card you received and updates the deck.
- Reward options can be upgraded cards: type "Armaments+" in the Add card tab ("Searing Blow+3" for a Searing Blow upgraded three times). The row is scored as the upgraded copy and Add puts the upgraded copy in the deck. The popup, Next best and the shop stay plain-only.
- Card rewards (Add card tab, Next best, Add cards popup) are ranked by 0.75 x log P(nearest boss) + 0.25 x log P(Heart) (act 1 head to floor 16, act 2 to 33, act 3 after; Heart-only models and the floor past the last boss keep the Heart ranking). A card is marked "trade-off" when the nearest boss and the Heart move in opposite directions by at least 0.5 points each. The numbers shown stay the Heart change. Decided 2026-11-04 from `scripts/head_tradeoff.py` and `scripts/head_effects.py` (`outstanding_experiments.md`, item 10); the thresholds in `index.html` (`BLEND`, `FLAG_PTS`) still need tuning on real decks.

## Things to know

- **Boss heads at later floors.** For a near-starter deck at floor 17 or later the act 2 and act 3 boss outputs are close to 100%, and the Heart output stays low. The server reproduces the saved models exactly, so this is the models' behaviour on an input they rarely saw. A boss you are already past is shown as 100% regardless.
- **Names.** The UI uses current in-game names and maps them to the dataset's columns. Owners and types are hand written in `build_catalog.py`; the data has neither. 26 old or beta card columns have no current name and are hidden. Steam Barrier is split across two columns in the data (`steambarrier` and the old `steam`); the UI uses `steambarrier`.
- **Transform relics** (Astrolabe, Pandora's Box) are scored by sampling the cards you could get (uniform over your character's pool, repeats allowed, never the same card). They stop at ±0.1 points, 2,000 draws or 2 seconds. Astrolabe's three picks are the three best cards ranked one at a time, then scored together. Rarity weighting and locked cards are not modeled.
- **Shop and what-ifs** score whole decks, so interactions between purchases count.
