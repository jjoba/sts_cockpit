# Outstanding experiment ideas (as of 2026-11-02)

Context for a fresh chat: the STS Cockpit FNN predicts Heart-kill probability from deck, relics, floor and character (929 input features). The v21 pipeline uses a 5-head multi-target network (act 1 boss, act 2 boss, act 3 first boss, act 3 second boss at A20 only, Heart kill) with a deterministic `md5(play_id) % 5 == 0` test split. Labels of -1 are ignored in the loss. The Heart head is last. The goal is a model that helps players choose cards and build decks, so correctly capturing card and relic interactions matters, and a small accuracy loss is acceptable in exchange for better interactions.

## Where things stand

Held-out Heart-kill performance (v21, same test set for all; one seed each, seed noise about 0.0002 log loss):

| Model | Log loss | AUC | Avg precision | Train time |
|---|---|---|---|---|
| fnn_v21_ft_st (finetune only, Heart head only) | 0.05485 | 0.9190 | 0.4609 | 52 min |
| fnn_v21_ft_mt (finetune only, all heads) | 0.05430 | 0.9219 | 0.4643 | 82 min |
| fnn_v21_beta_act3_mt (beta pretrain, act-3-boss label only) | 0.05389 | 0.9234 | 0.4696 | 109 min |
| fnn_v21_beta_mt (beta pretrain, all heads) | 0.05361 | 0.9247 | 0.4740 | 110 min |
| fnn_v21_alpha_beta_mt (alpha then beta pretrain, all heads) | 0.05326 | 0.9259 | 0.4797 | 396 min |
| fnn_v22_ft_st_cum (v22: cumulative upgrade encoding) | 0.05455 | 0.9201 | 0.4650 | 89 min |
| fnn_v22_ft_mt_cum | 0.05401 | 0.9227 | 0.4685 | 100 min |
| fnn_v22_beta_act3_mt_cum | 0.05354 | 0.9245 | 0.4747 | 252 min |
| fnn_v22_beta_mt_cum | 0.05330 | 0.9256 | 0.4790 | 282 min |
| fnn_v22_alpha_beta_mt_cum | 0.05301 | 0.9264 | 0.4843 | 551 min |

The v22 rows are the same recipes as the v21 rows above them with `upgrade_encoding=cumulative`. Train times are wall clock and were inflated by other load on the machine (a step benchmark showed the encoding costs about 3%); do not read them as a cost of the encoding. Precision, recall and F1 at 0.5, Brier and calibration error for every model are in `model_objects/bench/*/report_*.md`.

Interaction findings:
- The networks are mostly additive in card effects. Pretraining enlarges the learned pairwise interactions, but they stay small (mostly under 0.1 logit).
- Corruption + Dead Branch is the one large gain: single-target 0.01 vs alpha + beta 0.51 logit.
- Three-card combos (e.g. Barricade + Entrench + Body Slam) get no extra credit in any model, and those decks are still underpredicted by 6-11% on held-out rows.
- On card-pick decisions, the networks beat the additive models on agreement with Heart-killers' picks by 4-6 points (alpha + beta 56.8%, additive 51.0%).

Baselines to beat: see the table above. Do not compare against v20 or earlier models (different split and format).

## Ideas

### Interactions between cards (priority)

1. **Explicit pairwise branch.** Add a factorization-machine-style term, or a small pairwise network, on top of the main network, trained jointly. This is the most direct fix for stacked combos the models miss. It is also where the accuracy-for-interactions trade-off could appear. Evaluate with the combo and card-pick checks below, not log loss alone.
2. **Cumulative upgrade encoding (v22). Done 2026-11-02.** `upgrade_encoding=cumulative` gives a small, consistent gain on every recipe: log loss 0.0003 lower (0.00025-0.00035), AUC +0.0005-0.001, average precision +0.004-0.005, Brier and F1 better, precision/recall/F1 at 0.5 similar or slightly better. It makes rare upgrade states borrow from the plain card: for pairs with a strong interaction, the rarest upgrade state now keeps 106-116% of the interaction seen in the best-supported state (v21: 55-68%). It did not fix the negative upgrade effects (share of cards where upgrading looks harmful: 12-18% before and after, slightly worse for the beta and alpha+beta models), and combo residuals are unchanged (Barricade x Entrench train residual 0.41 to 0.36 multi-target, 0.46 to 0.44 single target; Barricade + Entrench + Body Slam and Corruption + Dead Branch still undervalued). Keep it as the default for new models. Second seed not run. Reports: `model_objects/bench/v22_*/`.
3. **Combo-targeted loss or data weighting.** Upweight rows that contain known combos, or build synthetic pairs. Untested guess: the loss gain from any single combo is too small to matter in training.

### Labels and heads

4. **Head-weight tuning.** Try a lower weight on the act 1 head, or drop it. Its card preferences were unrelated to the Heart head (r = 0.00).
5. **Late-floor row weighting.** Weight rows at floors 41+, where the finetune-only multi-target model was worse than single-target. Pretraining mostly fixed this, so this is lower priority.
6. **Ruby, emerald and sapphire keys.** Deferred by the user. Only the Ruby key (campfire RECALL) is logged, so the other two cannot be built from the current data.

### Training recipe

7. **More seeds.** Run a second seed of fnn_v21_beta_mt and fnn_v21_alpha_beta_mt. The gaps between the pretrained variants (0.0003-0.0004 log loss) are close to the noise level.
8. **Shorter alpha schedule.** The alpha run took 396 minutes for about 0.0004 log loss. Try 1-2 alpha epochs or a subsample to see how much of that gain is cheap to keep.

### Evaluation

9. **Broader interaction scan.** Run the top-20-cards-per-character pair scan on all five v21 models, and add significance tests on the card-pick differences. Both were skipped in the last round.
10. **Why act 1 card rankings disagree with the other acts.** The act 1 head's card preferences were unrelated to the Heart head (r = 0.00), and the question is whether that is a real difference or an artifact. Candidate explanations to separate: (a) real, act 1 rewards early damage and cheap cards that do not matter later; (b) selection, since runs that die in act 1 never reach later floors, so the act 1 label is measured on a different population (and at A20 the other heads have far fewer labels); (c) label leakage from deck-at-floor, since the decks seen with an act 1 label are mostly early-floor decks; (d) a head-specific artifact such as the shared trunk giving the act 1 head the card weights it needs for early decks. Start by correlating per-card effects (add card minus baseline, held-out decks) between every pair of heads, split by floor range and ascension, and compare with a model trained on act 1 label only. Use the existing bench (`scripts/interaction_bench.py`) with the head index changed from the last output logit. Informs item 4.
11. **Prior that upgrading never hurts.** In the v22 bench 12-18% of well-supported cards and 33-36% of rare-state cards have a negative model-implied upgrade effect (e.g. Lessons Learned -0.54, Metamorphosis -0.30, Unload -0.22), though nearly every upgrade is strictly better in the game. Hypothesis (user's): the model confounds upgrade choices with card quality. Players upgrade the wrong card, or upgrade it in runs that are already struggling or already doomed, so the upgraded copy's column soaks up that selection effect and reads as "this upgraded card is worse". Ways to impose the prior: (a) cumulative encoding with a non-negativity constraint on the `card_u` weights' effect (e.g. softplus on the `card_u` contribution, or a hinge penalty when the model's upgrade effect on a held-out batch goes below zero); (b) a regularizer that pulls `card_u` weights toward zero so the upgraded copy defaults to the plain card (cumulative encoding already does this in effect, but nothing pushes the sign); (c) a monotonic penalty computed on training batches by flipping each upgraded card to plain and requiring the logit to drop. Evaluate: share of cards with negative upgrade effect (target ~0), held-out log loss (expect a small loss if the confounding is real and the model was using it), and calibration of rare upgrade states. Also check whether the negative effects concentrate in cards that are mostly upgraded late or in struggling runs: compare floor and prior-win-rate of runs that upgrade each flagged card. Uses `upgrade_effects.csv` from the bench.
12. **Search the data for confounders.** Look for variables outside the deck itself that explain both which cards get chosen or upgraded and whether the run wins, so they can be added as inputs, used for stratification, or used to reweight rows. The aim is to remove spurious correlations the model learns, such as "upgrading this card is bad" (item 11) or card preferences that differ by act (item 10). Candidates to check, and what the data currently has:
    - **Player id.** The raw run events may carry a player identifier; so far only `play_id` (a run id, used for the hash split) is used. Strong players both win more and make different card and upgrade choices (they skip cards, upgrade selectively), so card effects absorb player skill. Check whether a player id exists, how many runs per player, and how much of the win rate is explained by player alone (a player-only baseline's log loss and AUC).
    - **Run metadata:** ascension level (already pooled across ascensions, so check it is an input or at least stratified on), game version and date (a card's strength changes between patches; the data is filtered to runs after 2020-07-01, `timestamp` at `scripts/process_data.py:156`), chosen seed (excluded), and whether the run is a daily or custom mode run.
    - **Upgrade source and timing:** whether a card was upgraded at a campfire, by an event, or from a relic (Whetstone, War Paint), the floor it was upgraded at, and what else the player could have done at that campfire (rest, lift, dig). Upgrading while at low HP instead of resting, or upgrading late in an already-doomed run, would read as "upgrading hurts".
    - **State not in the 929 features:** current and max HP, gold, potions, and Ruby key. These drive both choices and survival and are absent from the input vector.
    - **Survivorship by floor:** which rows each upgrade state appears in (deck size and floor distribution of plain versus upgraded copies of the same card).
    Method: for each candidate, (a) check whether it differs between plain-card and upgraded-card rows for the cards flagged in item 11, (b) re-fit the offset regression from the bench with the candidate as a control and see how much the upgrade effect moves, and (c) retrain with the candidate as an input or as a stratification weight and compare the upgrade effects, held-out log loss and calibration. If a confounder explains the negative effects, adding it as an input (and setting it to a neutral value at the cockpit's prediction time) is the cleanest fix, and it would make item 11's constraint unnecessary.

## Suggested order

11 and 12 first (confounders and an upgrade prior, since negative upgrade effects survived cumulative encoding), then 1, then 7 together with 8. Item 2 is done.

## How to compare

- Heart-head log loss, AUC and average precision on the hash-split held-out set (see the table above), plus the per-segment breakdown in each model's `segments.csv`.
- Counterfactual pair and triple interactions on curated combos (Barricade + Entrench, Barricade + Body Slam, Corruption + Feel No Pain, Catalyst + Noxious Fumes, Corruption + Dead Branch, Barricade + Entrench + Body Slam).
- Observed vs expected wins on decks containing each combo, with run-clustered z-scores.
- Card-pick calibration (per-card observed vs expected over 173 cards with 1,000+ picks) and agreement with Heart-killers' picks.
- Do not run two experiments at once with `log_experiment=true`.
