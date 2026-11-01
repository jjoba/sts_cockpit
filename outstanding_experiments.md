# Outstanding experiment ideas (as of 2026-11-01)

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

Interaction findings:
- The networks are mostly additive in card effects. Pretraining enlarges the learned pairwise interactions, but they stay small (mostly under 0.1 logit).
- Corruption + Dead Branch is the one large gain: single-target 0.01 vs alpha + beta 0.51 logit.
- Three-card combos (e.g. Barricade + Entrench + Body Slam) get no extra credit in any model, and those decks are still underpredicted by 6-11% on held-out rows.
- On card-pick decisions, the networks beat the additive models on agreement with Heart-killers' picks by 4-6 points (alpha + beta 56.8%, additive 51.0%).

Baselines to beat: see the table above. Do not compare against v20 or earlier models (different split and format).

## Ideas

### Interactions between cards (priority)

1. **Explicit pairwise branch.** Add a factorization-machine-style term, or a small pairwise network, on top of the main network, trained jointly. This is the most direct fix for stacked combos the models miss. It is also where the accuracy-for-interactions trade-off could appear. Evaluate with the combo and card-pick checks below, not log loss alone.
2. **Cumulative upgrade encoding (v22).** Upgraded cards currently use separate `_u` columns; Searing Blow already uses cumulative columns. Try `[1,1]` for an upgraded card instead of `[0,1]`. The option `upgrade_encoding=cumulative` is already built in `scripts/train_model.py` (default `split`). Full command and baselines are in `planned_experiments.md`. Run finetune-only single-target first against fnn_v21_ft_st, then multi-target against fnn_v21_ft_mt. Earlier tests of this idea were a wash, so expect a small effect. Check that the Barricade x Entrench residual on train rows (+0.41 single target, +0.43 multi-target) shrinks.
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

## Suggested order

1, then 2, then 7 together with 8.

## How to compare

- Heart-head log loss, AUC and average precision on the hash-split held-out set (see the table above), plus the per-segment breakdown in each model's `segments.csv`.
- Counterfactual pair and triple interactions on curated combos (Barricade + Entrench, Barricade + Body Slam, Corruption + Feel No Pain, Catalyst + Noxious Fumes, Corruption + Dead Branch, Barricade + Entrench + Body Slam).
- Observed vs expected wins on decks containing each combo, with run-clustered z-scores.
- Card-pick calibration (per-card observed vs expected over 173 cards with 1,000+ picks) and agreement with Heart-killers' picks.
- Do not run two experiments at once with `log_experiment=true`.
