# Planned experiments

## v22: cumulative upgrade encoding (like Searing Blow)
Logged 2026-10-31. **Run 2026-11-02** for all five v21 recipes (fnn_v22_ft_st_cum, ft_mt_cum, beta_act3_mt_cum, beta_mt_cum, alpha_beta_mt_cum); results and the comparison against v21 are in `outstanding_experiments.md` item 2 and `model_objects/bench/v22_*/report_*.md`. Small accuracy gain, better transfer to rare upgrade states, no fix for negative upgrade effects or combos. The text below is the original plan.

**Idea.** Today a plain card is [card=1, card_u=0] and an upgraded one is [card=0, card_u=1], so every upgrade state learns separately. Make an upgraded copy count in both columns ([1, 1]): `card` = total copies, `card_u` = how many are upgraded. This is what Searing Blow's columns already do (see name_utils.py). Pair and combo effects learned for a card then carry over to its upgraded version, and `card_u` only has to learn what upgrading adds.

**Already built (tested, default unchanged).** `upgrade_encoding=cumulative` in scripts/train_model.py. The model adds the `card_u` columns onto their `card` columns as its first step (350 card pairs, Searing Blow left out), so the data files and cockpit inputs do not change and no data rebuild is needed. Default is `split`, so existing runs are unaffected.

**Why it might help.**
- The first layer is linear, so the two encodings can represent the same functions. The difference is learning dynamics: copies in any upgrade state train the same `card` weights.
- 90 of 350 cards have an upgrade state (plain or upgraded) with under 5,000 finetuning training rows, 25 have under 1,000.
- In the v21 finetune-only networks, the first-layer weights of a card and its upgraded version are barely related (cosine 0.14 single target, 0.16 multi-target, vs 0.00 for random pairs; 0.06-0.07 for the cards with the rarest upgrade states).
- For Ironclad Barricade + Entrench + Body Slam (floor 20+, train) the 5,315 rows holding all three split into 8 upgrade cells of 333-1,865 rows.

**Evidence so far (weak).** Pairwise factorization model on held-out finetuning test, rank 8, log loss: split 0.05538 / 0.05541 (lambda 3e-5 / 1e-4), upgrade state merged for pairs 0.05522 / 0.05529, cumulative 0.05526 / 0.05522. Differences of 0.0001-0.0002 are about seed noise (~0.0002).

**To run (same settings as the controls, one change).**
```
python scripts/train_model.py --set model_version=fnn_v22_ft_st_cum stages=finetuning 'head_weights=[0,0,0,0,1]' upgrade_encoding=cumulative dropout=0.1 lr=0.03 lr_schedule=cosine lr_final_factor=0.01 batch_size=64 select_metric=log_loss
```
Then the multi-target version (drop the head_weights override) against fnn_v21_ft_mt.

**Compare against (v21, same data and hash split).**
- fnn_v21_ft_st: log loss 0.05485, AUC 0.9190, avg precision 0.4609.
- fnn_v21_ft_mt: log loss 0.05430, AUC 0.9219, avg precision 0.4643.
- Gain needs to beat seed noise (~0.0002), so run a second seed before believing a small win.
- Combo check: Barricade x Entrench residual on training rows (offset regression with the model's logits): +0.41 single target, +0.43 multi-target. Cumulative should shrink it. Also the card-pick agreement and per-card calibration checks on held-out runs.
- Do not use the log_experiment=true option for two runs at once, the experiments file is not safe to append to in parallel.
