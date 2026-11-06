# UI to-do: how the simulation tool should recommend cards

Decided 2026-11-04 from the head trade-off analysis (`scripts/head_tradeoff.py`, `scripts/head_effects.py`; outputs in `model_objects/bench/head_tradeoff_v24/` and `head_effects_v24/`; write-up in `outstanding_experiments.md`, item 10). Pick this up the next time the UI is updated.

## 1. Rank options by a blend, not by one head

For each option (skip, or each offered card), score the deck after the change with the five-head model and rank by

    score = 0.75 * log P(nearest boss) + 0.25 * log P(Heart)

- **Nearest boss** = act 1 head while the floor is 1 to 16, act 2 head for 17 to 33. In act 3 use the first act 3 boss head (first boss of act 3).
- The same blend, with `P(Heart)` replaced by the A20 act 3 second boss head (run win), is a reasonable alternative: it picks the same card 86 to 89% of the time and avoids the Heart label's dependence on taking the keys. Use Heart if the model has no usable second boss head.
- Model-based test on act 1 rewards: the blend gives 39.0% predicted act 1 boss survival, 1.98% run win and 0.86% Heart. Compare "Heart only" (37.4 / 2.00 / 0.89) and "act 1 only" (39.5 / 1.80 / 0.76). It gives back about a quarter of the act 1 cost and keeps about 90% of the Heart gain.
- Do not use a floor schedule or a threshold on P(boss) to change the weight. Both were tested and land on the same frontier as the fixed 0.25, with worse act 1 survival.
- Heart only models (`*_st`, v14 to v20) have no nearest boss head. For these keep the current Heart ranking.

## 2. Flag cards where the heads disagree

Show a small marker on an option when the near-term boss effect and the Heart effect have opposite signs, with the two numbers in the tooltip (for example "Corruption: -2.1 pts vs act 1 boss, +0.4 pts Heart").

- Compute the effect as the change in each head's probability from adding the card (plain copy) to the current deck at the current floor; `all_logits` in `scripts/head_effects.py` shows the forward pass of all five heads.
- Suggested rule: flag when the sign differs and both effects are at least 0.5 points. Tune the threshold on real decks.
- Aim of the flag: the card is a long-run pick that costs survival now (Corruption in act 1 without scaling, Barricade without block, and similar). Show it as a trade-off, not as a "don't take it".
- Order of the displayed list is still the blend in 1; the flag is for explanation.

## 3. Caveats to show next to the advice

- Effects are model estimates from observational runs (skill and selection are not controlled, and no player id is available at inference).
- Three-card synergies are undervalued on every head, so a combo the model rates lukewarm can be better than shown (Corruption + Dead Branch is the clearest case).
- Late-act heads are noisy; the 25% weight on the Heart head is the hedge against that.
