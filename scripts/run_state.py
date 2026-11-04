'''
    Per row state that the cards and relics do not carry, rebuilt from the per floor lists of the raw run json: HP, max HP and gold on arriving at a floor.
    The per floor lists hold the value at the END of each floor, so arriving at floor f is entry f - 2 (floor 1 is the start of the run).
    Values are stored as int8, the model rescales them (see EXTRA_SCALES in train_model.py).
'''

# Columns written after the 929 feature columns, in this order
EXTRA_COLS = ['hp_pct', 'max_hp', 'gold_25']
EXTRA_GROUPS = {'hp': ['hp_pct', 'max_hp'], 'gold': ['gold_25']}

# Used when a run has no list for the value (none of the finetuning or beta runs miss them, this is a guard): about the median of finetuning rows
NEUTRAL = {'hp_pct': 70, 'max_hp': 75, 'gold_25': 5}

# Start of a run at A20: full health, 99 gold
START_GOLD = 99

def at_floor(values, floor):
    # Value on arriving at floor, None if the list does not reach it
    if not isinstance(values, list) or floor < 2 or len(values) < floor - 1:
        return None
    v = values[floor - 2]
    return v if isinstance(v, (int, float)) else None

def run_state(event, floor):
    # Returns {column: int} for EXTRA_COLS. Edited values are clipped, runs with them are moved out of finetuning anyway (process_data.is_modded)
    max_hps = event.get('max_hp_per_floor')
    max_hp = at_floor(max_hps, floor)
    hp = at_floor(event.get('current_hp_per_floor'), floor)
    gold = at_floor(event.get('gold_per_floor'), floor)

    if floor == 1:
        max_hp = max_hps[0] if isinstance(max_hps, list) and max_hps and isinstance(max_hps[0], (int, float)) else None
        hp = max_hp
        gold = START_GOLD

    out = dict(NEUTRAL)
    if max_hp is not None and max_hp > 0:
        out['max_hp'] = int(min(max(max_hp, 1), 127))
        if hp is not None:
            out['hp_pct'] = int(round(100 * min(max(hp, 0), max_hp) / max_hp))
    if gold is not None:
        out['gold_25'] = int(min(max(gold, 0), 3000) // 25)
    return out
