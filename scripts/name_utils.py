'''
    Single source of truth for card and relic name cleaning.
    Used by create_master_lists.py and process_data.py so the master lists and the training columns cannot drift apart.
'''

import re

# Old in-game names -> current names (post normalization). Applied after cleaning, so keys are one word with no numbers.
# Eggs are not listed because normalization strips the trailing '2' (e.g. 'Molten Egg 2' -> 'moltenegg').
RENAMES = {
    'underhandedstrike': 'sneakystrike',
    'underhandedstrike_u': 'sneakystrike_u',
    'wireheading': 'foresight',
    'wireheading_u': 'foresight_u',
    'conservebattery': 'chargebattery',
    'conservebattery_u': 'chargebattery_u',
    'lockon': 'bullseye',
    'lockon_u': 'bullseye_u',
    'undo': 'equilibrium',
    'undo_u': 'equilibrium_u',
    'steampower': 'steambarrier',
    'steampower_u': 'steambarrier_u',
}

# Searing Blow's damage grows with every upgrade (it can be upgraded any number of times), so it can't share the single _u column other cards use.
# In the model input a copy is counted in the column of every level up to its own: searingblow = not upgraded, searingblow_u = upgraded
# at least once (what the column meant before), searingblow_u2plus = upgraded at least twice, and so on up to SEARING_BLOW_MAX (which also holds anything higher).
# Counting "at least" means one more upgrade adds one to one column, and what is learned about +1..+n carries over to +n+1
SEARING_BLOW_MAX = 10
SEARING_BLOW_COLUMNS = ['searingblow', 'searingblow_u'] + [f'searingblow_u{n}plus' for n in range(2, SEARING_BLOW_MAX + 1)]

def searing_blow_level(name):
    # Upgrades of a cleaned Searing Blow name, None for any other card. Here searingblow_u3 means EXACTLY +3 (clean_cards output),
    # which is not the searingblow_u3plus column. Exact names only live in the replay, the model only sees SEARING_BLOW_COLUMNS
    match = re.fullmatch(r'searingblow(_u(\d*))?', name)
    if match is None:
        return None
    if match.group(1) is None:
        return 0
    return int(match.group(2)) if match.group(2) else 1

def searing_blow_name(level):
    return 'searingblow' if level == 0 else 'searingblow_u' if level == 1 else f'searingblow_u{level}'

def searing_blow_columns(level):
    # Columns a copy of Searing Blow with this many upgrades is counted in
    level = min(level, SEARING_BLOW_MAX)
    return SEARING_BLOW_COLUMNS[:1] if level == 0 else SEARING_BLOW_COLUMNS[1:level + 1]

def expand_searing_blow(deck):
    # {card: qty} with exact level Searing Blow names -> the same deck in model columns
    expanded = {}
    for name, qty in deck.items():
        level = searing_blow_level(name)
        for column in [name] if level is None else searing_blow_columns(level):
            expanded[column] = expanded.get(column, 0) + qty
    return expanded

def sort_cards(cards):
    # Alphabetical, except the Searing Blow columns which are in level order (searingblow_u10plus would sort before searingblow_u2plus)
    return sorted(set(cards), key = lambda c: ('searingblow', SEARING_BLOW_COLUMNS.index(c)) if c in SEARING_BLOW_COLUMNS else (c, 0))

def clean_cards(master_deck):
    # Searing Blow's upgrade count has to be read before digits are stripped below (searingblow_u3 = +3)
    searing_blow = [re.fullmatch(r'searing[ _]?blow\+(\d+)', card_name, flags = re.IGNORECASE) for card_name in master_deck]

    # Removes starter card suffix (e.g. Strike_R) and digits (combines Fasting with Fasting2)
    cleaned = [re.sub(r'_.{1}|[0-9]', '', card_name).lower() for card_name in master_deck]

    # Spaces, underscores, and apostrophes removed since naming is inconsistent in the raw data
    cleaned = [re.sub(r' |_|\'', '', card_name) for card_name in cleaned]

    # Converts all upgrades to _u, except Searing Blow which keeps its count
    return [searing_blow_name(int(match.group(1))) if match else re.sub(r'\+.*$', '_u', card_name) for card_name, match in zip(cleaned, searing_blow)]

def clean_relics(relics):
    # Mashes relic names into one word, removes numbers (eggs end in ' 2') and apostrophes
    return [re.sub(r' |[0-9]|\'', '', r).lower() for r in relics]
