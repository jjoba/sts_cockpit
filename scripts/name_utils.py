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

def clean_cards(master_deck):
    # Removes starter card suffix (e.g. Strike_R) and digits (combines Fasting with Fasting2)
    master_deck = [re.sub(r'_.{1}|[0-9]', '', card_name).lower() for card_name in master_deck]

    # Spaces, underscores, and apostrophes removed since naming is inconsistent in the raw data
    master_deck = [re.sub(r' |_|\'', '', card_name) for card_name in master_deck]

    # Converts all upgrades to _u, clobbers all searing blow upgrades into 1
    return [re.sub(r'\+.*$', '_u', card_name) for card_name in master_deck]

def clean_relics(relics):
    # Mashes relic names into one word, removes numbers (eggs end in ' 2') and apostrophes
    return [re.sub(r' |[0-9]|\'', '', r).lower() for r in relics]
