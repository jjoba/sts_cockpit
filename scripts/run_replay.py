'''
    Rebuilds the deck and relics a player held at any floor of a run by replaying the event logs in the raw run json.
    The raw data only stores the FINAL deck/relics, so the replayed end state is compared to it. The size of the
    mismatch (residual) says how much of the run's deck history was not captured by the events replayed here.
'''

from collections import Counter

from name_utils import clean_cards, clean_relics

# Cleaned card names, Ascender's Bane is added separately at A10+
STARTING_DECKS = {
    'ironclad': {'strike': 5, 'defend': 4, 'bash': 1},
    'the_silent': {'strike': 5, 'defend': 5, 'survivor': 1, 'neutralize': 1},
    'defect': {'strike': 4, 'defend': 4, 'zap': 1, 'dualcast': 1},
    'watcher': {'strike': 4, 'defend': 4, 'eruption': 1, 'vigilance': 1},
}

# Boss relic picks have no floor in the data, these are the floors right after the act 1-3 bosses
BOSS_RELIC_FLOORS = [17, 34, 51]

# Order of operations when several things happen on the same floor
ADD, REMOVE, UPGRADE = 0, 1, 2

def get_card_events(event, card_set):
    # Returns list of (floor, order, op, cleaned card name). Only names in the master card list are kept
    events = []

    def collect(floor, op, raw_names):
        for name in clean_cards(raw_names or []):
            if op == UPGRADE:
                name = name[:-2] if name.endswith('_u') else name
            if name in card_set:
                events.append((floor, op, op, name))

    # Card rewards (floor 0 is the Neow choice). Skips, Singing Bowl, etc. fall out of the card_set check
    for c in event.get('card_choices', []):
        collect(c['floor'], ADD, [c.get('picked', 'SKIP')])

    for item, floor in zip(event.get('items_purchased', []), event.get('item_purchase_floors', [])):
        collect(floor, ADD, [item])

    for item, floor in zip(event.get('items_purged', []), event.get('items_purged_floors', [])):
        collect(floor, REMOVE, [item])

    for c in event.get('campfire_choices', []):
        if c.get('key') == 'SMITH':
            collect(c['floor'], UPGRADE, [c.get('data')])

    # Event field names other than cards_obtained follow the dump's naming, absent keys are simply skipped
    for e in event.get('event_choices', []):
        collect(e['floor'], ADD, e.get('cards_obtained'))
        collect(e['floor'], REMOVE, e.get('cards_removed'))
        collect(e['floor'], REMOVE, e.get('cards_transformed'))
        collect(e['floor'], UPGRADE, e.get('cards_upgraded'))

    # sorted() is stable so same floor, same op events stay in log order
    return sorted(events, key = lambda x: (x[0], x[1]))

def get_relic_floors(event, relic_set):
    # Returns list of (floor acquired, cleaned relic name) for every relic in the final relic list
    candidates = []

    for r in event.get('relics_obtained', []):
        candidates.append((r['floor'], clean_relics([r['key']])[0]))

    for e in event.get('event_choices', []):
        for r in e.get('relics_obtained', []) or []:
            candidates.append((e['floor'], clean_relics([r])[0]))

    for item, floor in zip(event.get('items_purchased', []), event.get('item_purchase_floors', [])):
        candidates.append((floor, clean_relics([item])[0]))

    boss_picks = clean_relics([b['picked'] for b in event.get('boss_relics', []) if b.get('picked')])

    relic_floors = []
    for name in clean_relics(event['relics']):
        if name not in relic_set:
            continue

        match = next((c for c in sorted(candidates) if c[1] == name), None)

        if match is not None:
            candidates.remove(match)
            relic_floors.append(match)

        # Not logged with a floor: boss relics get their boss floor, starting relic / Neow relic are floor 0
        elif name in boss_picks:
            relic_floors.append((BOSS_RELIC_FLOORS[min(boss_picks.index(name), 2)], name))
        else:
            relic_floors.append((0, name))

    return relic_floors

def apply_event(deck, op, name):
    if op == ADD:
        deck[name] += 1
    elif op == REMOVE:
        if deck[name] > 0:
            deck[name] -= 1
    # Searing Blow upgrades are all collapsed into _u, so a 2nd upgrade finds no base card and is a no-op
    elif deck[name] > 0:
        deck[name] -= 1
        deck[name + '_u'] += 1

def replay_run(event, card_set, relic_set, n_floors, rng):
    '''
        Returns None if the run cannot be replayed, otherwise (rows, residual) where
        rows = list of (floor, {card: qty}, [relics]) for up to n_floors floors sampled from 1..floor_reached
        Each row is the state on arriving at that floor (events logged on floors < floor applied)
        residual = number of cards the replayed final deck differs from the logged final deck by
    '''
    character = str(event.get('character_chosen', '')).lower()
    floor_reached = event.get('floor_reached')

    if character not in STARTING_DECKS or not isinstance(floor_reached, int) or floor_reached < 1:
        return None

    deck = Counter(STARTING_DECKS[character])
    if event.get('ascension_level', 0) >= 10:
        deck['ascendersbane'] += 1

    card_events = get_card_events(event, card_set)
    relic_floors = get_relic_floors(event, relic_set)

    sampled_floors = sorted(rng.sample(range(1, floor_reached + 1), min(n_floors, floor_reached)))

    rows = []
    i = 0
    for floor in sampled_floors:
        while i < len(card_events) and card_events[i][0] < floor:
            apply_event(deck, card_events[i][2], card_events[i][3])
            i += 1

        rows.append((floor, {k: v for k, v in deck.items() if v > 0}, [name for f, name in relic_floors if f < floor]))

    # Finish replay and compare to the logged final deck
    while i < len(card_events):
        apply_event(deck, card_events[i][2], card_events[i][3])
        i += 1

    final_deck = Counter(n for n in clean_cards(event['master_deck']) if n in card_set)
    residual = sum(abs(deck[k] - final_deck[k]) for k in set(deck) | set(final_deck))

    return rows, residual
