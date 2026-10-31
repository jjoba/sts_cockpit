'''
    Rebuilds the deck and relics a player held at any floor of a run by replaying the event logs in the raw run json.
    The raw data only stores the FINAL deck/relics, so the replayed end state is compared to it. The size of the
    mismatch (residual) says how much of the run's deck history was not captured by the events replayed here.
'''

from collections import Counter

from name_utils import clean_cards, clean_relics, expand_searing_blow, searing_blow_level, searing_blow_name

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

def in_card_set(name, card_set):
    # The master card list has Searing Blow's model columns, the replay tracks its exact upgrade level (see name_utils)
    return name in card_set or searing_blow_level(name) is not None

def upgraded(name):
    level = searing_blow_level(name)
    return name + '_u' if level is None else searing_blow_name(level + 1)

def get_card_events(event, card_set):
    # Returns list of (floor, order, op, cleaned card name). Only names in the master card list are kept
    # Upgrades are logged with the name of the card BEFORE the upgrade, which for Searing Blow includes its level (Searing Blow+2 becomes +3)
    events = []

    def collect(floor, op, raw_names):
        for name in clean_cards(raw_names or []):
            if op == UPGRADE and searing_blow_level(name) is None:
                name = name[:-2] if name.endswith('_u') else name
            if in_card_set(name, card_set):
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
    else:
        if deck[name] == 0 and searing_blow_level(name) is not None:
            # Only campfires log Searing Blow with its upgrades, events log it as plain Searing Blow whatever its level, so the highest one is upgraded
            levels = [searing_blow_level(k) for k in deck if deck[k] > 0 and searing_blow_level(k) is not None]
            name = searing_blow_name(max(levels)) if levels else name

        if deck[name] > 0:
            deck[name] -= 1
            deck[upgraded(name)] += 1

CURSES = {'ascendersbane', 'curseofthebell', 'necronomicurse', 'regret', 'doubt', 'shame', 'writhe', 'parasite', 'injury',
          'decay', 'normality', 'pain', 'clumsy', 'pride'}

# Neow bonus -> (removals, transforms, upgrades). card_choices has Neow's card picks but not which card was removed / transformed / upgraded
NEOW_BONUS = {
    'REMOVE_CARD': (1, 0, 0), 'REMOVE_TWO': (2, 0, 0),
    'TRANSFORM_CARD': (0, 1, 0), 'TRANSFORM_TWO_CARDS': (0, 2, 0),
    'UPGRADE_CARD': (0, 0, 1),
}

# Relics that change the deck without logging which cards: (removals, transforms, upgrades, curse gained), applied on the floor the relic was obtained
# Pandora's Box transforms every Strike / Defend so its transform count is whatever the final deck implies
RELIC_EFFECTS = {
    'emptycage': (2, 0, 0, None),
    'astrolabe': (0, 3, 0, None),
    'pandorasbox': (0, 'starters', 0, None),
    'tinyhouse': (0, 0, 1, None),
    'whetstone': (0, 0, 2, None),
    'warpaint': (0, 0, 2, None),
    'callingbell': (0, 0, 0, 'curseofthebell'),
    'necronomicon': (0, 0, 0, 'necronomicurse'),
}

# Cards that upgrade other cards mid combat, the floor of the upgrade is not logged
COMBAT_UPGRADERS = {'armaments', 'apotheosis', 'lessonlearned'}

STARTER_NAMES = {'strike', 'strike_u', 'defend', 'defend_u'}

def is_strike(name):
    # Vampires removes every card tagged Strike, only the Heavy Blade has the tag without having it in its name
    return 'strike' in name or 'heavyblade' in name

def upgrade_target(name, under):
    # Returns (card, upgrades) an unlogged upgrade of name ends up as in the logged deck, or None. A Searing Blow may have been upgraded
    # several times (e.g. Armaments), so any higher level the logged deck has too many of counts
    level = searing_blow_level(name)

    if level is None:
        return (name + '_u', 1) if under[name + '_u'] > 0 else None

    higher = [searing_blow_level(k) for k in under if under[k] > 0 and (searing_blow_level(k) or 0) > level]
    return (searing_blow_name(min(higher)), min(higher) - level) if higher else None

def infer_unlogged(event, start_deck, card_events, relic_floors, floor_reached, final_deck):
    '''
        The logs do not say which card Neow removed / transformed / upgraded, nor what some relics and in combat upgrades did.
        Whatever the replayed end deck has too many / too few of, compared to the logged final deck, is attributed to those
        sources in order, with the event placed on the source's own floor.
        Returns (extra card events, number of cards still unexplained)
    '''
    deck = Counter(start_deck)
    for floor, order, op, name in card_events:
        apply_event(deck, op, name)

    # over = in the replayed deck but not the logged one (unlogged removals, transforms, upgrades), under = the opposite (unlogged gains)
    over = Counter({k: deck[k] - final_deck[k] for k in deck if deck[k] > final_deck[k]})
    under = Counter({k: final_deck[k] - deck[k] for k in final_deck if final_deck[k] > deck[k]})

    extra = []

    def remove(n, floor):
        taken = 0
        for name in sorted(over, key = lambda k: (k not in STARTER_NAMES, k)):
            while over[name] > 0 and taken < n:
                over[name] -= 1
                taken += 1
                extra.append((floor, REMOVE, REMOVE, name))
        return taken

    def add(n, floor, only = None):
        added = 0
        for name in sorted(under):
            if only is not None and name not in only:
                continue
            while under[name] > 0 and added < n:
                under[name] -= 1
                added += 1
                extra.append((floor, ADD, ADD, name))
        return added

    def upgrade(n, floor, allowed = None):
        # n counts upgrades, not cards
        done = 0
        for name in sorted(over):
            if (name.endswith('_u') and searing_blow_level(name) is None) or (allowed is not None and name not in allowed):
                continue
            while over[name] > 0 and done < n:
                target = upgrade_target(name, under)
                if target is None or done + target[1] > n:
                    break
                over[name] -= 1
                under[target[0]] -= 1
                done += target[1]

                step = name
                for _ in range(target[1]):
                    extra.append((floor, UPGRADE, UPGRADE, step))
                    step = upgraded(step)
        return done

    def transform(n, floor, starters_only = False):
        # Transform = remove a card and gain a different one. Unexplained gains can be any card
        removed = 0
        for name in sorted(over, key = lambda k: (k not in STARTER_NAMES, k)):
            if starters_only and name not in STARTER_NAMES:
                continue
            while over[name] > 0 and removed < n:
                over[name] -= 1
                removed += 1
                extra.append((floor, REMOVE, REMOVE, name))
        add(removed, floor)
        return removed

    bonus = NEOW_BONUS.get(event.get('neow_bonus'), (0, 0, 0))
    remove(bonus[0], 0)
    transform(bonus[1], 0)
    upgrade(bonus[2], 0)

    if event.get('neow_cost') == 'CURSE':
        add(1, 0, only = CURSES)

    # Vampires: the Strikes lost are not logged, only the Bites gained
    for e in event.get('event_choices', []):
        if e.get('event_name') == 'Vampires' and str(e.get('player_choice', '')).startswith('Became a vampire'):
            for name in [k for k in over if is_strike(k)]:
                for _ in range(over[name]):
                    extra.append((e['floor'], REMOVE, REMOVE, name))
                over[name] = 0

    for floor, name in sorted(relic_floors):
        if name not in RELIC_EFFECTS:
            continue

        n_remove, n_transform, n_upgrade, curse = RELIC_EFFECTS[name]

        remove(n_remove, floor)
        transform(sum(over[k] for k in STARTER_NAMES) if n_transform == 'starters' else n_transform, floor, starters_only = n_transform == 'starters')
        upgrade(n_upgrade, floor)

        if curse is not None:
            add(1, floor, only = {curse})

    # Remaining upgrades are attributed to Armaments / Apotheosis, halfway between obtaining it and the end of the run
    upgrader_floors = [e[0] for e in card_events if e[2] == ADD and e[3].replace('_u', '') in COMBAT_UPGRADERS]

    if upgrader_floors:
        upgrade(len(over) * 100, (upgrader_floors[0] + floor_reached) // 2)

    return extra, sum(over.values()) + sum(under.values())

def replay_run(event, card_set, relic_set, n_floors, rng):
    '''
        Returns None if the run cannot be replayed, otherwise (rows, residual) where
        rows = list of (floor, {card: qty}, [relics]) for up to n_floors floors sampled from 1..floor_reached, cards as model columns (see expand_searing_blow)
        Each row is the state on arriving at that floor (events logged on floors < floor applied)
        residual = number of cards the replayed final deck still differs from the logged final deck by, after
        inferring the unlogged Neow / relic / combat upgrade changes (see infer_unlogged)
    '''
    character = str(event.get('character_chosen', '')).lower()
    floor_reached = event.get('floor_reached')

    if character not in STARTING_DECKS or not isinstance(floor_reached, int) or floor_reached < 1:
        return None

    start_deck = Counter(STARTING_DECKS[character])
    if event.get('ascension_level', 0) >= 10:
        start_deck['ascendersbane'] += 1

    # Some older runs have malformed log entries (no floor, null entries), which can't be placed in time
    try:
        card_events = get_card_events(event, card_set)
        relic_floors = get_relic_floors(event, relic_set)

        final_deck = Counter(n for n in clean_cards(event['master_deck']) if in_card_set(n, card_set))

        extra, residual = infer_unlogged(event, start_deck, card_events, relic_floors, floor_reached, final_deck)
    except (KeyError, TypeError, AttributeError, ValueError, IndexError):
        return None

    card_events = sorted(card_events + extra, key = lambda x: (x[0], x[1]))

    deck = Counter(start_deck)
    sampled_floors = sorted(rng.sample(range(1, floor_reached + 1), min(n_floors, floor_reached)))

    rows = []
    i = 0
    for floor in sampled_floors:
        while i < len(card_events) and card_events[i][0] < floor:
            apply_event(deck, card_events[i][2], card_events[i][3])
            i += 1

        rows.append((floor, expand_searing_blow({k: v for k, v in deck.items() if v > 0}), [name for f, name in relic_floors if f < floor]))

    return rows, residual
