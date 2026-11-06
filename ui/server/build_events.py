"""Writes data/events.json: the events the simulator knows, with the effect of every option written in terms the model can score.

Run from the project root:  python ui/server/build_events.py

The values are entered by hand from the game (Ascension 15+ numbers where they differ) and are not read from game files, so check them
against the game before relying on one. An option the model cannot judge (a fight, potions, Strength, a mini game) says so in `gaps`
or `fight`, and the simulator calls that out instead of guessing.

Effect keys (all optional, applied in this order, then the choices are made by the model):
  gold, gold_all, gold_per_act        gold change, lose everything, gold per act number (act 1 = 1)
  hp, hp_pct_max, heal_pct_max        HP change, HP change as a fraction of max HP (negative = loss), heal a fraction of max HP
  heal_full, max_hp, max_hp_pct       heal to full, max HP change (gains also heal), max HP change as a fraction
  add, curse, relic, lose_relic       card names to add ("Name+" = upgraded), curse names, relic names to gain or lose
  remove_curses, upgrade_all_basics   remove every curse, upgrade every Strike and Defend
  remove, duplicate, upgrade, transform (+ transform_up)   the player chooses n cards: the model picks the best ones
  random_cards {n, up, pool}, random_relics, random_relic_from [names], random_curse, upgrade_random   random results, averaged over draws
  card_reward {k, must}               choose one of k random cards (must = no skip), valued as the expected best card
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
EVENTS = []

def opt(name, desc, effects=None, outcomes=None, gaps=None, needs_relic=None, lacks_relic=None, fight=False, leave=False):
    o = {'name': name, 'desc': desc}
    for k, v in (('effects', effects), ('outcomes', outcomes), ('gaps', gaps), ('needs_relic', needs_relic), ('lacks_relic', lacks_relic)):
        if v:
            o[k] = v
    if fight:
        o['fight'] = True
    if leave:
        o['leave'] = True
    return o

def ev(name, acts, options, note='', unmodeled=None):
    e = {'name': name, 'acts': acts, 'options': options}
    if note:
        e['note'] = note
    if unmodeled:
        e['unmodeled'] = unmodeled
    EVENTS.append(e)

LEAVE = opt('Leave', 'Walk away', leave=True)
SHRINE = [1, 2, 3]

# Act 1
ev('Big Fish', [1], [
    opt('Banana', 'Heal a third of max HP', {'heal_pct_max': 0.33}),
    opt('Donut', '+5 max HP', {'max_hp': 5}),
    opt('Box', 'A random relic and the Regret curse', {'random_relics': 1, 'curse': ['Regret']})])
ev('The Cleric', [1, 2], [
    opt('Heal', 'Pay 35 gold, heal a quarter of max HP', {'gold': -35, 'heal_pct_max': 0.25}),
    opt('Purify', 'Pay 75 gold (50 below Ascension 15), remove a card', {'gold': -75, 'remove': 1}),
    LEAVE])
ev('Dead Adventurer', [1], [
    opt('Search', 'Each search can start an elite fight; rewards are gold, a relic or nothing',
        gaps=['An elite fight can start and the rewards are random with odds that change each search, none of that is modelled']),
    LEAVE])
ev('Golden Idol', [1], [
    opt('Take, then outrun', 'Golden Idol, then the Injury curse', {'relic': ['Golden Idol'], 'curse': ['Injury']}),
    opt('Take, then smash', 'Golden Idol, then lose 35% of max HP (25% below Ascension 15)', {'relic': ['Golden Idol'], 'hp_pct_max': -0.35}),
    opt('Take, then hide', 'Golden Idol, then lose 10% of max HP permanently', {'relic': ['Golden Idol'], 'max_hp_pct': -0.10}),
    LEAVE])
ev('Wing Statue', [1], [
    opt('Pray', 'Lose 7 HP, remove a card', {'hp': -7, 'remove': 1}),
    opt('Destroy', 'Needs an Attack that deals 10 or more damage, 50 to 80 gold', {'gold': 65},
        gaps=['Only available if you have an Attack with 10 or more damage, which the tool cannot check']),
    LEAVE])
ev('World of Goop', [1], [
    opt('Gather gold', '+75 gold, lose 11 HP', {'gold': 75, 'hp': -11}),
    opt('Leave it', 'Lose about 35 gold', {'gold': -35})])
ev('Living Wall', [1], [
    opt('Forget', 'Remove a card', {'remove': 1}),
    opt('Change', 'Transform a card', {'transform': 1}),
    opt('Grow', 'Upgrade a card', {'upgrade': 1})])
ev('Hypnotizing Colored Mushrooms', [1], [
    opt('Stomp', 'Fight 3 Fungi Beasts, Odd Mushroom as the reward', fight=True,
        gaps=['A fight: the model does not predict how much HP it costs or whether you win']),
    opt('Eat', 'Heal a quarter of max HP, the Parasite curse', {'heal_pct_max': 0.25, 'curse': ['Parasite']})])
ev('Scrap Ooze', [1], [
    opt('Reach inside (first try)', 'Lose 5 HP, a 25% chance of a relic', outcomes=[
        {'p': 0.25, 'label': 'Found a relic', 'effects': {'hp': -5, 'random_relics': 1}},
        {'p': 0.75, 'label': 'Nothing', 'effects': {'hp': -5}}],
        gaps=['You can keep trying and each try costs more HP with better odds, only the first try is scored']),
    LEAVE])
ev('Shining Light', [1], [
    opt('Enter', 'Lose 30% of max HP (20% below Ascension 15), upgrade 2 random cards', {'hp_pct_max': -0.30, 'upgrade_random': 2}),
    LEAVE])
ev('The Ssssserpent', [1], [
    opt('Agree', '+150 gold (175 below Ascension 15), the Doubt curse', {'gold': 150, 'curse': ['Doubt']}),
    opt('Disagree', 'Leave', leave=True)])

# Shrines, any act
ev('Bonfire Spirits', SHRINE, [
    opt('Offer a card', 'The card is removed, the reward depends on its rarity', {'remove': 1},
        gaps=['The healing depends on the rarity of the card you offer (common 5 HP, uncommon full heal, rare full heal and +10 max HP, curse a relic), which the tool does not know']),
    LEAVE])
ev('Duplicator', SHRINE, [opt('Pray', 'Duplicate a card', {'duplicate': 1}), LEAVE])
ev('Golden Shrine', SHRINE, [
    opt('Pray', '+50 gold (100 below Ascension 15)', {'gold': 50}),
    opt('Desecrate', '+275 gold, the Regret curse', {'gold': 275, 'curse': ['Regret']}),
    LEAVE])
ev('Lab', SHRINE, [opt('Take the potions', 'Three random potions', gaps=['Potions are not a model input, so their value cannot be scored'])])
ev('Match and Keep', SHRINE, [], unmodeled='A memory mini game: the result depends on your play and the cards revealed, nothing here for the model to predict')
ev('Note For Yourself', SHRINE, [], unmodeled='You swap a card for one saved from an earlier run, and that card is not known to the tool')
ev('Purifier', SHRINE, [opt('Pray', 'Remove a card', {'remove': 1}), LEAVE])
ev('Transmogrifier', SHRINE, [opt('Pray', 'Transform a card', {'transform': 1}), LEAVE])
ev('Upgrade Shrine', SHRINE, [opt('Pray', 'Upgrade a card', {'upgrade': 1}), LEAVE])
ev('Wheel of Change', SHRINE, [
    opt('Spin', 'One of six results, each equally likely', outcomes=[
        {'p': 1 / 6, 'label': 'Gold (100 per act)', 'effects': {'gold_per_act': 100}},
        {'p': 1 / 6, 'label': 'A random relic', 'effects': {'random_relics': 1}},
        {'p': 1 / 6, 'label': 'Full heal', 'effects': {'heal_full': True}},
        {'p': 1 / 6, 'label': 'The Decay curse', 'effects': {'curse': ['Decay']}},
        {'p': 1 / 6, 'label': 'Remove a card', 'effects': {'remove': 1}},
        {'p': 1 / 6, 'label': 'Lose 15% of max HP', 'effects': {'hp_pct_max': -0.15}}])])
ev('The Woman in Blue', SHRINE, [
    opt('Buy one potion', '20 gold', {'gold': -20}, gaps=['The potion is not scored, only the gold']),
    opt('Buy two potions', '30 gold', {'gold': -30}, gaps=['The potions are not scored, only the gold']),
    opt('Buy three potions', '40 gold', {'gold': -40}, gaps=['The potions are not scored, only the gold']),
    opt('Leave', 'Lose 5% of max HP at Ascension 15 and above', {'hp_pct_max': -0.05})])
ev('Fountain of Cleansing', SHRINE, [opt('Drink', 'Remove every curse', {'remove_curses': True}), LEAVE])

# Act 2
ev('Forgotten Altar', [2], [
    opt('Offer the Golden Idol', 'Lose the Golden Idol, gain Bloody Idol', {'lose_relic': ['Golden Idol'], 'relic': ['Bloody Idol']}, needs_relic='Golden Idol'),
    opt('Sacrifice', '+5 max HP, lose 35% of max HP (25% below Ascension 15)', {'max_hp': 5, 'hp_pct_max': -0.35}),
    opt('Desecrate', 'The Decay curse', {'curse': ['Decay']})])
ev('Cursed Tome', [2], [
    opt('Read it all', 'Lose 15 HP in total (16 below Ascension 15), a random book relic', {'hp': -15, 'random_relic_from': ['Necronomicon', 'Enchiridion', "Nilry's Codex"]}),
    opt('Stop early', 'Lose about 3 HP, nothing gained', {'hp': -3}),
    LEAVE])
ev('Knowing Skull', [2], [
    opt('Ask for a potion', 'Lose 6 HP', {'hp': -6}, gaps=['The potion is not scored']),
    opt('Ask for gold', '+90 gold, lose 6 HP', {'gold': 90, 'hp': -6}),
    opt('Ask for a card', 'A random colorless card, lose 6 HP', {'hp': -6, 'random_cards': {'n': 1, 'pool': 'colorless'}}),
    opt('Leave', 'Lose 6 HP', {'hp': -6})],
   note='Each ask costs HP and you can ask again; only one ask is scored.')
ev('Masked Bandits', [2], [
    opt('Pay everything', 'Lose all your gold', {'gold_all': True}),
    opt('Fight', 'Fight the bandits, Red Mask as the reward', fight=True, gaps=['A fight: HP cost and outcome are not predicted'])])
ev('Colosseum', [2], [], unmodeled='A fight sequence: the model does not predict fights or their rewards')
ev('The Library', [2], [
    opt('Read', 'Choose one of 20 random cards', {'card_reward': {'k': 20, 'must': True}}),
    opt('Sleep', 'Heal a third of max HP (20% at Ascension 15 and above)', {'heal_pct_max': 0.20})])
ev('The Nest', [2], [
    opt('Smash and grab', '+50 gold (99 below Ascension 15)', {'gold': 50}),
    opt('Stay in line', 'Ritual Dagger, lose 6 HP', {'add': ['Ritual Dagger'], 'hp': -6})])
ev('Council of Ghosts', [2], [
    opt('Accept', '3 Apparitions (5 below Ascension 15), lose half of max HP', {'add': ['Apparition', 'Apparition', 'Apparition'], 'max_hp_pct': -0.5}),
    opt('Refuse', 'Leave', leave=True)])
ev('Drug Dealer', [2], [
    opt('J.A.X.', 'Add J.A.X.', {'add': ['J.A.X.']}),
    opt('Test subject', 'Transform 2 cards', {'transform': 2}),
    opt('Mutagens', 'Mutagenic Strength relic', {'relic': ['Mutagenic Strength']})])
ev('Back to Basics', [2], [
    opt('Simplicity', 'Remove a card', {'remove': 1}),
    opt('Elegance', 'Upgrade every Strike and Defend', {'upgrade_all_basics': True})])

# Act 3
ev('Moai Head', [3], [
    opt('Jump inside', 'Full heal, lose 12.5% of max HP', {'heal_full': True, 'max_hp_pct': -0.125}),
    opt('Offer the Golden Idol', '+333 gold, lose the Golden Idol', {'gold': 333, 'lose_relic': ['Golden Idol']}, needs_relic='Golden Idol'),
    LEAVE])
ev('Tomb of Lord Red Mask', [3], [
    opt('Offer all your gold', 'Lose all gold, gain Red Mask', {'gold_all': True, 'relic': ['Red Mask']}, lacks_relic='Red Mask'),
    opt('Don the mask', '+222 gold', {'gold': 222}, needs_relic='Red Mask'),
    LEAVE])
ev('Mysterious Sphere', [3], [], unmodeled='A fight for a rare relic: the model does not predict fights')
ev('Secret Portal', [3], [], unmodeled='Skips straight to the boss: floors and rewards skipped are not something the model scores')

if __name__ == '__main__':
    cards = {c['name'] for c in json.load(open(f'{HERE}/data/cards.json'))}
    relics = {r['name'] for r in json.load(open(f'{HERE}/data/relics.json'))}
    bad = []

    def check_effects(where, E):
        for n in E.get('add', []) + E.get('curse', []):
            if n.rstrip('+') not in cards:
                bad.append(f'{where}: unknown card {n}')
        for n in E.get('relic', []) + E.get('lose_relic', []) + E.get('random_relic_from', []):
            if n not in relics:
                bad.append(f'{where}: unknown relic {n}')

    for e in EVENTS:
        for o in e['options']:
            where = f"{e['name']} / {o['name']}"
            if 'effects' in o:
                check_effects(where, o['effects'])
            for oc in o.get('outcomes', []):
                check_effects(where, oc['effects'])
            for r in (o.get('needs_relic'), o.get('lacks_relic')):
                if r and r not in relics:
                    bad.append(f'{where}: unknown relic {r}')
            if 'outcomes' in o and abs(sum(oc['p'] for oc in o['outcomes']) - 1) > 1e-9:
                bad.append(f'{where}: outcome odds do not sum to 1')
    if bad:
        raise SystemExit('\n'.join(bad))
    for e in EVENTS:
        for i, o in enumerate(e['options']):
            o['id'] = f"{''.join(ch for ch in e['name'].lower() if ch.isalnum())}_{i}"
    json.dump(EVENTS, open(f'{HERE}/data/events.json', 'w'), indent=1)
    print(f'{len(EVENTS)} events, {sum(len(e["options"]) for e in EVENTS)} options written to data/events.json')
