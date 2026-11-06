'''
Builds data/cards.json (display name, model column, owner, type, upgradable) for the UI.

Run from the project root:  python ui/server/build_catalog.py

The model columns are the source of truth (starting_decks.csv, which every v20+ model's input follows). Display names, owners and types are not in the data,
so they are written out below from the game and then checked against the columns: every card column must be covered exactly once and every name
below must match a column, otherwise this script stops and lists the differences.
'''

import json
import os
import re
import sys

import pandas as pd

ROOT = os.getcwd()
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, f'{ROOT}/scripts')
from name_utils import SEARING_BLOW_COLUMNS

def nk(s):
    return re.sub(r'[^a-z0-9.\-]', '', s.lower()).replace('-', '').replace('.', '')

# Current in-game name -> model column, only where normalizing the name does not give the column (the dataset still has some old card names)
COLUMN_OVERRIDES = {
    'Rushdown': 'adaptation', 'Wraith Form': 'wraithformv', 'J.A.X.': 'j.a.x.', 'Multi-Cast': 'multi-cast',
    # The dataset logs some cards under their internal ids
    'Claw': 'gash', 'Apparition': 'ghostly', 'Judgment': 'judgement', 'Recursion': 'redo', 'Tranquility': 'clearthemind',
    'Pressure Points': 'pathtovictory', 'Simmering Fury': 'vengeance', 'Nightmare': 'nightterror',
}

# owner -> type -> display names
TABLE = {
    'Ironclad': {
        'attack': 'Anger, Bash, Blood for Blood, Bludgeon, Body Slam, Carnage, Clash, Cleave, Clothesline, Dropkick, Feed, Fiend Fire, Headbutt, Heavy Blade, Hemokinesis, Immolate, Iron Wave, Perfected Strike, Pommel Strike, Pummel, Rampage, Reaper, Reckless Charge, Searing Blow, Sever Soul, Sword Boomerang, Thunderclap, Twin Strike, Uppercut, Whirlwind, Wild Strike',
        'skill': 'Armaments, Battle Trance, Bloodletting, Burning Pact, Disarm, Double Tap, Dual Wield, Entrench, Exhume, Flame Barrier, Flex, Ghostly Armor, Havoc, Impervious, Infernal Blade, Intimidate, Limit Break, Offering, Power Through, Rage, Second Wind, Seeing Red, Sentinel, Shockwave, Shrug It Off, Spot Weakness, True Grit, Warcry',
        'power': 'Barricade, Berserk, Brutality, Combust, Corruption, Dark Embrace, Demon Form, Evolve, Feel No Pain, Fire Breathing, Inflame, Juggernaut, Metallicize, Rupture',
    },
    'The Silent': {
        'attack': 'All-Out Attack, Backstab, Bane, Choke, Dagger Spray, Dagger Throw, Dash, Die Die Die, Endless Agony, Eviscerate, Finisher, Flechettes, Flying Knee, Glass Knife, Grand Finale, Heel Hook, Masterful Stab, Neutralize, Poisoned Stab, Predator, Quick Slash, Riddle with Holes, Skewer, Slice, Sneaky Strike, Sucker Punch, Unload',
        'skill': 'Acrobatics, Adrenaline, Backflip, Blade Dance, Blur, Bouncing Flask, Bullet Time, Burst, Calculated Gamble, Catalyst, Cloak and Dagger, Concentrate, Corpse Explosion, Crippling Poison, Deadly Poison, Deflect, Distraction, Dodge and Roll, Doppelganger, Escape Plan, Expertise, Leg Sweep, Malaise, Nightmare, Outmaneuver, Phantasmal Killer, Piercing Wail, Prepared, Reflex, Setup, Storm of Steel, Survivor, Tactician, Terror, Venomology',
        'power': 'A Thousand Cuts, Accuracy, After Image, Caltrops, Envenom, Footwork, Infinite Blades, Noxious Fumes, Tools of the Trade, Well-Laid Plans, Wraith Form',
    },
    'Defect': {
        'attack': 'All for One, Ball Lightning, Barrage, Beam Cell, Blizzard, Bullseye, Claw, Cold Snap, Compile Driver, Core Surge, Doom and Gloom, FTL, Go for the Eyes, Hyperbeam, Melter, Meteor Strike, Rebound, Rip and Tear, Scrape, Streamline, Sunder, Sweeping Beam, Thunder Strike',
        'skill': 'Aggregate, Amplify, Auto-Shields, Boot Sequence, Chaos, Charge Battery, Chill, Coolheaded, Consume, Darkness, Double Energy, Dualcast, Equilibrium, Fission, Force Field, Fusion, Genetic Algorithm, Glacier, Hologram, Leap, Multi-Cast, Recursion, Recycle, Reinforced Body, Reprogram, Reboot, Seek, Skim, Stack, Steam Barrier, Tempest, Turbo, White Noise, Zap',
        'power': 'Biased Cognition, Buffer, Capacitor, Creative AI, Defragment, Echo Form, Electrodynamics, Heatsinks, Hello World, Loop, Machine Learning, Self Repair, Static Discharge, Storm',
    },
    'Watcher': {
        'attack': 'Bowling Bash, Brilliance, Carve Reality, Conclude, Consecrate, Crush Joints, Cut Through Fate, Empty Fist, Eruption, Fear No Evil, Flurry of Blows, Flying Sleeves, Follow-Up, Just Lucky, Lesson Learned, Ragnarok, Reach Heaven, Sands of Time, Sash Whip, Signature Move, Talk to the Hand, Tantrum, Wallop, Weave, Wheel Kick, Windmill Strike',
        'skill': 'Alpha, Beta, Blasphemy, Collect, Conjure Blade, Crescendo, Deceive Reality, Deus Ex Machina, Empty Body, Empty Mind, Evaluate, Foreign Influence, Halt, Indignation, Inner Peace, Judgment, Meditate, Omniscience, Perseverance, Pray, Pressure Points, Prostrate, Protect, Sanctity, Scrawl, Simmering Fury, Spirit Shield, Swivel, Third Eye, Tranquility, Vault, Vigilance, Wave of the Hand, Wish, Worship, Wreath of Flame',
        'power': 'Battle Hymn, Deva Form, Devotion, Establishment, Fasting, Foresight, Like Water, Master Reality, Mental Fortress, Nirvana, Omega, Rushdown, Study',
    },
    'colorless': {
        'attack': 'Hand of Greed, Mind Blast, Swift Strike, The Bomb, Violence',
        'skill': 'Apotheosis, Bandage Up, Blind, Chrysalis, Dark Shackles, Deep Breath, Discovery, Dramatic Entrance, Enlightenment, Finesse, Flash of Steel, Forethought, Good Instincts, Impatience, Jack of All Trades, Madness, Magnetism, Master of Strategy, Mayhem, Metamorphosis, Panacea, Panache, Panic Button, Purity, Sadistic Nature, Secret Technique, Secret Weapon, Thinking Ahead, Transmutation, Trip',
        'power': '',
    },
    'curse': {
        'curse': "Ascender's Bane, Clumsy, Curse of the Bell, Decay, Doubt, Injury, Necronomicurse, Normality, Pain, Parasite, Pride, Regret, Shame, Writhe",
    },
    'status': {
        'status': 'Burn, Wound',
    },
    'special': {
        'attack': 'Bite, Expunger, J.A.X., Ritual Dagger, Shiv, Smite',
        'skill': 'Apparition, Insight, Miracle, Safety',
    },
    'starter': {
        'attack': 'Strike',
        'skill': 'Defend',
    },
}

def main():
    cols = list(pd.read_csv(f'{ROOT}/starting_decks.csv', nrows = 0).columns)
    relics = [l.strip() for l in open(f'{ROOT}/data/master_lists/master_relic_list.txt')]
    non_card = set(relics) | {'floor', 'defect', 'ironclad', 'the_silent', 'watcher'} | set(SEARING_BLOW_COLUMNS[1:])
    card_cols = [c for c in cols if c not in non_card and not c.endswith('_u')]
    have_up = {c[:-2] for c in cols if c.endswith('_u') and c[:-2] in card_cols}
    have_up.add('searingblow')

    cards, seen, unmatched = [], set(), []
    for owner, types in TABLE.items():
        for typ, names in types.items():
            for name in [n.strip() for n in names.split(',') if n.strip()]:
                col = COLUMN_OVERRIDES.get(name, nk(name))
                if col not in card_cols:
                    unmatched.append((name, col))
                    continue
                if col in seen:
                    print('duplicate', name, col)
                    continue
                seen.add(col)
                cards.append({'name': name, 'column': col, 'owner': owner, 'type': typ, 'upgradable': col in have_up})
    left = [c for c in card_cols if c not in seen]
    if unmatched or left:
        print('NAMES WITH NO COLUMN:', unmatched)
        print('COLUMNS WITH NO NAME:', left)
        if '--strict' in sys.argv:
            raise SystemExit(1)
    # Columns the table does not cover are old or beta card names that are still in the data. They are kept (so the model columns are all accounted for)
    # but marked legacy, and the UI neither offers nor accepts them
    for c in left:
        cards.append({'name': c, 'column': c, 'owner': 'legacy', 'type': 'unknown', 'upgradable': c in have_up})
    cards.sort(key = lambda c: c['name'].lower())
    json.dump(cards, open(f'{HERE}/data/cards.json', 'w'), indent = 1)
    print(len(cards), 'cards written;', len(left), 'columns without a hand-written name')

# Relics: one entry per line of data/master_lists/master_relic_list.txt (display names and owners are written out from the game, the list is the check)
RELIC_NAMES = """akabeko=Akabeko;anchor=Anchor;ancientteaset=Ancient Tea Set;artofwar=Art of War;astrolabe=Astrolabe;bagofmarbles=Bag of Marbles;bagofpreparation=Bag of Preparation;birdfacedurn=Bird-Faced Urn;blackblood=Black Blood;blackstar=Black Star;bloodvial=Blood Vial;bloodyidol=Bloody Idol;bluecandle=Blue Candle;boot=The Boot;bottledflame=Bottled Flame;bottledlightning=Bottled Lightning;bottledtornado=Bottled Tornado;brimstone=Brimstone;bronzescales=Bronze Scales;burningblood=Burning Blood;bustedcrown=Busted Crown;cables=Gold-Plated Cables;calipers=Calipers;callingbell=Calling Bell;captainswheel=Captain's Wheel;cauldron=Cauldron;centennialpuzzle=Centennial Puzzle;ceramicfish=Ceramic Fish;championbelt=Champion Belt;charonsashes=Charon's Ashes;chemicalx=Chemical X;circlet=Circlet;cloakclasp=Cloak Clasp;clockworksouvenir=Clockwork Souvenir;coffeedripper=Coffee Dripper;crackedcore=Cracked Core;cultistmask=Cultist Headpiece;cursedkey=Cursed Key;damaru=Damaru;darkstoneperiapt=Dark Stone Periapt;datadisk=Data Disk;deadbranch=Dead Branch;dodecahedron=Dodecahedron;dollysmirror=Dolly's Mirror;dreamcatcher=Dream Catcher;du-vudoll=Du-Vu Doll;ectoplasm=Ectoplasm;emotionchip=Emotion Chip;emptycage=Empty Cage;enchiridion=Enchiridion;eternalfeather=Eternal Feather;faceofcleric=Face of Cleric;fossilizedhelix=Fossilized Helix;frozencore=Frozen Core;frozenegg=Frozen Egg;frozeneye=Frozen Eye;fusionhammer=Fusion Hammer;gamblingchip=Gambling Chip;ginger=Ginger;girya=Girya;goldeneye=Golden Eye;goldenidol=Golden Idol;gremlinhorn=Gremlin Horn;gremlinmask=Gremlin Visage;handdrill=Hand Drill;happyflower=Happy Flower;holywater=Holy Water;horncleat=Horn Cleat;hoveringkite=Hovering Kite;icecream=Ice Cream;incenseburner=Incense Burner;inkbottle=Ink Bottle;inserter=Inserter;juzubracelet=Juzu Bracelet;kunai=Kunai;lantern=Lantern;leeswaffle=Lee's Waffle;letteropener=Letter Opener;lizardtail=Lizard Tail;magicflower=Magic Flower;mango=Mango;markofpain=Mark of Pain;markofthebloom=Mark of the Bloom;matryoshka=Matryoshka;mawbank=Maw Bank;mealticket=Meal Ticket;meatonthebone=Meat on the Bone;medicalkit=Medical Kit;melange=Melange;membershipcard=Membership Card;mercuryhourglass=Mercury Hourglass;moltenegg=Molten Egg;mummifiedhand=Mummified Hand;mutagenicstrength=Mutagenic Strength;necronomicon=Necronomicon;neowsblessing=Neow's Lament;nilryscodex=Nilry's Codex;ninjascroll=Ninja Scroll;nlothsgift=N'loth's Gift;nlothsmask=N'loth's Hungry Face;nuclearbattery=Nuclear Battery;nunchaku=Nunchaku;oddlysmoothstone=Oddly Smooth Stone;oddmushroom=Odd Mushroom;oldcoin=Old Coin;omamori=Omamori;orangepellets=Orange Pellets;orichalcum=Orichalcum;ornamentalfan=Ornamental Fan;orrery=Orrery;pandorasbox=Pandora's Box;pantograph=Pantograph;papercrane=Paper Krane;paperfrog=Paper Phrog;peacepipe=Peace Pipe;pear=Pear;pennib=Pen Nib;philosophersstone=Philosopher's Stone;pocketwatch=Pocketwatch;potionbelt=Potion Belt;prayerwheel=Prayer Wheel;preservedinsect=Preserved Insect;prismaticshard=Prismatic Shard;purewater=Pure Water;questioncard=Question Card;redmask=Red Mask;redskull=Red Skull;regalpillow=Regal Pillow;ringoftheserpent=Ring of the Serpent;ringofthesnake=Ring of the Snake;runiccapacitor=Runic Capacitor;runiccube=Runic Cube;runicdome=Runic Dome;runicpyramid=Runic Pyramid;sacredbark=Sacred Bark;selfformingclay=Self-Forming Clay;shovel=Shovel;shuriken=Shuriken;singingbowl=Singing Bowl;slaverscollar=Slaver's Collar;sling=Sling of Courage;smilingmask=Smiling Mask;snakeskull=Snake Skull;sneckoeye=Snecko Eye;sozu=Sozu;spiritpoop=Spirit Poop;ssserpenthead=Ssserpent Head;stonecalendar=Stone Calendar;strangespoon=Strange Spoon;strawberry=Strawberry;strikedummy=Strike Dummy;sundial=Sundial;symbioticvirus=Symbiotic Virus;teardroplocket=Teardrop Locket;theabacus=The Abacus;thecourier=The Courier;thespecimen=The Specimen;threadandneedle=Thread and Needle;tingsha=Tingsha;tinychest=Tiny Chest;tinyhouse=Tiny House;toolbox=Toolbox;torii=Torii;toughbandages=Tough Bandages;toxicegg=Toxic Egg;toyornithopter=Toy Ornithopter;tungstenrod=Tungsten Rod;turnip=Turnip;twistedfunnel=Twisted Funnel;unceasingtop=Unceasing Top;vajra=Vajra;velvetchoker=Velvet Choker;violetlotus=Violet Lotus;warpaint=War Paint;warpedtongs=Warped Tongs;whetstone=Whetstone;whitebeaststatue=White Beast Statue;wingedgreaves=Winged Greaves;wristblade=Wrist Blade;yang=Duality"""
RELIC_OWNERS = {'Ironclad':'burningblood blackblood brimstone championbelt charonsashes magicflower paperfrog redskull runiccube selfformingclay','The Silent':'ringofthesnake ringoftheserpent papercrane snakeskull hoveringkite tingsha toughbandages thespecimen ninjascroll twistedfunnel wristblade','Defect':'crackedcore frozencore datadisk emotionchip cables inserter nuclearbattery runiccapacitor symbioticvirus','Watcher':'purewater holywater damaru yang teardroplocket cloakclasp goldeneye melange violetlotus'}

def build_relics():
    keys = [l.strip() for l in open(f'{ROOT}/data/master_lists/master_relic_list.txt') if l.strip()]
    display = dict(x.split('=', 1) for x in RELIC_NAMES.split(';'))
    owner = {k: c for c, v in RELIC_OWNERS.items() for k in v.split()}
    missing, extra = [k for k in keys if k not in display], [k for k in display if k not in keys]
    if missing or extra:
        raise SystemExit(f'relic names do not match the master list: missing {missing}, extra {extra}')
    relics = sorted(({'name': display[k], 'column': k, 'owner': owner.get(k, 'shared')} for k in keys), key = lambda r: r['name'].lower())
    json.dump(relics, open(f'{HERE}/data/relics.json', 'w'), indent = 1)
    print(len(relics), 'relics written')

main()
build_relics()
