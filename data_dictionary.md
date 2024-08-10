# STS Data Dump Data Dictionary
Run down of fields found in JSON objects pulled from google drive.
Data from reddit 'insert post link here'

## Modeling Fields
Fields used for model.
Passed to production database.

- **master_deck** *(list)*: deck at the end of the run
    - upgraded cards are denoted by '+1' recoded to '_u' to make data easier to work with as '+' is a special reserved character in regex
    - For version 1 of model all searing blow upgrades are treated as +1
    - stored as list of <u>individual</u> cards. QTY can be determined by counting instances
    - example file has '_R' as the end of strikes and defends. Unsure about meaning of this
- **character_chosen** *(str)*: name in caps of the character being used in this run
- **victory** *(bool)*: whether or not the player won. This was only looked at with an example that beat the heart, so unsure if this applies to only beating act 3. This is what the model is attempting to predict. 'true' means the player won.
- **play_id** *(str)*: run ID; used as ID field when joining data together

**Modeling Data Format Example**
Each row is a separate run.
Assume this extends for all cards (aka very wide).
One model will cover all characters.

| strike | defend | bash | bash_u | wild_strike | ... | character | victory |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 4 | 5 | 0 | 1 | 0 | ... | ironclad | false

## Filtering Fields
Fields that would be useful to filter data on.
Not passed to production database.

- **is_trial** *(bool)*: no idea what this is, but I'm assuming it has to do with a trial version of the game.
    - false -> finetuning
    - true -> pre-training alpha
- **is_daily** *(bool)*: whether the run is a daily or not. Out of scope for model and contains high amounts of variance due to random conditions
    - false -> finetuning
    - true -> pre-training alpha
- **chose_seed** *(bool)*: whether or not the player chose the seed. This can have a massive impact on the results and is not reflective of standard gameplay.
    - false -> finetuning
    - true -> pre-training alpha
- **timestamp** *(int)*: time of run in UTC watcher was released after the 2018-2019 data in addition to several rebalancing patches being released. Only use most current data.
    - *>=* 1593576000 -> finetuning
    - *<* 1593576000 -> pre-training alpha
- **is_beta** *(bool)*: whether the run was done on a beta version of the game
    - false -> finetuning
    - true -> pre-training alpha
- **is_endless** *(bool)*: whether the game was played in endless mode. This is not standard gameplay.
    - false -> finetuning
    - true -> pre-training alpha
- **ascension_level** *(int)*: ascension (aka difficulty) level the run was played on
    - 20 -> finetuning (may lower threshold later if insufficient data)
    - [0, 19] -> pre-training beta
        - *Note*: this is the only way a run can end up in pre-training beta. Logic: if all other checks result in finetuning except this one, then send run to pre-training beta

## Currently Irrelevant Fields
Documentation of other fields not needed now, but that could be useful later.
These are not stored in production data base

- **gold_per_floor** *(list)*: amount of gold in inventory at each floor
- **floor_reached** *(int)*: highest floor number reached
- **playtime** *(int)*: amount of time (seconds?) that was spent during the run
- **items_purged** *(list)*: cards removed from the deck through the run
    - unsure if in order of removal
    - unsure if this list can only contain cards (theoretically it could also contain relics if removed from an event)
- **score** *(int)*: score at end of the run
- **local_time** *(int)*: timestamp when the time was completed in a given timezone. Could be an interesting analysis feature at some point to see if players do better or worse at certain times of day.
- **is_ascension_mode** *(bool)*: whether it is an ascension run; redundant to ascension level field
- **campfire_choices** *(dict)*: player actions at each campfire
    - **data** *(str)* *(optional)*: name of card smithed
    - **floor** *(int)*: floor number of campsite
    - **key** *(str)*: action taken in CAPS
- **neow_cost** *(str)*: negative effective of option from floor zero (neow)
- **seed_source_timestap** *(int)*: I think this is the timestamp from when the run started (e.g. when the seed was pulled)
- **circlet_count** *(int)*: number of circlet relics. The player gets a circlet when obtaining a relic, but all other available relics have already been taken. No effect.
- **relics** *(list)*: relics aquired by the user, assumed to be in acquisition order
- **potions_floor_usage** *(list)*: floor number a potion was used on
- **damage_taken** *(dict)*: combat of all fights throughout the run
    - **damage** *(int)*: amount of damage taken
    - **enemies** *(str)*: name of enemies fought
        - If multiple then format is qty. space name (e.g. '2 Louse')
    - **floor** *(int)*: floor number combat occured on
    - **turns** *(int)*: number of turns during combat
- **seed_played** *(int)*: seed used for the run
- **potions_obtained** *(dict)*: which potions were obtained by the player
    - **(floor)** *(int)*: floor number potioned was obtained on
    - **(key)** *(str)*: name of the potion. Note: some have spaces in the name some do not (e.g. SmokeBomb and Energy Potion are both used)
- **path_per_floor** *(list)*: single letters denoting the player path throughout the run. null is boss
- **campfire_rested** *(int)*: number of campfires the player rested at during the run
- **item_purchase_floors** *(list)*: floor number where an item was purchased
- **current_hp_per_floor** *(list)*: amount of hp remaining at each floor. Unsure if recorded at the start or end of an encounter.
- **gold** *(int)*: amount of gold remaining at end of run
- **new_bonus** *(str)*: player choice at floor zero/neow
- **is_prod** *(bool)*: unsure what this is
- **campfire_upgraded** *(int)*: number of campfires player upgraded at
- **win_rate** *(float)*: win rate of the player. In the example I looked at the player was on A5 and won the round, but the winrate was zero. Unsure if/when this is calculated.
- **build_version** *(date)*: version date of the game being played. This is covered via other fields.
- **purchased_purges** *(int)*: number of times cards were purchased to be purged at the merchant
- **card_choices** *(dict)*: cards offered to the player
    - **not_picked** *(list)*: cards shown, but not added to the deck by the player
    - **picked** *(str)*: card selected by the player. If no card was slelected, this field is populated with 'SKIP'
    - **floor** *(int)*: floor number of card choice
- **player_experience** *(int)*: player's score accumulated across all runs
- **relics_obtained** *(dict)*: which relic was obtained by the player
    - **floor** *(int)*: floor number relic was obtained on
    - **key** *(str)*: name of the relic
- **event_choices** *(dict)*: all events (? spaces) visited by the player
    - **damage_healed** *(int)*: amount of hp restored to player by the event
    - **gold_gained** *(int)*: amount of gold gained by the player from the event
    - **player_choice** *(str)*: name of the choice taken by the player. 'Ignored' is populated if player did not take any action
    - **damage_taken** *(int)*: amount of damage taken by the player from the event
    - **max_hp_gain** *(int)*: amount added to players max health
    - **max_hp_loss** *(int)*: amount of max hp lost by players due to event
    - **relics_obtained** *(list)* *(optional)*: name(s) of relices obtained by the player
    - **event_name** *(str)*: name of the event
    - **floor** *(int)*: floor number where event took place
    - **gold_loss** *(int)*: amount of gold lost by the player.
    - **cards_obtained** *(list)* *(optional)*: names of card(s) given t player by the event
- **items_purged_floors** *(list)*: list of floor numbers where items (e.g. cards) were purged
- **potions_floor_spawned** *(list)*: floor number where potions were available to the player to take