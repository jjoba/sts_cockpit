'''
    Program to assist slay the spire run

    TODO:
        Add store simulation
        Add data processing
        Win rate formatting
'''

import torch
import os
import re

import pandas as pd
import numpy as np
import tkinter as tk

from torch import nn
from tkinter import filedialog
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

import pdb

class Cockpit:

    @torch.no_grad()
    def __init__(self):
        # root = tk.Tk()
        # root.withdraw()

        # Determines whether to have pytorch use CPU or GPU
        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )

        print('What would you like to do?')
        self.user_choice = input('1. Start a new run \n2. Train Model\n3. Build Data\n')

        if self.user_choice == '1':
            self.run_setup()
            self.internal_state = 'support_run'
            self.run_game_loop()

        elif self.user_choice == '2':
            print('Not implemented yet')
            self.internal_state = 'train_model'
        elif self.user_choice == '3':
            print('Not implemented yet')
            self.internal_state = 'build_data'
        else:
            print('Invalid option')

    def get_win_rate(self, deck, print_results=True):
        # Predicts the win rate from some deck -> this is NOT self.deck
        x = torch.from_numpy(deck.to_numpy().astype(np.float32)).to(self.device)
        pred = self.model(x)
        
        if print_results:
            print(f'Predicted Win Rate: "{pred}"')

        return pred

    def run_setup(self):
        print('Which model would you like to use?')

        # Loads pytorch model and sets to eval mode
        model_path = filedialog.askopenfilename()
        self.model = torch.jit.load(model_path).to(self.device)
        self.model.eval()

        # Sets up starting deck and character
        self.deck = pd.read_csv(os.getcwd() + '/starting_decks.csv')
        self.deck.fillna(0, inplace = True)

        valid_input = False

        while valid_input == False:
            self.user_choice = input('Which character are you using? ')        
            
            if self.user_choice == 'ironclad':
                self.deck = self.deck[self.deck.ironclad == True]
                self.get_win_rate(self.deck)
                valid_input = True
            
            elif self.user_choice == 'the_silent':
                self.deck = self.deck[self.deck.the_silent == True]
                self.get_win_rate(self.deck)
                valid_input = True
            elif self.user_choice == 'defect':
                self.deck = self.deck[self.deck.defect == True]
                self.get_win_rate(self.deck)
                valid_input = True
            elif self.user_choice == 'watcher':
                self.deck = self.deck[self.deck.watcher == True]
                self.get_win_rate(self.deck)
                valid_input = True
            else:
                print('unknown character')

    def run_game_loop(self):

        # Game item variables to maintain internally for filtering
        self.non_removable = ['ironclad', 'the_silent', 'defect', 'watcher', 'unknown', 'ascendersbane']
        self.curses = ['doubt', 'regret']

        # Loads in the master relic list
        f = open(f'{os.getcwd()}/data/master_lists/master_relic_list.txt', 'r')
        master_relic_list = f.readlines()
        f.close()
        self.relic_list = [re.sub('\n', '', relic) for relic in master_relic_list]

        # Main menu loop for the game
        self.continue_run = True
        
        while self.continue_run == True:
            self.user_choice = input('\nWhat Would you like to do?\n1. Simulate\n2. Update Deck & Relics\n3. View Current Deck & Win Rate\n0. Exit\n')

            # Simulate
            if self.user_choice == '1':
                valid_choice = False

                while valid_choice == False:
                    choice = input('1. Which card should I add?\n2. Which card should I remove\n3. Which card should I upgrade?\n4. Simulate shop\n5. Next best card to add\n6. Which relic should I add?\n9. Go back\n')

                    # Set up temporary copy of the current deck for the simulation
                    sim_deck = self.deck.copy()

                    # Simulate Addition
                    if choice == '1':

                        valid_sim_choice = False

                        # Buffer to append simulated cards onto
                        cards_added = ['current_deck']

                        while valid_sim_choice is False:
                            sim_choice  = input('1. Add card to simulation\n2. Run simulation\n9. Go back\n')

                            # Add card(s) to sim
                            if sim_choice == '1':

                                valid_card = False
                                while valid_card is False:
                                    
                                    card = input('Enter card name or type 9 to go back: ')

                                    # Add card to sim
                                    if card in self.deck.columns:
                                        temp_sim_deck = self.deck.copy()

                                        temp_sim_deck[card] += 1

                                        # Joins results onto the simulated decks
                                        sim_deck = pd.concat([sim_deck, temp_sim_deck])
                                        cards_added.append(card)

                                        print(f'added {card}')
                                        valid_card = True
                                    
                                    # Go back to sim choices menu
                                    elif card == '9':
                                        valid_card = True
                                    else:
                                        print('Unknown Card Name')
                        
                            # Run simulation
                            if sim_choice == '2':
                                self.get_win_rate(sim_deck)
                                print(cards_added)
                                valid_sim_choice = True

                            # Go back: leave sim menu
                            if sim_choice == '9':
                                valid_sim_choice = True                   

                    # Simulate Removal
                    elif choice == '2':
                        cards_removed = ['current_deck']

                        for card in self.deck.columns:
                            
                            # Make sure we're not simming a character card or ascenders bane
                            if card in self.non_removable + self.relic_list:
                                pass
                            
                            # Add a row for any card currently present in the deck. subtract 1 from qty                            
                            elif self.deck[card].all() > 0:
                                temp_sim_deck = self.deck.copy()
                                temp_sim_deck[card] -= 1
                                
                                sim_deck = pd.concat([sim_deck, temp_sim_deck])
                                cards_removed.append(card)

                        # Get expected impact of each card removal
                        self.get_win_rate(sim_deck)
                        print(cards_removed)
                        
                        valid_choice = True

                    # Simulate Upgrade
                    elif choice == '3':
                        cards_upgraded = ['current_deck']

                        for card in self.deck.columns:
                            # Make sure we're not simming a character card or ascenders bane
                            if card in self.non_removable + self.curses + self.relic_list:
                                pass
                            
                            # No multi-upgrades
                            elif '_u' in card:
                                pass
                            
                            # Add a row for any card currently present in the deck. subtract 1 from qty                            
                            elif self.deck[card].all() > 0:
                                temp_sim_deck = self.deck.copy()
                                temp_sim_deck[card] -= 1
                                temp_sim_deck[card + '_u'] += 1
                                
                                sim_deck = pd.concat([sim_deck, temp_sim_deck])
                                cards_upgraded.append(card)

                        # Get expected impact of each card upgraded
                        self.get_win_rate(sim_deck)
                        print(cards_upgraded)

                        valid_choice = True

                    # Simulate Shop
                    elif choice == '4':
                        print('Not implemented yet. Use addition and removal options')
                        valid_choice = True

                    # Next best card
                    elif choice == '5':
                        next_best_card = ['current_deck']

                        for card in self.deck.columns:
                            # Make sure we're not simming a character card or ascenders bane or adding a relic
                            if card in self.non_removable + self.curses + self.relic_list:
                                pass
                            
                            else:
                                temp_sim_deck = self.deck.copy()
                                temp_sim_deck[card] += 1
                                
                                sim_deck = pd.concat([sim_deck, temp_sim_deck])
                                next_best_card.append(card)

                        # Get expected impact of each card upgraded
                        pred = self.get_win_rate(sim_deck, print_results=False)
                        index_max = np.argmax(pred)
                        print(f'Next best card to add is {next_best_card[index_max]} with a win rate of {pred[index_max]}')

                        valid_choice = True

                    # Which relic should I add?
                    elif choice == '6':
                        valid_sim_choice = False

                        # Buffer to append simulated cards onto
                        relics_added = ['current_deck']

                        while valid_sim_choice is False:
                            sim_choice  = input('1. Add relic to simulation\n2. Run simulation\n9. Go back\n')

                            # Add card(s) to sim
                            if sim_choice == '1':

                                valid_relic = False
                                while valid_relic is False:
                                    
                                    relic = input('Enter relic name or type 9 to go back: ')

                                    # Add card to sim
                                    if relic in self.relic_list:
                                        temp_sim_deck = self.deck.copy()

                                        temp_sim_deck[relic] += 1

                                        # Joins results onto the simulated decks
                                        sim_deck = pd.concat([sim_deck, temp_sim_deck])
                                        relics_added.append(relic)

                                        print(f'added {relic}')
                                        valid_relic = True
                                    
                                    # Go back to sim choices menu
                                    elif relic == '9':
                                        valid_relic = True
                                    else:
                                        print('Unknown Relic Name')
                        
                            # Run simulation
                            if sim_choice == '2':
                                self.get_win_rate(sim_deck)
                                print(relics_added)
                                valid_sim_choice = True

                            # Go back: leave sim menu
                            if sim_choice == '9':
                                valid_sim_choice = True
                    
                    # Go back to main menu
                    elif choice == '9':
                        valid_choice = True

                    # Invalid
                    else:
                        print(print(f'{choice} is invalid'))

            # Update deck
            elif self.user_choice == '2':
                valid_choice = False

                while valid_choice == False:
                    self.user_choice = input('\nWhat Would you like to do?\n1. Add a card\n2. Remove a card\n3. Upgrade a card\n4. Add a relic\n5. Remove a relic\n9. Go back\n')
                    
                    # Add
                    if self.user_choice == '1':
                        valid_card = False

                        while valid_card is False:
                            card = input('Which card would you like to add? 9 to go back\n')
                            
                            if card in self.deck.columns:
                                self.deck[card] += 1
                                valid_card = True
                            
                            # Go Back
                            elif card == '9':
                                valid_card = True
                            
                            # Try for new card
                            else:
                                print(f'I do not recognize cardname: {card}')
                        
                        if card != '9':
                            print(f'After adding {card} the new win rate is')
                            self.get_win_rate(self.deck)
                        
                        # Break Choice Loop
                        valid_choice = True

                    # Remove
                    elif self.user_choice == '2':
                        valid_card = False

                        while valid_card is False:
                            card_qty_g0 = True
                            card = input('Which card would you like to remove? 9 to go back\n')
                            
                            if card in self.deck.columns:
                                if card in self.non_removable + self.curses + self.relic_list:
                                    print(f'Cannot remove card {card}')
                                elif self.deck[card].all() > 0:
                                    self.deck[card] -= 1
                                    valid_card = True
                                else:
                                    print(f'You do not have any copies of {card} and cannot remove')
                                    card_qty_g0 = False
                            
                            # Go Back
                            elif card == '9':
                                valid_card = True
                            
                            # Try for new card
                            else:
                                print(f'I do not recognize cardname: {card}')
                        
                        if card != '9' and card_qty_g0 == True:
                            print(f'After removing {card} the new win rate is')
                            self.get_win_rate(self.deck)
                        
                        # Break Choice Loop
                        valid_choice = True

                    # Upgrade
                    elif self.user_choice == '3':
                        valid_card = False

                        while valid_card is False:
                            card = input('Which card would you like to upgrade? 9 to go back\n')
                            
                            if card in self.deck.columns:
                                # Check for upgraded card being passed
                                if '_u' in card:
                                    print('Please supply unupgraded card name (i.e. with xxx_u)')
                                elif card in self.non_removable + self.curses + self.relic_list:
                                    print('This item cannot be upgraded')
                                elif self.deck[card].all() == 0:
                                    print(f'You do not have any copies of {card} to upgrade')                               
                                else:
                                    # Add one to upgraded card and then remove un-upgraded
                                    self.deck[card + '_u'] += 1
                                    self.deck[card] -= 1
                                    valid_card = True
                            
                            # Go Back
                            elif card == '9':
                                valid_card = True
                            
                            # Try for new card
                            else:
                                print(f'I do not recognize cardname: {card}')
                        
                        if card != '9':
                            print(f'After upgrading {card} the new win rate is')
                            self.get_win_rate(self.deck)
                        
                        # Break Choice Loop
                        valid_choice = True

                    # Add relic
                    elif self.user_choice == '4':
                        valid_relic = False

                        while valid_relic is False:
                            relic = input('Which relic would you like to add? 9 to go back\n')
                            
                            if relic in self.relic_list:
                                self.deck[relic] += 1
                                valid_relic = True
                            
                            # Go Back
                            elif relic == '9':
                                valid_relic = True
                            
                            # Try for new card
                            else:
                                print(f'I do not recognize relic: {relic}')
                        
                        if relic != '9':
                            print(f'After adding {relic} the new win rate is')
                            self.get_win_rate(self.deck)
                        
                        # Break Choice Loop
                        valid_choice = True

                    # Remove relic
                    elif self.user_choice == '5':
                        valid_relic = False

                        while valid_relic is False:
                            relic_qty_g0 = True
                            relic = input('Which card would you like to remove? 9 to go back\n')
                            
                            if relic in self.deck.columns:
                                if relic not in self.relic_list:
                                    print(f'Cannot remove item {relic}')
                                elif self.deck[relic].all() > 0:
                                    self.deck[relic] -= 1
                                    valid_relic = True
                                else:
                                    print(f'You do not have {relic}')
                                    relic_qty_g0 = False
                            
                            # Go Back
                            elif relic == '9':
                                valid_relic = True
                            
                            # Try for new card
                            else:
                                print(f'I do not recognize relic: {relic}')
                        
                        if relic != '9' and relic_qty_g0 == True:
                            print(f'After removing {relic} the new win rate is')
                            self.get_win_rate(self.deck)
                        
                        # Break Choice Loop
                        valid_choice = True

                    # Go back
                    elif self.user_choice == '9':
                        valid_choice == True

                    # Invalid
                    else:
                        print(print(f'{self.user_choice} is invalid'))

            # View Deck & Win Rate
            elif self.user_choice == '3':
                print('Current Deck')
                print({k:int(v[0]) for (k, v) in self.deck.to_dict('list').items() if v[0] > 0})
                print('Win Rate')
                self.get_win_rate(self.deck)

            # Exit
            elif self.user_choice == '0':
                self.continue_run = False
            
            # Invalid
            else:
                print(f'{self.user_choice} is invalid')

if __name__ == '__main__':

    # Initialize
    cockpit = Cockpit()