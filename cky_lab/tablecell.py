#
# You fill this in with whatever helpful functions and data fields you need.

#from _typeshed import Self
from nltk.grammar import *

class TableCell:

    def __init__(self):
        self.NT    = dict()
        self.UN    = dict()
        self.WORD  = ""
        self.BP    = dict()


    def addNonTerminal(self, key, val):
        #set checked flag
        checked = False

        #check to see if terminal already exists
        for nts in self.NT.keys():

            #check to see if there already exists a val
            if nts == key:
                if self.NT[nts] < val:
                    self.NT[key] = val 
            
                #set flag to true if been checked
                checked = True
        
        if not checked:
            self.NT[key] = val
        
    #add to unary rules
    def addUnary(self, key, tag):

        if len(key) > 0:
            for item in key:
                self.UN[item] = tag

    def get_prob(self, rule):
        try:
            return self.NT[rule]
        except:
            return 0

    def update_NT(self):

        try:
            #init vars to compare to
            highest_prob = 0.00
            highest_item = None

            #loop through unary rules
            for items in self.UN.keys():
                if items[1] > highest_prob:
                    highest_prob = items[1]
                    highest_item = items

            #get value associated with highest item
            pos_tag = self.UN[highest_item]
            pos_tag_prob = self.NT[pos_tag]
            
            #get final probability and add to NT
            final_prob = pos_tag_prob * highest_prob
            self.NT[str(Production.lhs(highest_item[0]))] = final_prob
        except:
            pass

    def ret_NT(self):
        return self.NT

    def ret_highest(self):

        #set vars to be compared against
        highest      = 0
        keyOfHighest = ""

        #loop through NT and get one with highest probability
        for nts in self.NT.keys():
            if self.NT[nts] > highest:
                highest = self.NT[nts]
                keyOfHighest = nts

        return keyOfHighest

    def set_highest(self):
        self.NT['S'] = 1

    def set_ch(self, tag, pointer):
        self.BP[tag] = pointer
    
    def ret_bp(self, tag):

        try:
            return self.BP.get(tag)
        except:
            None
    
    def set_word(self, w):
        self.WORD = w

    def get_word(self):
        return self.WORD

    def __str__(self):
        return str(self.NT)

      
    


