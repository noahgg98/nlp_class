#
# NLTK TREE API
# https://www.nltk.org/_modules/nltk/tree.html
#

import os
import argparse
import nltk
import ptbreader 
from grammar import Grammar,Lexicon,clean_trees
from tablecell import TableCell
from nltk.grammar import *

lexicon = Lexicon(ptbreader.read_trees_from_file('../data/ptb/wsj'))
grammar = Grammar(ptbreader.read_trees_from_file('../data/ptb/wsj'))

#def get_rule_prob(left, right, right2):
def get_needed_grammar(leftTable, rightTable):

    posRules = list()

    #get rules from each tables
    ltr = leftTable.ret_NT()
    rtr = rightTable.ret_NT()


    
    #get all possible combos
    for lr in list(ltr.keys()):

        #set left rule to compare
        leftrule  = grammar.get_binaryrules_by_left_child(lr)

        #check against right rules
        for rr in list(rtr.keys()):
            
            #get right rule grammar
            rightrule = grammar.get_binaryrules_by_right_child(rr)

            for rule in leftrule:
                #print(rule)
                if rule in rightrule:
                    #print(rule)
                    posRules.append(rule)
    
                
    return posRules



def get_best_parse(sentence):
    #print(grammar.get_binaryrules_by_left_child("V"))
    """ 
    sentence: a list of words (strings)
    This function creates a parse tree and returns a tree object of nltk.Tree
    """
    
    # These 2 lines create your CKY table out of TableCell objects (tablecell.py)
    N = len(sentence)
    table = [[TableCell() for x in range(N+1)] for y in range(N+1)]
    

    #
    # WRITE CODE HERE
    for ii in range(0, N):
        begin = ii
        end = ii+1

        #add word to table
        table[begin][end].set_word(sentence[begin])

        #Add the POS tags for this word.
        for tag in lexicon.get_all_tags():
            if lexicon.get_rule_probability(tag,sentence[begin]) > 0:
                table[begin][end].addNonTerminal(tag, lexicon.get_rule_probability(tag,sentence[begin]))
                table[begin][end].addUnary(grammar.get_unaryrules_by_child(tag), tag)
    
        table[begin][end].update_NT()
    


    
    # P( NP -> DT NN ) = Count( NP -> DT NN ) / Count(NP)
    # WRITE CODE HERE
    # Fill in the rest of the table
    for span in range(2, N+1):
        for begin in range(0,N-span+1):
            end = begin + span
            for split in range(begin+1,end):
                
                #loop through grammarf rules
                for rule in get_needed_grammar(table[begin][split],table[split][end]):
            
                    #get probability
                    prob = table[begin][split].get_prob(str(Production.rhs(rule[0])[0])) * table[split][end].get_prob(str(Production.rhs(rule[0])[1])) * rule[1]
                    
                    #add if not in table cell yet
                    if(table[begin][end].get_prob(rule[0])==0):

                        #set four part base pointer with each run getting paried with parent
                        table[begin][end].set_ch(str(Production.lhs(rule[0])), [str(Production.rhs(rule[0])[0]), table[begin][split], str(Production.rhs(rule[0])[1]),table[split][end]])
                        table[begin][end].addNonTerminal(str(Production.lhs(rule[0])), prob)


                    #add if higher probability than highest
                    if(prob > table[begin][end].get_prob(str(Production.lhs(rule[0])))):
                        table[begin][end].set_ch(str(Production.lhs(rule[0])), [str(Production.rhs(rule[0])[0]), table[begin][split], str(Production.rhs(rule[0])[1]),table[split][end]])
                        table[begin][end].addNonTerminal(str(Production.lhs(rule[0])), prob)


                    #Add all unary rules that match
                    table[begin][end].addUnary(grammar.get_unaryrules_by_child(str(Production.lhs(rule[0]))), str(Production.lhs(rule[0])))

                table[begin][end].update_NT()

    print_my_table(table)


    # Construct the Tree from the filled-in table
    table[0][N].set_highest()
    mytree = build_tree(table[0][N])


    mytree.un_chomsky_normal_form()

    
    print(mytree)
    return mytree
    
    
    
#NOT 100 % accuracy I think it has to do with the way I implemented table cell in the first two parts
def build_tree(node):
    # WRITE CODE HERE
    treeItem = node.ret_highest()
    #base case
    if node.ret_bp(treeItem) == None:
        return nltk.Tree(node.ret_highest(), [node.get_word()])

    #get base pointers based on tag
    base     = node.ret_bp(treeItem)
    left     = base[1]
    right    = base[3] 

    #build subtrees
    t = build_tree(left)
    s = build_tree(right)

    #create tree from subtrees
    p = nltk.Tree(node.ret_highest(), [t, s])
    
    # delete this when ready!
    return p




#
# ----------------------------------------
# DO NOT CHANGE ANYTHING BELOW HERE
# ----------------------------------------
#

def train(trees):
    global lexicon, grammar
    
    # Convert the trees to CNF here...
    for tree in trees:
        tree.chomsky_normal_form()
    
    # Requires the trees to be in CNF!
    lexicon = Lexicon(trees)
    grammar = Grammar(trees)

def print_my_table(table):
    for i in range(len(table)-1):
        for j in range(i, len(table)):
            print(i,j)
            print(table[i][j])

def evaluate_one(guess, gold, printout=False):
    golds = set(gold.productions())
    guesses = set(guess.productions())

    # Exact tree match.
    if golds-guesses == 0 and len(golds)==len(guesses):
        return (1.0,1.0,1.0,True)

    # Compute P/R/F1.
    local_fp = len(guesses-golds)
    local_tp = len(guesses) - len(guesses-golds)
    local_fn = len(golds-guesses)

    if printout:
        p  = local_tp / (local_tp+local_fp)
        r  = local_tp / (local_tp+local_fn)
        if (p+r) > 0:
            f1 = 2 * (p*r)/(p+r)
        else:
            f1 = 0
        print("Prec=%.3f Recall=%.3f F1=%.3f" % (p,r,f1))
    
    return (local_fp,local_tp,local_fn,False)

    
def evaluate(predicted, goldtrees):
    fp = 0
    tp = 0
    fn = 0
    exact = 0
    for i in range(0,len(goldtrees)):
        (local_fp,local_tp,local_fn,exactmatch) = evaluate_one(predicted[i],goldtrees[i])
        fp += local_fp
        tp += local_tp
        fn += local_fn

    p = tp / (tp+fp)
    r = tp / (tp+fn)
    if (p+r) > 0:
        f1 = 2 * (p*r)/(p+r)
    else:
        f1 = 0
    return (p,r,f1)


def test(trees, N):
    trees = trees[0:N]
    
    predictions = []
    for tree in trees:
        tokens = tree.leaves()
        print('CKY:', tokens)
        guess = get_best_parse(tokens)
        predictions.append(guess)
        evaluate_one(guess,tree,printout=True)        

    (P,R,F1) = evaluate(predictions, trees)
    print("OVERALL: Prec=%.3f\tRecall=%.3f\tF1=%.3f" % (P,R,F1))
        

if __name__ == '__main__':

    max_sent_length = 20

    # Command-line argument parser
    parser = argparse.ArgumentParser(description="ckyparser.py")
    parser.add_argument('-data', action="store", dest="data", type=str, default='data', help='Directory containing the tweets subdirectory')
    parser.add_argument('-sub', action="store", dest="sub", type=str, default='wsj/ptb', help='Directory containing the tweets subdirectory')
    parser.add_argument('-train', action="store", dest="train", type=str, default='5', help='Directory containing the tweets subdirectory')
    parser.add_argument('-test', action="store", dest="test", type=str, default='25', help='Directory containing the tweets subdirectory')    
    args = parser.parse_args()

    # miniTest
    if args.sub == 'miniTest':
        datapath = args.data + os.path.sep + 'miniTest'
        train_trees = ptbreader.read_minitest(datapath, 1, 4)
        test_trees = ptbreader.read_minitest(datapath, 4, 5)
    # full PTB
    else:
        trainmax = int(args.train)
        datapath = args.data + os.path.sep + 'ptb/wsj'
        train_trees = ptbreader.read_trees(datapath, 0, trainmax)
        test_trees = ptbreader.read_trees(datapath, 20, 21)

    train_trees = clean_trees(train_trees, max_sent_length)
    test_trees  = clean_trees(test_trees, max_sent_length)

    # Train the CKY parser!
    print("Training...")
    train(train_trees)
    
    # Test the CKY parser
    print("Testing...")
    num = int(args.test)
    test(test_trees, num)
