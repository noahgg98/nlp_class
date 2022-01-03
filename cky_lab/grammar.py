#
# @author Nate Chambers
#
# I highly suggest you don't change anything in here.
#
import nltk


class Lexicon:

    def __init__(self, trees):
        self.words = dict()
        self.pos   = dict()
        self.pos_unique_rules = dict()
        self.wordstags = dict()
        self.total_word_types = 0
        self.totalTokens = 0
        
        for tree in trees:
            # sanity check
            if type(tree) != nltk.tree.Tree:
                print("ERROR: improper type given to Lexicon initialization:", type(tree))
                return

            # Count word/POS pairs.
            for (w,p) in tree.pos():
                self.words[w]  = self.words.get(w,0) + 1
                self.pos[p]    = self.pos.get(p,0) + 1
                if not (w,p) in self.wordstags:
                    self.pos_unique_rules[p] = self.pos_unique_rules.get(p,0) + 1
                self.wordstags[(w,p)] = self.wordstags.get((w,p),0) + 1
                self.totalTokens += 1

        self.total_word_types = len(self.words.keys())
        #print(self.wordstags)

    def get_all_tags(self):
        """ @return a list of all POS tags seen in training """
        return self.pos.keys()

    def get_rule_probability(self, tag, word):
        """ @return the probability of a unary lexicon rule: POS->word """
        tagN = self.pos.get(tag,0)
        c_wordtag = self.wordstags.get((word,tag),0)
        k = .00001
        
        # P(word | tag ) =  C(word,tag) / C(tag)
        #                == C(word,tag)+1 / C(tag)+numRulesWith(tag)
        #print('**',tag,word, c_wordtag, tagN, (c_wordtag + k) / (tagN + self.total_word_types*k))
        #return (c_wordtag + k) / (tagN + self.total_word_types*k)

        # Full Bayes' rule increases 7 F1 points over the above.
        p_tag = tagN / self.totalTokens
        c_word = self.words.get(word,0)
        if c_word < 5:
            c_word += k
            c_wordtag += (self.pos_unique_rules[tag]*k) / self.total_word_types
        p_word = (k + c_word) / (self.totalTokens + self.total_word_types*k);
        p_tag_given_word = c_wordtag / c_word;
        return p_word * p_tag_given_word / p_tag

    def word_exists(self, word):
        """ @return True if the given word was seen in training, False otherwise. """
        return word in self.words

    def tag_exists(self, tag):
        """ @return True if the given POS tag was seen in training, False otherwise. """
        return tag in self.pos



class Grammar:
    
    def __init__(self, trees):
        # For counting
        self.labelcounts = dict()
        self.binary_rule_counts = dict()        
        self.unary_rule_counts = dict()

        # For probabilities
        self.unary_rules_by_child = dict()
        self.binary_rules_by_left_child = dict()        
        self.binary_rules_by_right_child = dict()
        
        for tree in trees:
            # sanity check
            if type(tree) != nltk.tree.Tree:
                raise RuntimeError("ERROR: improper type given to Lexicon initialization: " + str(type(tree)))

            self._count_nonterminals(tree)

        self._compute_probabilities()
      
    def get_binaryrules_by_left_child(self, child):
        """ 
        Finds all binary grammar rules that have the given child as its first/left child.
        Creates a tuple pair for each discovered rule: (rule, probability)
        The rule is an nltk.grammar.Production object. The probability is a double.
        @return a List of tuple pairs
        """
        return self.binary_rules_by_left_child.get(child, [])

    def get_binaryrules_by_right_child(self, child):
        """
        Same as the left_child function, but finds rules with the given child as its right child.
        """
        return self.binary_rules_by_right_child.get(child, [])
    
    def get_unaryrules_by_child(self, child):
        """
        Finds all unary grammar rules with the given child on its left-hand side.
        Creates a tuple pair for each discovered rule: (rule, probability)
        The rule is an nltk.grammar.Production object. The probability is a double.
        @return a List of tuple pairs
        """
        return self.unary_rules_by_child.get(child, [])

    
    def _count_nonterminals(self, tree):
        for prod in tree.productions():
            # CNF check
            if len(prod.rhs()) > 2 or len(prod.rhs()) < 1:
                raise RuntimeError("ERROR: tree is not in chomsky normal form: " + str(tree))

            if type(prod.rhs()[0]) != str:
                parent = str(prod.lhs())
                self.labelcounts[parent] = self.labelcounts.get(parent,0) + 1
                if len(prod.rhs()) == 2:
                    self.binary_rule_counts[prod] = self.binary_rule_counts.get(prod,0) + 1
                else:
                    self.unary_rule_counts[prod] = self.unary_rule_counts.get(prod,0) + 1


    def _compute_probabilities(self):
        # Unary Rules        
        for (unary,count) in self.unary_rule_counts.items():
            parent = str(unary.lhs())
            child = str(unary.rhs()[0])
            # init list
            if not child in self.unary_rules_by_child:
                self.unary_rules_by_child[child] = list()

            # probability
            prob = count / self.labelcounts[parent]
            self.unary_rules_by_child[child].append( (unary,prob) )

        # Binary Rules
        for (binary,count) in self.binary_rule_counts.items():
            parent     = str(binary.lhs())
            leftchild  = str(binary.rhs()[0])
            rightchild = str(binary.rhs()[1])

            # init lists
            if not leftchild in self.binary_rules_by_left_child:
                self.binary_rules_by_left_child[leftchild] = list()
            if not rightchild in self.binary_rules_by_right_child:
                self.binary_rules_by_right_child[rightchild] = list()

            # probability
            prob = count / self.labelcounts[parent]
            self.binary_rules_by_left_child[leftchild].append( (binary,prob) )
            self.binary_rules_by_right_child[rightchild].append( (binary,prob) )



def condense_tags(trees):        
    for tree in trees:
        condense_tree_tags(tree)
        
def condense_tree_tags(tree):
    """ 
    Tags in Penn Treebank have IDs like WHNP-124, NP-SBJ, etc.
    This removes any hyphen extended non-terminal to simplify the set of rules.
    """
    tag = tree.label();
    hyphen = tag.find('-')
    if hyphen > 0:
        tree.set_label(tag[0:hyphen])
    if tree.height() > 2:
        for t in tree:
            condense_tree_tags(t)

#
# https://stackoverflow.com/questions/33939486/how-to-identify-and-remove-trace-trees-from-nltk-trees
# Removes some * traces, but not all...
#
def del_traces(t):
    """ Trees have traces ** in them where arguments are implicit. This removes many of them. """
    for ind, leaf in reversed(list(enumerate(t.leaves()))):
        if leaf.startswith("*") and leaf.endswith("*"):
            postn = t.leaf_treeposition(ind)
            parentpos = postn[:-1]
            while parentpos and len(t[parentpos]) == 1:
                postn = parentpos
                parentpos = postn[:-1]
            del t[postn]

def remove_longs(trees, max_len):
    """ @return a new list of trees that filters out any trees with > max_len words. """
    keep = []
    for t in trees:
        if len(t.leaves()) <= max_len:
            keep.append(t)
    return keep

def remove_empty_roots(trees):
    """ All WSJ trees start with an empty '' unary rule, this removes that wrapper. """
    for i in range(len(trees)):
        t = trees[i]
        if t.label() == '' and len(t) == 1:
            trees[i] = t[0]

def clean_trees(trees, max_len):
    remove_empty_roots(trees)
    condense_tags(trees)
    for tree in trees:
        del_traces(tree)

    trees = remove_longs(trees, max_len)
    return trees