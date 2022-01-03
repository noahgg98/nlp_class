#
# @author Nate Chambers
#
# I highly suggest you don't change anything in here.
#
import os
from nltk.corpus import ptb
from nltk import Tree


def read_minitest(dirpath, low=1, high=5):
    alltrees = []
    for i in range(low,high):
        filepath = dirpath + os.path.sep + str(i) + '.mrg'
        if os.path.isfile(filepath):
            alltrees.extend( read_trees_from_file(filepath) )
    return alltrees


def read_trees_from_file(filepath):
    trees = []    
    if os.path.isfile(filepath):
        fh = open(filepath)

        treestr = ''
        for line in fh:
            # If new tree starting
            if line[0] == '(':
                # Build a tree from the previous lines!
                if len(treestr) > 0:
                    trees.append(Tree.fromstring(treestr))
                treestr = line.strip()
            elif not line.isspace():
                treestr += line.strip()
        # Save the final tree
        trees.append(Tree.fromstring(treestr))
    return trees


def read_trees(path, low=0, high=25):
    # Subdirectories '00' through '24'
    ids = []
    for i in range(0,25):
        if i < 10: istr = '0' + str(i)
        else: istr = str(i)
        ids.append(istr)

    # Read in the trees.
    trees = []
    for i in range(low,high):
        subpath = path + os.path.sep + ids[i] + os.path.sep
        for f in sorted(os.listdir(subpath)):
            filepath = subpath+os.path.sep+f
            if os.path.isfile(filepath):
                fh = open(filepath)

                treestr = ''
                for line in fh:
                    # If new tree starting
                    if line[0] == '(':
                        # Build a tree from the previous lines!
                        if len(treestr) > 0:
                            trees.append(Tree.fromstring(treestr))
                        treestr = line.strip()
                    elif not line.isspace():
                        treestr += line.strip()
                # Save the final tree
                trees.append(Tree.fromstring(treestr))
                
    print('Loaded', len(trees), 'trees!')
    return trees



# Testing
if __name__ == '__main__':

    trees = read_trees('../data/ptb/wsj')
    print('final tree:', trees[-1])
    print('final tree type:', type(trees[-1]))
