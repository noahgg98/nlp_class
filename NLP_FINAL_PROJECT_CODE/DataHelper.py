
"""all_embeddings = here_is_your_function_return_all_data()
all_embeddings = np.array(all_embeddings)
np.save('embeddings.npy', all_embeddings)

If you're saving into google colab, then you can download it to your local computer. Whenever you need it, just upload it and load it.
all_embeddings = np.load('embeddings.npy')
"""
from bert import *
import numpy

class DataHelper:

    def __init__(self,fn):

        #define three traits of a tweet
        #tweet string    --> string of tweet text
        #tweet embedding --> Bert Embedding
        #tweet Tag       --> political tag of tweet
        self.TWEETSTR = list()
        self.TWEETEMB = list()
        self.TWEETTAG = list()

        #to get tweets from
        self.FN = fn


    #getters
    #class meant to be stored in list of DataHelper objs
    def ret_strings(self):
        return self.TWEETSTR
    
    def ret_embs(self):
        return self.TWEETEMB

    def ret_tags(self):
        return self.TWEETTAG

    
    def get_data(self):

        #open file and grab file names to look at
        with open(self.FN, 'r') as f:

            #append to file list
            for file in f.readlines():

                npf  = file.strip('\n')
                fn = npf.split(" ")[0]
                tag  = npf.split(" ")[1]
                print("Opening {}...".format(npf))

                #open original json to read tweet strings from
                list_str = npf.split('-')
                fstrii = str(list_str[0]) + "-" + str(list_str[1]) + "-" + str(list_str[2]) \
                        + "-" + str(list_str[3]) + "-" + str(list_str[4]) + ".jsonl"
                    

                #retrieve embeddings and add to list
                embeds = np.load(fn)

                #loop through embeds and add to list
                print("Loading Embeddings...")
                for em in embeds:
                    self.TWEETEMB.append(em.astype('float64'))
                    self.TWEETTAG.append(tag)
                
                #get tweet string that matches with above tags and tweet embeddings
                with jsonlines.open(fstrii) as fjson:
                    print("Loading tweet strings from {}...".format(fstrii))
                    for tweetstr in fjson.iter():
                        self.TWEETSTR.append(tweetstr["Text"])
                    print("DONE...")


if __name__=="__main__":
    dh = DataHelper("test.txt")
    dh.get_data()

    #sanity check
    l1 = dh.ret_embs()
    l2 = dh.ret_strings()
    l3 = dh.ret_tags()

    #sizes should be the same
    print(len(l1))
    print(len(l2))
    print(len(l3))
               

	
	
	
