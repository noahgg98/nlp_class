from transformers import DistilBertTokenizer, DistilBertModel, DistilBertConfig
import numpy as np
import jsonlines
import re



class Bert:

    #how to save to a file
    """all_embeddings = here_is_your_function_return_all_data()
    all_embeddings = np.array(all_embeddings)
    np.save('embeddings.npy', all_embeddings)

    If you're saving into google colab, then you can download it to your local computer. Whenever you need it, just upload it and load it.
    all_embeddings = np.load('embeddings.npy')
    """

    #set up initial variables
    #accepts file name if .txt where to get tweet files from
    #embeddings and file list inited as lists
    def __init__(self, fn):

        self.FN        = fn
        self.EMBEDS    = list()
        self.TWEETS    = list()
        self.FILES     = list()
        self.Tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.Config    = DistilBertConfig.from_pretrained('distilbert-base-uncased', output_hidden_states=True)
        self.Model     = DistilBertModel.from_pretrained('distilbert-base-uncased', config=self.Config)

    def clean(self, sentence):
        words = sentence.split(" ")
        s = list()
        for w in words:
            punctuation = "#()/\\\"\'"
            w = re.sub(r'http\S+', '', w)
            w = re.sub(r'RT[A-Za-z0-9]+', '', w)
            w = re.sub(r'@[A-Za-z0-9]+', '', w)
            for p in punctuation:
                w = w.replace(p, "")
            if not w  == "":
                s.append(w)
        return " ".join(s)
        
    #simple function with one purpose
    #grabs file names from ready tweet files
    #stores them in the global files list
    #files grabbed from here should be .jsonl
    def read_from_file(self):

        #open file and grab file names to look at
        with open(self.FN, 'r') as f:

            #append to file list
            for file in f.readlines():
                self.FILES.append(file.strip('\n'))

    #gets the tweet text from a file and creates bert embedding
    #embeddings are saved into a numpy array which can them be grabbed later
    def get_tweets(self):

        #loop through .jsonl files and parse the objs
        for file in self.FILES:
            
            #print file file
            print("Reading from ..." + " " + file)

            #set embed file name
            embedFile = file.split(".")[0] + "-" + "Numpy.npy"

            #index used if you need to only go
            #through part of the file
            #index = 0

            with jsonlines.open(file) as f:
                for line in f.iter():

                    #append bert embedding --> will need to chane 'full_text' in actual implementation
                    self.EMBEDS.append(self.bert_embedding(self.clean(line['Text'])).detach().numpy()) # or whatever else you'd like to do

                    #see above for index usage
                    #if index == 10:
                    #    break


                    #index+=1

            #save embeds to a file
            all_embeddings = np.array(self.EMBEDS)
            np.save(embedFile, all_embeddings)

            #clear EMBEDS for next file
            self.EMBEDS = list()


    #takes a tweet as input
    #tweet should be cleaned and devoid of junk
    #appends embedding to self.Embeds
    def bert_embedding(self, sent):

        # take in tokenized input and set output
        inputs = self.Tokenizer(sent, return_tensors="pt")
        outputs = self.Model(**inputs)

        #grab last hideen state
        last_hidden_states = outputs.last_hidden_state

        #self.EMBEDS.append(last_hidden_states[0][0])
        return last_hidden_states[0][0]

if __name__=='__main__':
    myb = Bert('ready_tweet_files.txt')
    myb.read_from_file()
    myb.get_tweets()
