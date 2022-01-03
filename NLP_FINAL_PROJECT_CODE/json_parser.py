import jsonlines
import json
import csv

MAX_TWEET = 100000


#create list of files to be opened
#also add label of what tweets in those files should
#be labeled as
files_to_open = list()
labels        = list()

#get file names of what needs to be labelled
with open('files.txt', "r") as f:
    for fn in f.readlines():
        file_split = fn.split(" ")
        files_to_open.append(file_split[0])
        labels.append(file_split[1])

#to write new files names to ready files
fready = open('ready_tweet_files.txt', "a")

#create index to keep track of what label to use
label_index = 0



#for each file grab the tweet
#for fn in files_to_open:
for fn in files_to_open:

    #split tweets up into batches
    file_tweet_index = 0
    file_num = 0

    print("Opening {}...".format(fn))

    #create writer for jsonl file
    jsonfile = fn.split(".")[0] + "-0" + "-parsed" + ".jsonl"
    print("Writing to {}...".format(jsonfile))
    file2write = jsonlines.open(jsonfile, "w")

    #write to next pipeline file
    fready.write(jsonfile)

    with jsonlines.open(fn) as f:


        #iterate through json
        #extract needed data and write to new file
        for line in f.iter():

            file_tweet_index += 1

            #allows for tweets to be written in batches of MAX_NUM amount
            if file_tweet_index == MAX_TWEET:
                file_tweet_index = 0
                file_num += 1

                #create writer for jsonl file
                jsonfile = fn.split(".")[0] + "-{}".format(file_num) + "-parsed" + ".jsonl"
                print("Writing to {}...".format(jsonfile))
                file2write = jsonlines.open(jsonfile, "w")


            jsonobj = {"Text":line['full_text'], "Label": labels[label_index].strip("\n")}
            file2write.write(jsonobj) # or whatever else you'd like to do
            
    label_index+=1

