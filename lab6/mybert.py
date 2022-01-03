from transformers import BertTokenizer, BertModel, BertConfig
	    
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
config = BertConfig.from_pretrained('bert-base-uncased', output_hidden_states=True)
model = BertModel.from_pretrained('bert-base-uncased', config=config)



def sent_func(sent):
    global tokenizer
    global config
    global model

    # take in tokenized input and set output
    inputs = tokenizer(sent, return_tensors="pt")
    outputs = model(**inputs)

    #grab last hideen state
    last_hidden_states = outputs.last_hidden_state

    #print
    return last_hidden_states[0][0]

if __name__=='__main__':
    sent_func("Hello I am delighted")

