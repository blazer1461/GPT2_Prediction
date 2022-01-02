# Author: Samer Nour Eddine (snoure01@tufts.edu) and Spencer Ha (sha01@tufts.edu)
import torch
import math
import os
import csv
from pathlib import Path
from torch._C import set_num_interop_threads
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import numpy as np

def readWritefromAFile(i, path):
    #change this to your own path that contains the stimuli
    file = open(path + i)
    file_lines = ""
    for i, line in enumerate(file):
        if i < 2:
            continue
        else:
            # Here add some code that parses out and does not read the split 
            # string when processing in the GPT2 model.
            #if line[0:3] == "SP1: " or "SP2: ":
                #file_lines = file_lines.split(" ", 1)

            file_lines = file_lines + line
    return file_lines 


def softmax(x):
	exps = [np.exp(i) for i in x]
	tot= sum(exps)
	return [i/tot for i in exps]
    
def Sort_Tuple(tup):  
  
    # (Sorts in descending order)  
    # key is set to sort using second element of  
    # sublist lambda has been used  
    tup.sort(key = lambda x: x[1])  
    return tup[::-1]

# Load pre-trained model (weights) - this takes the most time
model = GPT2LMHeadModel.from_pretrained('gpt2-large', output_hidden_states = True, output_attentions = True)
model.eval()

tokenizer = GPT2Tokenizer.from_pretrained('gpt2-large')

def cloze_finalword(text, i, path):
    '''
    This is a version of cloze generator that can handle words that are not in the model's dictionary.
    '''
    #Requires Python 3.5 or greater. Checks to see if a folder to output text files exists, otherwise create it
    totalSuprisal = 0
    Path("/Users/spencer/Downloads/GPT2/csvSur_Lenas_stims_probs_new/").mkdir(parents = True, exist_ok = True)
    #write_path = 'Users/spencer/Downloads/GPT2/Lenas_stims_probs/'
    #below code changes the i value from a .cha file to write to a .txt file
    baseFile = os.path.splitext(i)[0]
    #if you want the resulting output to be as a text file. comment it out if you want a csv file instead

    #writeFile = open('/Users/spencer/Downloads/GPT2/Lenas_stims_probs_new/' + baseFile + '.txt', 'w')

    #create a dictionary with the key as file, word pos, word, line, likelihood
    filed = open('/Users/spencer/Downloads/GPT2/csvSur_Lenas_stims_probs_new/' + baseFile + '.csv', mode = 'w')
    filedSuprisal = open('/Users/spencer/Downloads/GPT2/csvSur_Lenas_stims_probs_new/' + baseFile + 'perplexity' + '.csv', mode = 'w')
    writeFile = csv.writer(filed, delimiter= ',')
    writeFileSuprisal = csv.writer(filedSuprisal, delimiter = ',')
    
    text = text.split()
    sentence_thus_far = ""
    counter = 0
    line_number = 1 
    for word in text:
        if word == "SP2:":
            line_number = line_number + 1
        counter = counter + 1
        if counter == 1:
            sentence_thus_far = word
        else:
            sentence_thus_far =  sentence_thus_far + " " + word
        if counter < 2:
            continue
        whole_text_encoding = tokenizer.encode(sentence_thus_far)
    # Parse out the stem of the whole sentence (i.e., the part leading up to but not including the critical word)
        text_list = sentence_thus_far.split()
        text_list = text_list[-500:]
        stem = ' '.join(text_list[:-1])
        stem_encoding = tokenizer.encode(stem)
    # cw_encoding is just the difference between whole_text_encoding and stem_encoding
    # note: this might not correspond exactly to the word itself
    # e.g., in 'Joe flicked the grasshopper', the difference between stem and whole text (i.e., the cw) is not 'grasshopper', but
    # instead it is ' grass','ho', and 'pper'. This is important when calculating the probability of that sequence.
        cw_encoding = whole_text_encoding[len(stem_encoding):]

    # Run the entire sentence through the model. Then go back in time and look at what the model predicted for each token, starting at the stem.
    # e.g., for 'Joe flicked the grasshopper', go back to when the model had just received 'Joe flicked the' and
    # find the probability for the next token being 'grass'. Then for 'Joe flicked the grass' find the probability that
    # the next token will be 'ho'. Then for 'Joe flicked the grassho' find the probability that the next token will be 'pper'.

    # Put the whole text encoding into a tensor, and get the model's comprehensive output
        tokens_tensor = torch.tensor([whole_text_encoding])
    
        with torch.no_grad():
            outputs = model(tokens_tensor)
            predictions = outputs[0]   

        logprobs = []
    # start at the stem and get downstream probabilities incrementally from the model(see above)
    # I should make the below code less awkward when I find the time
        start = -1-len(cw_encoding)
        for j in range(start,-1,1):
                raw_output = []
                for i in predictions[-1][j]:
                        raw_output.append(i.item())
    
                logprobs.append(np.log(softmax(raw_output)))
            
    # if the critical word is three tokens long, the raw_probabilities should look something like this:
    # [ [0.412, 0.001, ... ] ,[0.213, 0.004, ...], [0.002,0.001, 0.93 ...]]
    # Then for the i'th token we want to find its associated probability
    # this is just: raw_probabilities[i][token_index]
        conditional_probs = []
        for cw,prob in zip(cw_encoding,logprobs):
                conditional_probs.append(prob[cw])
    # now that you have all the relevant probabilities, return their product.
    # This is the probability of the critical word given the context before it.
        product=np.exp(np.sum(conditional_probs))
        suprisal = -math.log(product, 10)
        totalSuprisal = suprisal + totalSuprisal
        writeFile.writerow([counter, line_number, word, sentence_thus_far, product, suprisal])
    averageSuprisal = totalSuprisal / counter
    perplexity = math.pow(math.e, averageSuprisal)
    writeFileSuprisal.writerow([perplexity])
        #writeFile.writelines(sentence_thus_far)
        #writeFile.writelines('\n')
        #writeFile.writelines(str(product))
        #writeFile.writelines('\n')
        #print(sentence_thus_far)
    print(baseFile + " has been completed")

def main(): 
    #get_inputFile = input("What file do you want to access? (include the .cha part!) ")
    path = "/Users/spencer/Downloads/GPT2/Lenas_stims/"
    filelist = os.listdir(path)
    for i in filelist:
        file_lines = readWritefromAFile(i, path)
    #text_cloze = input("What text do you want to find the probability for? ")
        cloze_finalword(file_lines, i, path)

main()