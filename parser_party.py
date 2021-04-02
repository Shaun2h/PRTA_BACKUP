import os
import torch
from transformers import *

# Hello.
# A BERT sequence has the following format:
# single sequence: [CLS] X [SEP]
# pair of sequences: [CLS] A [SEP] B [SEP]
# ('[CLS]', 101)
# ('[SEP]', 102)
propaganda_classes = ["<PAD>", "O", "Name_Calling,Labeling", "Repetition", "Slogans", "Appeal_to_fear-prejudice", "Doubt", "Exaggeration,Minimisation", "Flag-Waving", "Loaded_Language", "Reductio_ad_hitlerum", "Bandwagon", "Causal_Oversimplification", "Obfuscation,Intentional_Vagueness,Confusion", "Appeal_to_Authority", "Black-and-White_Fallacy", "Thought-terminating_Cliches", "Red_Herring", "Straw_Men", "Whataboutism"]  



sentence_yn = ["Non-prop", "Prop"] # prop or non prop?

propaganda_tag2idx = {tag:idx for idx, tag in enumerate(propaganda_classes)}
propaganda_idx2tag = {idx:tag for idx, tag in enumerate(propaganda_classes)}
grouped =True
if grouped:
    propaganda_idx2tag = {0:["<PAD>"], 1:["O"], 2:["Repetition" ,"Slogans", "Appeal_to_fear-prejudice", "Flag-Waving", "Reductio_ad_hitlerum", "Bandwagon", "Appeal_to_Authority"], 3:["Causal_Oversimplification", "Obfuscation", "Intentional_Vagueness, Confusion", "Black-and-White_Fallacy", "Thought-terminating_Cliches"], 4:["Name_Calling,Labeling", "Doubt", "Loaded_Language", "Straw_Men"], 5:["Exaggeration,Minimisation", "Red_Herring", "Whataboutism"]}

    propaganda_tag2idx = {"<PAD>":0, "O":1, "Repetition":2 , "Slogans":2, "Appeal_to_fear-prejudice":2, "Flag-Waving":2, "Reductio_ad_hitlerum":2, "Bandwagon":2, "Appeal_to_Authority":2, "Causal_Oversimplification":3, "Obfuscation,Intentional_Vagueness,Confusion":3, "Black-and-White_Fallacy":3, "Thought-terminating_Cliches":3, "Name_Calling,Labeling":4, "Doubt":4, "Loaded_Language":4, "Straw_Men":4, "Exaggeration,Minimisation":5, "Red_Herring":5, "Whataboutism":5}

sentence_yn_tag2idx = {tag:idx for idx, tag in enumerate(sentence_yn)}
sentence_yn_idx2tag = {idx:tag for idx, tag in enumerate(sentence_yn)}
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

class sentences_dataset(torch.utils.data.Dataset):
    def __init__(self,articles,labels):
        super(sentences_dataset,self).__init__()
        self.articles = articles
        self.labels = labels
    
    def __len__(self):
        return len(self.articles)
        
    def __getitem__(self,idx):
        return torch.tensor(self.articles[idx]),torch.tensor(self.labels[idx])

def semeval_parsin(tokenizer,target_file):
    # returns ALL semeval stuff.
    all_sentences = []
    all_labels = []
    all_yn =[]
    tokenised_sentences = []
    tokenised_labels = []
    for opened_file in [open(os.path.join("devtrain_semeval2020_11",target_file),"r",encoding="utf-8")]:
        with opened_file as trainfile:
            sentence_latest = []
            tokenised_sentence_latest = [] 
            tokenised_labels_latest = []
            label_latest = []
            haspropo = False
            for line in trainfile:
                a = line.split()
                if len(a)>1:
                    sentence_latest.append(a[0]) # append to version that doesn't bother with tokenised stuff
                    encoded_version = tokenizer.encode(a[0], pad_to_max_length = False) # don't pad it.
                    for _ in range(len(encoded_version)):
                        tokenised_labels_latest.append(propaganda_tag2idx[a[1]]) # repeat. as many times as the tokenised version.
                    tokenised_sentence_latest.extend(encoded_version) # extend. the label
                    label_latest.append(propaganda_tag2idx[a[1]]) # append the label to the version of the list that doesn't do tokenised
                    if propaganda_tag2idx[a[1]]>1:
                        haspropo=True
                else:
                    # end of an article/sentence...
                    tokenised_sentence_latest.insert(0,101) # add CLS token to the start of tokenised list
                    tokenised_sentence_latest.append(102) # add a SEP token to the end of tokenised list.
                    tokenised_labels_latest.insert(0,1) # insert "O" for the start
                    tokenised_labels_latest.append(0) # append the appropriate labels of <PAD> to the ends.
                    tokenised_sentence_latest.extend([0]*(512 - len(tokenised_sentence_latest)))
                    tokenised_labels_latest.extend([0]*(512 - len(tokenised_labels_latest)))
                    # pad both to 512.
                    all_sentences.append(sentence_latest)
                    all_labels.append(label_latest)
                    if len(tokenised_sentence_latest)==512:
                        tokenised_sentences.append(tokenised_sentence_latest)
                        tokenised_labels.append(tokenised_labels_latest)
                    else:
                        tokenised_sentence_latest_TRUNC = tokenised_sentence_latest[:511]
                        tokenised_sentence_latest_TRUNC.append(102) # add a SEP token to the end of tokenised list after the trunc
                        tokenised_labels_latest_TRUNC = tokenised_labels_latest[:511]
                        tokenised_labels_latest_TRUNC.append(102) # append the appropriate SEP
                        tokenised_sentences.append(tokenised_sentence_latest_TRUNC)
                        tokenised_labels.append(tokenised_labels_latest_TRUNC)
                    if haspropo:
                        all_yn.append(1)
                    else:
                        all_yn.append(0)
                    sentence_latest = []
                    label_latest = []
                    haspropo = False
                    tokenised_sentence_latest = [] 
                    tokenised_labels_latest = []
    return all_sentences, all_labels, all_yn, tokenised_sentences, tokenised_labels

def datathon_parsin(dev_train_test):
    # first step. obtain data from files.
    basedirectory = os.path.join("datatahon dataset (protechn_corpus_eval)","tasks-2-3",dev_train_test)
    fileslist = os.listdir(os.path.join(basedirectory))
    articles = []
    articleslices = []
    propo_or_no=[]
    article_cutouts_labels = []
    for i in fileslist:
        if ".txt" in i:
            articles.append(i.split(".")[0])
    for article in articles:
        if dev_train_test=="train":
            # for some reason, it's not .labels.tsv for train only...
            labels = open(os.path.join(basedirectory,article+".task3.labels"),"r",encoding="utf-8")
            article = open(os.path.join(basedirectory,article+".txt"),"r",encoding="utf-8")
        else:
            labels = open(os.path.join(basedirectory,article+".labels.tsv"),"r",encoding="utf-8")
            article = open(os.path.join(basedirectory,article+".txt"),"r",encoding="utf-8")
        articlestring = article.read()
        lastcount = 0
        articlelength = len(articlestring)
        propo_or_no_latest = []
        article_cutouts_labels_latest = []
        article_cutouts_latest = []
        for line in labels:
            if len(line)>5:
                singularitem = line.split()
                latest_label = propaganda_tag2idx[singularitem[1]]
                cutstart = int(singularitem[2])
                cutend = int(singularitem[3])
                if len(articlestring[lastcount:cutstart])>=1: # in case the previous is a propo directly.
                    # since the previous is not a propo directly, append non prop
                    propo_or_no_latest.append(1)
                    article_cutouts_latest.append(articlestring[lastcount:cutstart])
                    article_cutouts_labels_latest.append(propaganda_tag2idx["O"])
                propo_or_no_latest.append(0) # append prop
                article_cutouts_latest.append(articlestring[cutstart:cutend])
                article_cutouts_labels_latest.append(latest_label)
                lastcount=cutend
        if lastcount!=articlelength-1:
            propo_or_no_latest.append(1)
            article_cutouts_latest.append(articlestring[lastcount:articlelength-1])
            article_cutouts_labels_latest.append(propaganda_tag2idx["O"])
        articleslices.append(article_cutouts_latest)
        propo_or_no.append(propo_or_no_latest)
        article_cutouts_labels.append(article_cutouts_labels_latest)
        article.close()
        # ending up!
    return articleslices,article_cutouts_labels,propo_or_no

def search_len(labels_input_array):
    # we need to identify the actual size of the input array too. Might as well do it live since the rest is preprocessed.
    try:
        if type(labels_input_array)==type(torch.tensor([1])): # if it's a tensor
            cut_target=labels_input_array.numpy().tolist().index(0)
            # so we know to ignore from then on
        else:
            cut_target=labels_input_array.index(0)
        return cut_target # so we know where to cut till
        
    except ValueError:
        # 0 does not exist. within. There is no padding.
        return len(labels_input_array)

def propo_or_no(labels_input_array):
    result = torch.where(labels_input_array>1,labels_input_array,torch.tensor([1]))
    if sum(result.view(-1))>1:
        return torch.tensor([1])
    else:
        return torch.tensor([0])

def datathon_to_full_set(article_by_sentence,article_sentence_labels,tokenizer):
    # second step. Taking articleslices and article_cutouts_labels from datathon_parsin method,
    # convert to tokens only and a label.
    # both are lists of lists, of length of 512.
    tokenarticles = []
    tokenlabels = []
    for idx in range(len(article_by_sentence)):
        totalarticle_tokens=[]
        totalarticlelabels =[]
        for inner_idx in range(len(article_by_sentence[idx])):
            c = tokenizer.encode(article_by_sentence[idx][inner_idx], pad_to_max_length = False) # we don't add special tokens here. we add them manually at the end.
            # keep in mind that 512 is the MAX this pretrained bert can support
            totalarticle_tokens.extend(c)
            newlabels = [article_sentence_labels[idx][inner_idx]]*len(c)
            totalarticlelabels.extend(newlabels)
            #  "<PAD>" => we use this label for cls and sep, and our padding.
            # keep in mind that the tokenized version is padded with 0s at the end.
        totalarticle_tokens.insert(0,101)# insert CLS at the start
        totalarticlelabels.insert(0,1) # "O" label for CLS at the start.
        totalarticle_tokens.append(102) # insert SEP at the end.
        totalarticle_tokens.extend([0] * (512 - len(totalarticle_tokens))) # pad to max length tokens
        totalarticlelabels.extend([0] * (512 - len(totalarticlelabels))) # pad to max length labels
        # truncate if it's >512....
        if len(totalarticle_tokens)>512:
            total_article_tokens_TRUNC = totalarticle_tokens[:511]
            total_article_tokens_TRUNC.append(102) # insert SEP at the end for this new trunc.
            totalarticlelabels_TRUNC = totalarticlelabels[:511]
            totalarticlelabels_TRUNC.append(0)
            tokenarticles.append(total_article_tokens_TRUNC)  
            tokenlabels.append(totalarticlelabels_TRUNC)
            
        else:
            tokenarticles.append(totalarticle_tokens)
            tokenlabels.append(totalarticlelabels)
    return tokenarticles,tokenlabels


"""
propaganda_classes = ["<PAD>", "O", "Name_Calling,Labeling", "Repetition", "Slogans", "Appeal_to_fear-prejudice", "Doubt", "Exaggeration,Minimisation", "Flag-Waving", "Loaded_Language", "Reductio_ad_hitlerum", "Bandwagon", "Causal_Oversimplification", "Obfuscation,Intentional_Vagueness,Confusion", "Appeal_to_Authority", "Black-and-White_Fallacy", "Thought-terminating_Cliches", "Red_Herring", "Straw_Men", "Whataboutism"]  
sentence_yn = ["Non-prop", "Prop"] # prop or non prop?
propaganda_tag2idx = {tag:idx for idx, tag in enumerate(propaganda_classes)}
propaganda_idx2tag = {idx:tag for idx, tag in enumerate(propaganda_classes)}
sentence_yn_tag2idx = {tag:idx for idx, tag in enumerate(sentence_yn)}
sentence_yn_idx2tag = {idx:tag for idx, tag in enumerate(sentence_yn)}
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')


print("preparing semeval")
semeval_sentences, semeval__labels, semeval__yn, semeval_tokenised_sentences, semeval_tokenised_labels = semeval_parsin(tokenizer)
print("sameval completed.")


datathon_target_files = ["dev","train"]
test_files = ["test"]
train_dev_articleslices = []
train_dev_article_cutouts_labels = []
train_dev_propo_or_no = []

print("\n\npreparing train datathon files.")
for datathontarget in datathon_target_files:
    articleslices_latest, article_cutouts_labels_latest, propo_or_no_latest = datathon_parsin(datathontarget)
    train_dev_articleslices.extend(articleslices_latest)
    train_dev_article_cutouts_labels.extend(article_cutouts_labels_latest)
    train_dev_propo_or_no.extend(propo_or_no_latest)
train_dev_tokenarticles,train_dev_tokenlabels = datathon_to_full_set(train_dev_articleslices,train_dev_article_cutouts_labels,tokenizer)

print("train datathon files completed.")

test_articleslices = []
test_article_cutouts_labels = []
test_propo_or_no = []

print("\n\npreparing test datathon files.")
for test_targets in test_files:
    # we only have datathon test files....
    test_articleslices_latest, test_article_cutouts_labels_latest, test_propo_or_no_latest = datathon_parsin(test_targets)
    test_articleslices.extend(articleslices_latest)
    test_article_cutouts_labels.extend(article_cutouts_labels_latest)
    test_propo_or_no.extend(propo_or_no_latest)
test_dev_tokenarticles,test_dev_tokenlabels = datathon_to_full_set(test_articleslices,test_article_cutouts_labels,tokenizer)
print("test datathon files completed.\n\n")



total_train_articles = train_dev_tokenarticles+semeval_tokenised_sentences
total_train_labels = train_dev_tokenlabels+semeval_tokenised_labels


##################
# running checks.
# if all lengths are not 512, ded dedo

# print(total_train_articles[0])
# print(len(total_train_articles[0]))

# print(total_train_articles[-1])
# print(len(total_train_articles[-1]))


# print(total_train_labels[0])
# print(len(total_train_labels[0]))


# print(total_train_labels[-1])
# print(len(total_train_labels[-1]))

##################
"""

