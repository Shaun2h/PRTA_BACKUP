import torch
import pickle
import torch.utils.data
import json
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.nn.functional import relu, tanh, sigmoid
from transformers import *
#pip install pytorch_pretrained_bert==0.4.0
from pytorch_pretrained_bert.optimization import BertAdam, warmup_linear
from pytorch_pretrained_bert import BertModel, modeling
from pytorch_pretrained_bert.modeling import PreTrainedBertModel
from parser_party import * # import all parser party methods. They are relatively uniquely named so it should be safe namespace wise.
from transformers import AdamW
article_to_diagnose_test = """
Couple question why police waited so long to search New Mexico compound
(CNN) A New Mexico couple told authorities months ago they thought a missing Georgia boy and his fugitive father were living in a filthy compound on their property.
But police did not carry out a search of the property until last week, Taos County Sheriff Jerry Hogrefe said, because they did not believe they had probable cause in the case.
That delay is facing scrutiny in the wake of Monday's discovery of a young boy's remains at the compound.
"They were dragging their feet.
They were taking too long," said Tanya Badger, who with her husband, Jason, told authorities about the boy's suspected presence at the compound.
"Even if they were trying to build a case or whatnot, a child's life is at stake."
It's not clear whether the remains are those of Abdul-Ghani Wahhaj, a child with severe medical problems who disappeared from Georgia about nine months ago.
The remains were discovered in a wretched compound along with 11 starving children, authorities said.
More details about the horrid compound could be revealed Wednesday when the five adults arrested from the site make their first court appearances.
Authorities raided the compound in Amalia, New Mexico, on Friday as part of their search for Abdul-Ghani, whose father, Siraj Wahhaj, allegedly abducted him from Georgia in November.
Neighbors raised alarm about a suspect
"""

class prta_system(torch.nn.Module):
    def __init__(self,propodict=None):
        super(prta_system, self).__init__()
        # (1, 512,768) # (sentence, max len, dimensions)
        sentence_list_yn = ["Non-prop", "Prop"] # prop or non prop?
        self.bert_model = BertModel.from_pretrained('bert-base-multilingual-cased')
        #self.propaganda_classes = ["<PAD>", "O", "Name_Calling,Labeling", "Repetition", "Slogans", "Appeal_to_fear-prejudice", "Doubt", "Exaggeration,Minimisation", "Flag-Waving", "Loaded_Language", "Reductio_ad_hitlerum", "Bandwagon", "Causal_Oversimplification", "Obfuscation,Intentional_Vagueness,Confusion", "Appeal_to_Authority", "Black-and-White_Fallacy", "Thought-terminating_Cliches", "Red_Herring", "Straw_Men", "Whataboutism"]
        if propodict:
            classnum = len(list(propodict.keys()))
        else:
            classnum = len(propaganda_classes)
        self.propo_layer = torch.nn.Linear(768,classnum) # for each of the things.
        self.dropout = torch.nn.Dropout(0.2,inplace=False) #20% dropout chance..
        self.sigmoid = torch.nn.Sigmoid()
        self.gate = torch.nn.Linear(768, 2) # goes from 768 to a yes no
        self.gate_final = torch.nn.Linear(2,1) # goes to a single yes no decider linear layer.


    def forward(self, input_ids, attention_mask=None):
        tagouts, sentence_propa_detected = self.bert_model(input_ids, attention_mask=attention_mask)#!!!!!!!
        tagouts = self.dropout(tagouts) #dropout from bert.
        wordbased_propaganda = self.propo_layer(tagouts)
        initial_gate_output = self.gate(sentence_propa_detected)# get 768-> 2 output
        gate_final_output = self.gate_final(initial_gate_output) # get 2 -> 1 output for gate.
        sigmoided = self.sigmoid(gate_final_output)
        token_level_out = torch.mul(sigmoided.unsqueeze(1), wordbased_propaganda) # perform multiply to obtain weighting of all outputs.        
        logits_for_proppin = [token_level_out, gate_final_output]
        # relevant logits are: (wordtag after weightage from verdict on sentence relevance, final verdict on whether a sentence is relevant, after RELU)
        final_argmax_output = [logits_for_proppin[i].argmax(-1) for i in range(2)]
        # argmax of token_level_out [batch_size,totalwords,propaganda_chosen], argmax of gate_final_output(propaganda or not???)
        #print(logits_for_proppin[0].shape)
        #print(logits_for_proppin[1].shape)
        #torch.Size([4, 512, 6])
        #torch.Size([4, 1])

        return logits_for_proppin, final_argmax_output



class BertMultiTaskLearning(PreTrainedBertModel):
    def __init__(self, config):
        super(BertMultiTaskLearning, self).__init__(config)
        self.bert = BertModel.from_pretrained('bert-base-multilingual-cased')
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.ModuleList([nn.Linear(config.hidden_size, itemiser[i]) for i in range(len(itemiser))])
        self.apply(self.init_bert_weights)
        self.masking_gate = nn.Linear(2, 1)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        sequence_output, pooled_output = self.bert(input_ids, attention_mask=attention_mask)
        sequence_output = self.dropout(sequence_output)
        pooled_output = self.dropout(pooled_output)
        token_level = self.classifier[0](sequence_output)
        sen_level = self.classifier[1](pooled_output)
        gate = sigmoid(self.masking_gate(sen_level))
        dup_gate = gate.unsqueeze(1).repeat(1, token_level.size()[1], token_level.size()[2])
        wei_token_level = torch.mul(dup_gate, token_level)
        logits = [wei_token_level, gate]
        y_hats = [logits[i].argmax(-1) for i in range(2)]
        #print(logits[0].shape)
        #print(logits[1].shape)
        # torch.Size([4, 512, 6])
        # torch.Size([4, 1])

        
        return logits, y_hats




def predict_for_sample(tokenizer,inputitem,model,target_device,idx2tag):
    encoded_version = tokenizer.encode(inputitem, pad_to_max_length = False) # don't pad it.
    string_tokens = tokenizer.tokenize(inputitem)

    if len(encoded_version)<512:
        origin_length = len(encoded_version) # get true length
        encoded_version.extend([0]*(512 - len(encoded_version))) #pad
        attentionmask = torch.tensor([1]*origin_length+[0]*(512 - origin_length)).unsqueeze(0).to(target_device)
    else:
        encoded_version = encoded_version[:511]
        encoded_version.append(102) # add a SEP token to the end of tokenised list after the trunc
        attentionmask = torch.tensor([1]*511+[0]).unsqueeze(0).to(target_device)
        origin_length = 511
        string_tokens = string_tokens[:511] 
    with torch.no_grad():
        output = model(torch.tensor(encoded_version).unsqueeze(0).to(target_device),attentionmask)
        _, argmaxed_output = output[:origin_length]
        classes = argmaxed_output[0]
        propaganda_flag = argmaxed_output[1].squeeze(0)
        actual_classes = classes[:origin_length]
        
        print("Probability of propaganda presence:",propaganda_flag)

        tagsoutlist = []

        propaganda_probability = torch.nn.Sigmoid()(propaganda_flag.cpu().float())
        for i in classes.squeeze(0):
            tagsoutlist.append(idx2tag[i.cpu().item()])
        
        for counter in range(len(string_tokens)):
            print(string_tokens[counter],tagsoutlist[counter])

if __name__=="__main__":
    #torch.set_printoptions(threshold=100000)


    demo=False
    grouped=True


    propaganda_classes = ["<PAD>", "O", "Name_Calling,Labeling", "Repetition", "Slogans", "Appeal_to_fear-prejudice", "Doubt", "Exaggeration,Minimisation", "Flag-Waving", "Loaded_Language", "Reductio_ad_hitlerum", "Bandwagon", "Causal_Oversimplification", "Obfuscation,Intentional_Vagueness,Confusion", "Appeal_to_Authority", "Black-and-White_Fallacy", "Thought-terminating_Cliches", "Red_Herring", "Straw_Men", "Whataboutism"]  
    sentence_yn = ["Non-prop", "Prop"] # prop or non prop?
    propaganda_tag2idx = {tag:idx for idx, tag in enumerate(propaganda_classes)}

    propaganda_idx2tag = {idx:tag for idx, tag in enumerate(propaganda_classes)}
    if grouped:
        propaganda_idx2tag = {0:["<PAD>"], 1:["O"], 2:["Repetition" ,"Slogans", "Appeal_to_fear-prejudice", "Flag-Waving", "Reductio_ad_hitlerum", "Bandwagon", "Appeal_to_Authority"], 3:["Causal_Oversimplification", "Obfuscation,Intentional_Vagueness,Confusion", "Black-and-White_Fallacy", "Thought-terminating_Cliches"], 4:["Name_Calling,Labeling", "Doubt", "Loaded_Language", "Straw_Men"], 5:["Exaggeration,Minimisation", "Red_Herring", "Whataboutism"]}

        propaganda_tag2idx = {"<PAD>":0, "O":1, "Repetition":2 , "Slogans":2, "Appeal_to_fear-prejudice":2, "Flag-Waving":2, "Reductio_ad_hitlerum":2, "Bandwagon":2, "Appeal_to_Authority":2, "Causal_Oversimplification":3, "Obfuscation,Intentional_Vagueness,Confusion":3, "Black-and-White_Fallacy":3, "Thought-terminating_Cliches":3, "Name_Calling,Labeling":4, "Doubt":4, "Loaded_Language":4, "Straw_Men":4, "Exaggeration,Minimisation":5, "Red_Herring":5, "Whataboutism":5}
    itemiser = [len(list(propaganda_idx2tag.keys())),len(sentence_yn)]
    sentence_yn_tag2idx = {tag:idx for idx, tag in enumerate(sentence_yn)}
    sentence_yn_idx2tag = {idx:tag for idx, tag in enumerate(sentence_yn)}
    #tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    batch_size = 8
    total_epochs = 91
    
    if torch.cuda.is_available():
        target_device = torch.device("cuda")
    else:
        target_device = torch.device("cpu")

    #target_device = torch.device("cpu")

    if not "train.pickle" in os.listdir() or not "test.pickle" in os.listdir() or not "val.pickle" in os.listdir():
        datathon_target_files = ["dev","train"]
        test_files = ["test"]
        train_dev_articleslices = []
        train_dev_article_cutouts_labels = []
        train_dev_propo_or_no = []
            
        print("preparing semeval")
        semeval_sentences, semeval__labels, semeval__yn, semeval_tokenised_sentences, semeval_tokenised_labels = semeval_parsin(tokenizer,"semeval_2020_11_train.txt")
        semeval_sentences_dev, semeval__labels_dev, semeval__yn_dev, semeval_tokenised_sentences_dev, semeval_tokenised_labels_dev = semeval_parsin(tokenizer,"semeval_2020_11_dev.txt")
        print("sameval completed.")

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
        test_tokenarticles,test_tokenlabels = datathon_to_full_set(test_articleslices,test_article_cutouts_labels,tokenizer)
        print("test datathon files completed.\n\n")

        total_train_articles = train_dev_tokenarticles+semeval_tokenised_sentences
        total_train_labels = train_dev_tokenlabels+semeval_tokenised_labels
        pickle.dump([total_train_articles,total_train_labels],open("train.pickle","wb"))
        pickle.dump([semeval_tokenised_sentences_dev,semeval_tokenised_labels_dev],open("val.pickle","wb"))
        pickle.dump([test_tokenarticles,test_tokenlabels],open("test.pickle","wb"))


    if demo:
        model = BertMultiTaskLearning.from_pretrained('bert-base-cased')
        tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
        model.load_state_dict(torch.load("grouped_prta0.0059697711387112955_50_0.025991644460410565.torch"))
        model = model.to(target_device)
        model.eval()
        inputitem=""""The Forward, a formerly Jewish paper which today specializes in attacking Jews for an audience of lefty hipsters, decided to do a drive-by attack on Jordan Peterson. “Is Jordan Peterson Enabling Jew Hatred?”  Jordan Peterson isn't. The Forward is.  In recent months, the Forward chose to run an editorial defending radical leftist Jeremy Corbyn even while British Jews were protesting his anti-Semitism. ""Leave Jeremy Corbyn Alone,"" it declared. It ran another editorial defending the Polish government's historical revisionism and censorship. And then there's, ""3 Jewish Moguls Among Eight Who Own as Much as Half the Human Race.""  Stormfront couldn't have improved on that headline.  When a D.C. councilman claimed that Jews control the weather, the Forward ran, ""The Shameful Character Assassination Of Trayon White"" and ""Why We Should Applaud The Politician Who Said Jews Control The Weather"".  None of this counts the compulsive pro-Hamas propaganda, the celebration of anti-Israel hate groups like JVP and If Not Now. Or its efforts to whitewash the recent Hamas ""march"" attacks on Israel. We're not even talking about Israel here. Just plain, old-fashioned Jew-hatred.  The stuff that The Forward, especially under Jane Eisner, has come to specialize in.  And being accused of white nationalism by The Forward is meaningless. This is the same anti-Semitic tabloid that decided to run a piece accusing Jews of white nationalism.  No, seriously. It was titled, ""White Nationalism Is Spreading In The Orthodox Community.""  I haven't linked to anything because it's my policy not to link to hate sites. And that's what the Forward is.  The only mistake Jordan Peterson made was talking to the Forward in the first place." """

        #inputitem="""The Biden administration revealed Monday that it will carry out an interagency review of scientific policies in place across the federal government, including some implemented by President Donald Trump’s administration that "eschewed scientific integrity." The review will be led by the White House Office of Science and Technology Policy, which said that some of the policies put in place and followed by the previous administration would be up for review. "OSTP’s Scientific Integrity Task Force will be taking a whole-of-government, forward-looking review of science across federal agencies, in part by examining practices that were antithetical to that mission over the last four years – including Trump-era policies that eschewed scientific integrity in favor of politics," Julia Krieger, spokesperson for the White House Office of Science and Technology Policy, said in a statement obtained by Fox News. "By inaugurating an inclusive review process by federal scientists of all backgrounds, OSTP is dedicated to building on lessons learned to better serve the American people with practical, science-based results." In a memo released on Monday, the agency noted that it will review the policies to determine whether they prevent improper political interference, prevent the suppression or distortion of information, support diversity among scientists, and advance equitable delivery of the government’s programs. The Trump administration came under fire for its policies regarding science and data, most notably during the coronavirus pandemic. The former president was accused of downplaying the virus and potentially misrepresenting data, as well as misleading the public on the efficacy of some key health measures, including masks and potential treatments.Biden has already undone a number of measures taken by the former president, including rejoining the Paris Climate Accord and the World Health Organization. """

        #inputitem="""China has passed sweeping changes to Hong Kong's electoral rules which will tighten its control over the city. The number of directly elected seats in parliament has been cut almost by half, and prospective MPs will first be vetted by a pro-Beijing committee to ensure their loyalty to the mainland. The aim is to ensure only "patriotic" figures can run for positions of power. Critics warn it will mean the end of democracy, as it removes all opposition from the city parliament. On Tuesday, Chinese state media reported that the country's top decision-making body, the NPC Standing Committee, voted unanimously to pass it. This amends the annexes of Hong Kong's mini-constitution, the Basic Law. All opposition to the Chinese Communist Party has effectively been obliterated from formal politics in Hong Kong. When these changes were announced a few weeks ago, they drew criticism internationally. On Twitter, Australia's Foreign Minister Marise Payne spoke of her country's concern that the new arrangements in the territory would "weaken its democratic institutions". That was being kind. The US Secretary of State, Antony Blinken, described the moves as an "assault on democracy". The truth is, the city's electoral system was already rigged to ensure that the "pro-Beijing" camp could never lose control of the mini-parliament, the Legislative Council. Its leader, the chief executive, was already directly chosen by a pro-Beijing committee. The last time the only real elections in Hong Kong were held - those for local governments - the pro-democracy camp took control of all but one district council right across the city. This clearly spooked Beijing so now they've stopped pro-democracy candidates from standing at all."""
        predict_for_sample(tokenizer,inputitem,model,target_device,propaganda_idx2tag)
        quit()






    # load all stuff from pickles.
    total_train_articles,total_train_labels = pickle.load(open("train.pickle","rb"))
    semeval_tokenised_sentences_dev,semeval_tokenised_labels_dev = pickle.load(open("val.pickle","rb"))
    test_tokenarticles,test_tokenlabels = pickle.load(open("test.pickle","rb"))

    train_dataset = sentences_dataset(total_train_articles,total_train_labels)
    val_dataset = sentences_dataset(semeval_tokenised_sentences_dev,semeval_tokenised_labels_dev)
    test_dataset = sentences_dataset(test_tokenarticles,test_tokenlabels)

    train_dataloader = torch.utils.data.DataLoader(train_dataset,batch_size=batch_size, shuffle=True,num_workers=0)
    val_dataloader = torch.utils.data.DataLoader(val_dataset,batch_size=batch_size, shuffle=True,num_workers=0)
    test_dataloader = torch.utils.data.DataLoader(test_dataset,batch_size=batch_size, shuffle=True,num_workers=0)
    #model = prta_system(propaganda_idx2tag)
    #print(model.bert_model)
    model = BertMultiTaskLearning.from_pretrained('bert-base-cased')
    #print("-"*900)
    #print(model.bert)
    model = model.to(target_device)


    o_weightage = 0.003
    weightages_criterion = [0,o_weightage]+[(1-o_weightage)/(len(list(propaganda_idx2tag.keys()))-2)]*(len(list(propaganda_idx2tag.keys()))-2)
    #  actual group sample proportions: 0.254,0.153,0.492,0.101
    weightages_criterion = [0,o_weightage,0.2,0.4,0.097,0.3]
                    # 2 because we ignore O and PAD
    criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor(weightages_criterion).to(target_device),ignore_index=0)
    binary_criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([3932/14263]).to(target_device)) # this implies they had a batch size of 1...
    #optimizer = AdamW(model.parameters(), lr=3e-5)
    num_train_optimization_steps = int(len(train_dataset) / batch_size ) * total_epochs
    param_optimizer = list(model.named_parameters())
    param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=3e-5,
                         warmup=0.1,
                         t_total=num_train_optimization_steps)
    








    # predict_for_sample(tokenizer,article_to_diagnose_test,model,target_device,propaganda_idx2tag) # method to run model on text sentences
    for epoch in range(total_epochs):
        model.train()
        examples_run_counter = 0
        training_summed_loss = 0
        train_tag_tracker={}
        val_tag_tracker={}
        for i in propaganda_idx2tag:
            train_tag_tracker[i] = {0:0,1:0,2:0,"tag":propaganda_idx2tag[i]}
            val_tag_tracker[i] = {0:0,1:0,2:0,"tag":propaganda_idx2tag[i]}
        #print(train_tag_tracker)
        def sum_n_acc(trackerdict):
            bads=0
            goods=0
            for trackerdictid in trackerdict:
                if trackerdictid==1 or trackerdictid==0:
                    continue # ignore.
                bads+=trackerdict[trackerdictid][0]
                goods+=trackerdict[trackerdictid][1]
            return goods/(bads+goods), bads, goods
        print("------------------beginning epoch------------------")
        for idx,(articles,labels) in enumerate(train_dataloader):
            optimizer.zero_grad()
            entity_lengths = []
            binary_losses = 0

            propaganda_losses = 0
            for i in range(len(articles)):
                entity_lengths.append(search_len(articles[i])) # get actual length of sequence
                examples_run_counter+=entity_lengths[-1]
            attn_mask = (labels>=1).long() # attention mask generation
            articles = articles.to(target_device)
            attn_mask = attn_mask.to(target_device)
            logits_out, argmaxed_output = model(articles,attn_mask)
            for logit_instance_counter in range(len(logits_out[1])):
                binary_losses += binary_criterion(logits_out[1][logit_instance_counter],propo_or_no(labels).float().to(target_device))
            labels = labels.long().to(target_device)
            #print("-----------------------------------for loop begins ----------------------------------------")
            for logit_instance_counter in range(len(logits_out[0])):
                #print(articles[logit_instance_counter])
                #print(logits_out[0][logit_instance_counter][:entity_lengths[logit_instance_counter]].shape)
                #print(labels[logit_instance_counter][:entity_lengths[logit_instance_counter]])
                #print("---------------------------one for loop ends-------------------------------------------")
                propaganda_losses += criterion(logits_out[0][logit_instance_counter][:entity_lengths[logit_instance_counter]],labels[logit_instance_counter][:entity_lengths[logit_instance_counter]])
            for instance_counter_acc in range(len(logits_out[0])):
                for output_counter_acc in range(entity_lengths[instance_counter_acc]):
                    actual_predicted_ = int(torch.argmax(logits_out[0][instance_counter_acc][:entity_lengths[instance_counter_acc]][output_counter_acc]))
                    train_tag_tracker[actual_predicted_][2] = train_tag_tracker[actual_predicted_][2] + 1  # count number of times this tag appeared

                    if labels[instance_counter_acc][:entity_lengths[instance_counter_acc]][output_counter_acc]>1: 
                        randomsample =logits_out[0][instance_counter_acc][:entity_lengths[instance_counter_acc]][output_counter_acc]
                    if actual_predicted_==labels[instance_counter_acc][:entity_lengths[instance_counter_acc]][output_counter_acc]:
                        train_tag_tracker[int(labels[instance_counter_acc][:entity_lengths[instance_counter_acc]][output_counter_acc])][1] = train_tag_tracker[int(labels[instance_counter_acc][:entity_lengths[instance_counter_acc]][output_counter_acc])][1] + 1 
                    else:
                        train_tag_tracker[int(labels[instance_counter_acc][:entity_lengths[instance_counter_acc]][output_counter_acc])][0] = train_tag_tracker[int(labels[instance_counter_acc][:entity_lengths[instance_counter_acc]][output_counter_acc])][0] + 1
                
            training_summed_loss+=binary_losses.item()
            training_summed_loss+=propaganda_losses.item()
            batch_summed_loss = 0.1*binary_losses+0.9*propaganda_losses
            batch_summed_loss.backward()
            optimizer.step()
            if examples_run_counter%(batch_size%1000000)==0:
                print("latest loss:"+str(batch_summed_loss))
                print("random sample of a propaganda:", randomsample)
        print("------------------Training completed------------------")
        print(train_tag_tracker)
        acc, misses, hits = sum_n_acc(train_tag_tracker)
        print("general accuracy:",acc)
        print("general misses:",misses)
        print("general hits:",hits)

        print("total epoch loss: ",training_summed_loss)
        mean_training = training_summed_loss/examples_run_counter
        print("average train loss:",mean_training)
        print("validating..")
        model.eval()
        val_examples_run_counter = 0
        val_summed_loss = 0

        with torch.no_grad():
            for idx,(articles,labels) in enumerate(val_dataloader):
                entity_lengths = []
                binary_losses = 0

                propaganda_losses = 0
                for i in range(len(articles)):
                    entity_lengths.append(search_len(articles[i])) # get actual length of sequence
                    val_examples_run_counter+=entity_lengths[-1]
                attn_mask = (labels>=1).long() # attention mask generation
                articles = articles.to(target_device)
                attn_mask = attn_mask.to(target_device)
                logits_out, argmaxed_output = model(articles,attn_mask)
                for logit_instance_counter in range(len(logits_out[1])):
                    binary_losses += binary_criterion(logits_out[1][logit_instance_counter],propo_or_no(labels).float().to(target_device))
                labels = labels.long().to(target_device)
                for logit_instance_counter in range(len(logits_out[0])):
                    propaganda_losses += criterion(logits_out[0][logit_instance_counter][:entity_lengths[logit_instance_counter]],labels[logit_instance_counter][:entity_lengths[logit_instance_counter]])
                val_summed_loss+=binary_losses.item()
                val_summed_loss+=propaganda_losses.item()
                for instance_counter_acc in range(len(logits_out[0])):
                    for output_counter_acc in range(entity_lengths[instance_counter_acc]):
                        actual_predicted_ = int(torch.argmax(logits_out[0][instance_counter_acc][:entity_lengths[instance_counter_acc]][output_counter_acc]))
                        if actual_predicted_==labels[instance_counter_acc][:entity_lengths[instance_counter_acc]][output_counter_acc]:
                            val_tag_tracker[int(labels[instance_counter_acc][:entity_lengths[instance_counter_acc]][output_counter_acc])][1] = train_tag_tracker[int(labels[instance_counter_acc][:entity_lengths[instance_counter_acc]][output_counter_acc])][1] + 1 
                        else:
                            val_tag_tracker[int(labels[instance_counter_acc][:entity_lengths[instance_counter_acc]][output_counter_acc])][0] = train_tag_tracker[int(labels[instance_counter_acc][:entity_lengths[instance_counter_acc]][output_counter_acc])][0] + 1
            mean_val_loss = val_summed_loss/val_examples_run_counter
            print("------------------Validation completed------------------")
        print(val_tag_tracker)
        acc, misses, hits = sum_n_acc(val_tag_tracker)
        print("general accuracy:",acc)
        print("general misses:",misses)
        print("general hits:",hits)
        print("An epoch was completed.")
        if epoch%10 ==0:
            torch.save(model.state_dict(),"grouped_prta"+str(mean_training)+"_"+str(epoch)+"_"+str(mean_val_loss)+".torch")
            print("saved the model")
