import os

propaganda_classes = ["<PAD>", "O", "Name_Calling,Labeling", "Repetition", "Slogans", "Appeal_to_fear-prejudice", "Doubt", "Exaggeration,Minimisation", "Flag-Waving", "Loaded_Language", "Reductio_ad_hitlerum", "Bandwagon", "Causal_Oversimplification", "Obfuscation,Intentional_Vagueness,Confusion", "Appeal_to_Authority", "Black-and-White_Fallacy", "Thought-terminating_Cliches", "Red_Herring", "Straw_Men", "Whataboutism"]  
sentence_yn = ["Non-prop", "Prop"] # prop or non prop?
propaganda_tag2idx = {tag:idx for idx, tag in enumerate(propaganda_classes)}
propaganda_idx2tag = {idx:tag for idx, tag in enumerate(propaganda_classes)}
sentence_yn_tag2idx = {tag:idx for idx, tag in enumerate(sentence_yn)}
sentence_yn_idx2tag = {idx:tag for idx, tag in enumerate(sentence_yn)}
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

all_sentences = []
all_labels = []
all_yn =[]
for opened_file in [open("semeval_2020_11_dev.txt","r",encoding="utf-8"),open("semeval_2020_11_train.txt","r",encoding="utf-8")]:
    with opened_file as trainfile:
        sentence_latest = []
        label_latest = []
        haspropo = False
        for line in trainfile:
            a = line.split()
            if len(a)>1:
                sentence_latest.append(a[0])
                label_latest.append(propaganda_tag2idx[a[1]])
                if propaganda_tag2idx[a[1]]>1:
                    haspropo=True
            else:
                all_sentences.append(sentence_latest)
                all_labels.append(label_latest)
                if haspropo:
                    all_yn.append(1)
                else:
                    all_yn.append(0)
                sentence_latest = []
                label_latest = []
                haspropo = False
for i in range(len(all_yn)):
    print(all_sentences[i])
    print(all_labels[i])
    print(all_yn[i])



