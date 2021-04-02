import os


propaganda_classes = ["<PAD>", "O", "Name_Calling,Labeling", "Repetition", "Slogans", "Appeal_to_fear-prejudice", "Doubt", "Exaggeration,Minimisation", "Flag-Waving", "Loaded_Language", "Reductio_ad_hitlerum", "Bandwagon", "Causal_Oversimplification", "Obfuscation,Intentional_Vagueness,Confusion", "Appeal_to_Authority", "Black-and-White_Fallacy", "Thought-terminating_Cliches", "Red_Herring", "Straw_Men", "Whataboutism"]  
sentence_yn = ["Non-prop", "Prop"] # prop or non prop?
propaganda_tag2idx = {tag:idx for idx, tag in enumerate(propaganda_classes)}
propaganda_idx2tag = {idx:tag for idx, tag in enumerate(propaganda_classes)}
sentence_yn_tag2idx = {tag:idx for idx, tag in enumerate(sentence_yn)}
sentence_yn_idx2tag = {idx:tag for idx, tag in enumerate(sentence_yn)}
fileslist = os.listdir()
articles = []
# article_cutouts = []
# startstops = []
# article_cutoutslabels = []
articleslices = []
propo_or_no=[]
article_cutouts_labels = []
for i in fileslist:
    if ".txt" in i:
        articles.append(i.split(".")[0])
for article in articles:
    labels = open(article+".labels.tsv","r",encoding="utf-8")
    article = open(article+".txt","r",encoding="utf-8")
    # startstops_latest = []
    # article_cutoutlabels_latest = []
    # article_cutouts_latest=[]
    # articlestring = article.read()
    # for line in labels:
        # if len(line)>5:
            # singularitem = line.split()
            # latest_label = singularitem[1]
            # startstops_latest.append(int(singularitem[2]),int(singularitem[3]))
            # article_cutoutlabels_latest.append(sentence_yn_tag2idx[singularitem[1]])
            # article_cutouts_latest.append(articlestring[int(singularitem[2]):int(singularitem[3])])
    # articles.append(articlestring) # get the entire article.
    # article_cutouts.append(article_cutouts_latest) # get each of the individual phrases
    # startstops.append(startstops_latest) # for each of the character start stop things
    # article_cutoutslabels.append(article_cutoutlabels_latest) # for each of the article cutouts labels, as a list.
    
    
    # Version 2 of labels, more usable as a dataset
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
            
    # finish up all unused stuffs...
    if lastcount!=articlelength-1:
        propo_or_no_latest.append(1)
        article_cutouts_latest.append(articlestring[lastcount:articlelength-1])
        article_cutouts_labels_latest.append(propaganda_tag2idx["O"])
    articleslices.append(article_cutouts_latest)
    propo_or_no.append(propo_or_no_latest)
    article_cutouts_labels.append(article_cutouts_labels_latest)
    article.close()
    # for i in article_cutouts_latest:
        # print(i)
        # print("-----"*10)
    # print(article_cutouts_labels_latest)
    # print(propo_or_no_latest)
    # print("\n")
    # input()
