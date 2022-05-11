import torch

def topic_model_cls_article_num(topic_w_article):
    clsdict = {0:[],1:[],2:[]}
    for item in topic_w_article:
        clsdict[torch.argmax(torch.tensor(item[2])).tolist()].append(item[1].tolist()[0])
    for k,v in clsdict.items():
        print(k,len(v),listItemFrequence(v))

def listItemFrequence(list):
    res = {}
    for item in list:
        if item not in res.keys():
            res.setdefault(item,1)
        else:
            res[item] += 1
    return res

def topic_model_cls_word_num(topic_w):
    clsdict = {0:[],1:[],2:[]}
    #res.append([word,tp.tolist(),torch.argmax(tp).tolist()])

    for item in topic_w:
        clsdict[item[2]].append(item[1])
    for k,v in clsdict.items():
        print(k,len(v))

def get_related_word(word,topic_w):
    word_item = None
    for tw in topic_w:
        if tw == word:
            word_item = tw
            break
    if word_item == None:
        print("[unhandled eorr]:",word,"not in topic_w")
        raise Exception

    word_item_tpz = torch.tensor(word_item[1])
    distance = []
    for tw in topic_w:
        w,t,tz = tw
        t = torch.tensor(t)
        td = torch.dist(t,word_item_tpz)
        if td < 0.1:
            distance.append((w,td))
    
    distance = sorted(distance,key=lambda x:x[1])

    for w,d in distance:
        print(w,d)

    return distance 
