from tokenizers import CharBPETokenizer
import pickle
import torch
import jieba
import Config

def Train(txt_paths:list,save_path = Config.envir_path):
    tokenizer = CharBPETokenizer()

    tokenizer.train(txt_paths)

    print(save_path)
    tokenizer.save("%s/ModelPickle/json/topic-model.tokenizer.json"%save_path)

    with open("%s/ModelPickle/topic_token.pickle"%save_path,"wb+") as tt:
        pickle.dump(tokenizer,tt)
    
    return tokenizer

class OrSentence:
    def __init__(self,ids) -> None:
        self.ids = ids

class OrTokenizer:
    def __init__(self,dataset) -> None:

        self.dct = OrTokenizer.Train(dataset)
        self.len = len(self.dct) + 1
        self.keys = list(self.dct.keys())
    
    def __getitem__(self,ix):
        return self.keys[ix]
    
    def __len__(self):
        return len(self.keys)

    def listIndex(self):
        count = 0
        for word in self.dct.keys():
            self.dct[word] = count
            count += 1

    def encode(self,sentence:str):
        ws = list(jieba.cut(sentence))
        res = []
        for w in ws:
            if w in self.dct.keys():
                res.append(self.dct[w])
            else:
                res.append(self.len)
        return OrSentence(res)

    def decode(self,ids:list):
        res = []
        for ix in ids:
            if ix < self.len - 1:
                res.append(self.keys[ix])
            else:
                res.append("[None]")
        return res
    
    def Train(dataset):
        dataset.getToken(False)
        res = {}
        for ix in range(len(dataset)):
            txt,label = dataset[ix]
            if txt == -1:
                continue
            spws = list(jieba.cut(txt))
            for word in spws:
                word = word.lower()
                if word in res.keys():
                    res[word]+=1
                else:
                    res.setdefault(word,0)
        dataset.getToken()
        return res
        
def TrainWithoutBPE(dataset,save_path = "%s/ModelPickle"%(Config.envir_path)):
    tokenizer = OrTokenizer(dataset)
    with open("%s/topic_token.pickle"%save_path,"wb+") as tt:
        pickle.dump(tokenizer,tt)
    return tokenizer



def one_hot(xs,tokenizer,ont_hot_len):
    try:
      te = tokenizer.encode(xs)
      teids = te.ids
      #去重
      t_dic = {}
      for ix in teids:
          t_dic.setdefault(ix,0)
      oh = torch.nn.functional.one_hot(torch.tensor(list(t_dic.keys())), num_classes=ont_hot_len).to(dtype=torch.float32)
      # print(te)
      return oh.sum(axis = 0)
    except Exception as e:
        print("[npc report eorr]",te.ids,ont_hot_len)
        raise e

if __name__=="__main__":

    # Initialize a tokenizer
    tokenizer = CharBPETokenizer()

    # Then train it!
    tokenizer.train([ "./path/to/files/1.txt", "./path/to/files/2.txt" ])

    # Now, let's use it:
    encoded = tokenizer.encode("I can feel the magic, can you?")

    # And finally save it somewhere
    tokenizer.save("./ModelPickle/json/topic-model.tokenizer.json")


# from tokenizers import CharBPETokenizer

# # Initialize a tokenizer
# vocab = "./path/to/vocab.json"
# merges = "./path/to/merges.txt"
# tokenizer = CharBPETokenizer(vocab, merges)

# # And then encode:
# encoded = tokenizer.encode("I can feel the magic, can you?")
# print(encoded.ids)
# print(encoded.tokens)


# ##################查看数据分布###################
# maxlen = 0
# maxid = 0
# maxtokens = None
# mean = []

# def findMaxMinArticle(txt,label):
#     global maxlen,maxtokens,mean,maxid
#     res = topic_token.encode(txt)
#     res_len = len(res)
#     tmaxid = max(res.ids)
#     if tmaxid > maxid:
#         maxid = tmaxid
#     mean.append(res_len)
#     if res_len > maxlen:
#         maxlen = res_len
#         maxtokens = res

# imdbloader.throughDataBeforeRead(findMaxMinArticle,1000)
# print(maxlen)
# print(maxtokens.ids)
# print(sum(mean)/len(mean))
# print(maxid)

# # plt.plot(mean,[i for i in range(len(mean))])
# # plt.show()