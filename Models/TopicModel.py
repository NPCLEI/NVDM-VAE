from Models.VAE import VAE
import torch
import Config
from DataLoader.IMDB import Loader as IMDB
from torch.utils.data import DataLoader
from Models.WordEmbedding import OrTokenizer, one_hot
import utils
import matplotlib.pyplot as plt

def TrainNVMD(imdbdataset,net:VAE = None) -> VAE:
    loader  = DataLoader(imdbdataset, batch_size = Config.batch_size, shuffle=True)
    if net == None:
        net = VAE(dtype="NVMD")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("[npc report] your device is ",device)
    net.device = device
    net.to(device)

    optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)
    mean_losses = []
    perlx = []

    # last_loss,count_last_loss = 10,0
    for t in range(4):
        t_losses = []
        batch_count = 0
        for item in loader:
            # print(item)
            x,y = item

            re_x,mu,logvar = net(x.to(device))

            loss = VAE.loss(x.to(device),re_x.to(device),mu.to(device),logvar.to(device),net.recon_loss_fc)     # must be (1. nn output, 2. target), the target label is NOT one-hotted
            
            t_losses.append(loss.tolist())

            optimizer.zero_grad()   # clear gradients for next train
            loss.backward()         # backpropagation, compute gradients
            optimizer.step()        # apply gradients
            
            mean_losses.append(sum(t_losses)/len(t_losses))
            batch_count += 1
            print("[npc report]","echo:",t,
            "(%d/%d[%2.2f%%])"%(
                    batch_count,
                    len(imdbdataset)/Config.batch_size,
                    100*batch_count/(len(imdbdataset)/Config.batch_size)
                ),
                "loss:",mean_losses[-1])
            perlx.append(Perplexity(len(loader),50,t_losses))
        print("[npc report]","echo",t,"perplexity:",perlx[-1])
        net.save()

    plt.plot(range(len(perlx)),perlx,color = 'red')
    plt.show()
    return net

from tokenizers import CharBPETokenizer
from torch.nn import functional
from Models import WordEmbedding
def Gen_Topic_W(topicVAE:VAE,dataset,tokenizer:CharBPETokenizer=None,printRes = False):
    orTokenizer = OrTokenizer(dataset)
    topicVAE.to(Config.device)
    res = []
    for word in orTokenizer:
        # print("?",tokenizer.encode(word))
        try:
            oh = WordEmbedding.one_hot(word,tokenizer,Config.vocLen).to(dtype=torch.float32)
            tp = VAE.toInference(topicVAE,torch.unsqueeze(oh,0).to(Config.device)).to(torch.device('cpu'))

            res.append([word,tp.tolist()[0],torch.argmax(tp[0]).tolist()])
            if printRes:
                print(res[-1])
        except Exception as e:
            print("[npc report eorr]",oh.shape,"vae inferece eorr:",e,"auto handle:skip")

    return res

def Gen_Topic_W_article(topicVAE:VAE,dataset:IMDB,tokenizer:CharBPETokenizer=None):
    topicVAE.to(Config.device)
    from torch.utils.data import DataLoader
    
    imdbloader = DataLoader(dataset, batch_size = 1, shuffle=False)

    res = []
    ix = 0
    for emb,lab in imdbloader:
        try:
            tp = VAE.toInference(topicVAE,emb.to(Config.device)).to(torch.device('cpu'))

            res.append([dataset.table[ix],lab,tp.tolist()])
            ix += 1
        except Exception as e:
            print("[npc report eorr]",dataset.table[ix],"vae inferece eorr:",e,"auto handle:skip")
    return res

def Collect_topic_words(W,topic = 4,line = 0.7):
    res = []
    for word,prob in W:
        if prob[topic] == max(prob) and word != '' and prob[topic] > line:
            res.append((word,prob[topic])) 
    res.sort(key=lambda x:x[1],reverse=True)
    return res

import math
def Perplexity(num_of_doc,doc_len,losses):
    return math.exp((-1/num_of_doc)*(1/doc_len)*sum(losses))
