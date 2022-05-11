from os import path
import pickle
import Config

def CheckModel(mode_name,train_func,continue_train = False,retrain = False):
    f_path = "%s/ModelPickle/%s.pickle"%(Config.envir_path,mode_name)
    model = None
    if path.exists(f_path) and not retrain:
        print("[npc report] model:%s ,file exists,loading"%(mode_name),end = '')
        with open(f_path,"rb+") as model:
            model = pickle.load(model)
            print("... loaded")
            if continue_train:
                print("[npc report] model:%s ,file exists,user choose to continue train the model"%(mode_name))
                model = train_func(model)
                SaveObj(model,mode_name)
    else:
        print("[npc report] model:%s ,file not exists,try to train the model"%(mode_name))
        model = train_func(None)
        SaveObj(model,mode_name)
    return model


def SaveObj(obj,name = "obj",path = Config.envir_path):
    import pickle
    f_name = "%s/ModelPickle/%s.pickle"%(path,name)
    with open(f_name, 'wb+') as net_file:
        pickle.dump(obj,net_file)






# ##################查看数据分布###################
# maxlen = 0
# maxtokens = None
# mean = []

# def findMaxMinArticle(txt,label):
#     global maxlen,maxtokens,mean
#     res = topic_token.encode(txt)
#     res_len = len(res)
    
#     mean.append(res_len)
#     if res_len > maxlen:
#         maxlen = res_len
#         maxtokens = res

# imdbloader.throughDataBeforeRead(findMaxMinArticle,1000)
# print(maxlen)
# print(maxtokens)
# print(sum(mean)/len(mean))

# plt.plot(mean,[i for i in range(len(mean))])
# plt.show()