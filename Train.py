import utils
from DataLoader.IMDB import Loader as IMDBLoader
from Models import TopicModel
from Models import WordEmbedding
# import matplotlib.pyplot as plt
import sys
sys.path.append("./")
import Config
import Statistics


#第零步，把数据读进内存

#第一步，训练主题模型的token
if Config.envir_system == "Windows":
    imdbdataset = IMDBLoader(r'E:\Dataset\IMDB\train',Config.read_data_num)
else:
    imdbdataset = IMDBLoader('/content/aclImdb/train',Config.read_data_num)
topic_tokenizer = utils.CheckModel(
    'topic_token',
    lambda nl:WordEmbedding.Train(imdbdataset.listDataPath())
)

#第二步，训练主题模型的VAE
# print(WordEmbedding.one_hot("Do not panic.",topic_token,10000))
batch_size = 10
Config.vocLen = 10000
imdbdataset.setTokenizer(topic_tokenizer,Config.vocLen)

topic_nvmd_net = utils.CheckModel(
    "full_topic_nvmd_net",
    lambda ml: TopicModel.TrainNVMD(imdbdataset,ml),
    continue_train = False,
    retrain=False
)

#第三步，形成主题矩阵

topic_w = utils.CheckModel(
    "topic_w",
    lambda ml: TopicModel.Gen_Topic_W(
            topic_nvmd_net,
            dataset=imdbdataset,
            tokenizer=topic_tokenizer,
            printRes=False
        ),
    retrain=False
)

Statistics.topic_model_cls_word_num(topic_w)

Statistics.get_related_word("role",topic_w)

# train_topic_w_article = utils.CheckModel(
#     "train_topic_w_article",
#     lambda ml: TopicModel.Gen_Topic_W_article(topic_nvmd_net,dataset=imdbdataset,tokenizer=topic_tokenizer),
#     continue_train = False,
#     retrain=True
# )

# Statistics.topic_model_cls_article_num(train_topic_w_article)

# topic_w_article = 
# print(topic_w_article)


# txts = {"cls1":[],"cls2":[]}

# for ix in range(len(imdbdataset)):
#     txt,emb = imdbdataset.getIndexTxt(ix)
#     emb = topic_nvmd_net.inference(emb)
#     print(ix,emb)
#     if emb.tolist()[0] > emb.tolist()[1]:
#         txts["cls1"].append(txt)
#     else:
#         txts["cls2"].append(txt)

# print("###########cls1###########")
# print(len(txts["cls1"]))

# print("###########cls2###########")
# print(len(txts["cls2"]))