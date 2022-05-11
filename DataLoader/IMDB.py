from cmath import e
import os
import torch
from torch.utils.data import Dataset, DataLoader
from Models import WordEmbedding

class Loader(Dataset):
    def __init__(self,d_path,num,cls_folder = ["neg","pos","unsup"],bindLabel = False):
        super(Loader, self).__init__()
        """
            cls_folder:类别文件夹,d_path下对应的与类别同名的文件夹
        """
        self.path = d_path
        self.cls_folder = cls_folder if len(cls_folder) != 0 else os.listdir(d_path)

        self.data = []
        self.labels = []

        self.len = 0
        self.loadNum = num
        self.table = []
        self.__prepare__()

        self.getItemToken = True


    def getToken(self,flag = True):
        self.getItemToken = flag

    def setTokenizer(self,tokenizer,one_hot_len):
        self.tokenizer = tokenizer
        self.one_hot_len = one_hot_len

    def getIndexTxt(self,index):
        with open(self.table[index],"r") as txt:
            txtc = txt.read()
            return  txtc,WordEmbedding.one_hot(txtc,self.tokenizer,self.one_hot_len)

    def __getitem__(self,index):
        # print(self.table[index])
        try:
            with open(self.table[index],"r") as txt:
                txt_content = txt.read()
                if self.getItemToken:
                    return WordEmbedding.one_hot(txt_content,self.tokenizer,self.one_hot_len),self.labels[index]
                else:
                    return txt_content,len(txt_content)
        except Exception as e:
            print("[npc report eorr]",e,self.one_hot_len,self.table[index],"auto handle:convert -> zero")
            return torch.zeros((1,self.one_hot_len)),0

    def __len__(self):
        return len(self.table)

    def __prepare__(self):
        read_list = self.__prepare_read_list__(self.loadNum)
        
        #for cls folder
        for cls_i in range(len(read_list)):
            tpath,rnum = read_list[cls_i]
            #for txts
            l,r = rnum
            file_list = os.listdir(tpath)
            r = r if r > 0 else len(file_list)
            
            for f in file_list[l:r]:
                self.table.append("%s/%s"%(tpath,f))
                self.labels.append(cls_i)

    def __prepare_read_list__(self,num):
        read_list = []
        if type(num) == int:
            read_list = [["%s/%s"%(self.path,cls),(0,num)] for cls in self.cls_folder]
        elif type(num) == list:
            read_list = num
        return read_list

    def through(self,num,func):
        read_list = self.__prepare_read_list__(num)

        for cls_i in range(len(read_list)):
            tpath,rnum = read_list[cls_i]
            #for txts
            l,r = rnum
            r = r if r > 0 else len(file_list)
            file_list = os.listdir(tpath)
            for f in file_list[l:r]:
                func(f,tpath,rnum)

    def listDataPath(self):
        read_list = self.__prepare_read_list__(self.loadNum)
        
        res = []
        #for cls folder
        for cls_i in range(len(read_list)):
            tpath,rnum = read_list[cls_i]
            #for txts
            l,r = rnum
            file_list = os.listdir(tpath)
            r = r if r > 0 else len(file_list)
            
            for f in file_list[l:r]:
                res.append("%s/%s"%(tpath,f))

        return res

    def throughDataBeforeRead(self,func,num=-1,read=True):
        
        read_list = []
        if type(num) == int:
            read_list = [["%s/%s"%(self.path,cls),(0,num)] for cls in self.cls_folder]
        elif type(num) == list:
            read_list = num
        

        #for cls folder
        for cls_i in range(len(read_list)):
            tpath,rnum = read_list[cls_i]
            #for txts
            l,r = rnum
            file_list = os.listdir(tpath)
            r = r if r > 0 else len(file_list)
            for txt_i in range(l,r):
                try:
                    if read:
                        with open("%s/%s"%(tpath,file_list[txt_i]),"r") as txt:
                            func(txt.read(),cls_i)
                    else:
                        func("%s/%s"%(tpath,file_list[txt_i]))
                except:
                    pass

        return self

    def read(self,token = None):
        read_list = self.__prepare_read_list__()

        #for cls folder
        expect_num = 0
        for cls_i in range(len(read_list)):
            tpath,rnum = read_list[cls_i]
            #for txts
            l,r = rnum
            file_list = os.listdir(tpath)
            r = r if r > 0 else len(file_list)

            for txt_i in range(l,r):
                try:
                    with open("%s/%s"%(tpath,file_list[txt_i]),"r") as txt:
                        self.data.append(txt.read())
                        self.labels.append(cls_i)
                except:
                    expect_num += 1

        return self

if __name__ == "__main__":
    ld = Loader(r'E:\Dataset\IMDB\train/')
    # ld.read(10)
    # print(ld.listDataPath(10))
