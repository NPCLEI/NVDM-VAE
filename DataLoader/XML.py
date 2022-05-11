from cProfile import label
import xml.etree.ElementTree as ET
from torch.utils.data import Dataset, DataLoader

class XmlDataset(Dataset):
    def __init__(self,xml_path):
        super(XmlDataset, self).__init__()
        self.xml_path = xml_path
        self.parser = ET.parse(xml_path)
        self.root = self.parser.getroot()
        self.len = len(self.root)
    
    def __getitem__(self,index):
        try:
            senitem = self.root[index]
            sentence = senitem[0].text
            # print(sentence,len(senitem))
            aspect_terms = senitem[1]
            res = [sentence]
            label = aspect_terms[0].attrib["polarity"]
            for aspect_term in aspect_terms:
                if aspect_term.attrib["polarity"] != label:
                    break
                res.append(aspect_term.attrib["term"])
            return "[sep]".join(res),label
        except:
            # print(sentence,senitem.attrib["id"])
            return -1,-1

    def __len__(self):
        return self.len

if __name__ == "__main__":
    d = XmlDataset(r"D:\Dataset\NLP-SA-Apect-Level\laptops-trial.xml")
    print(len(d))
    for i in range(d.len):
        if d[i] != (-1,-1):
            print(d[i])

######################教程#######################
# <?xml version="1.0" encoding="utf-8"?>
# <list>
# <student id="stu1" name="stu">
#    <id>1001</id>
#    <name>张三</name>
#    <age>22</age>
#    <gender>男</gender>
# </student>
# <student id="stu2" name="stu">
#    <id>1002</id>
#    <name>李四</name>
#    <age>21</age>
#    <gender>女</gender>
# </student>
# </list>

'''
遇到问题没人解答？小编创建了一个Python学习交流QQ群：531509025
寻找有志同道合的小伙伴，互帮互助,群里还有不错的视频学习教程和PDF电子书！
'''
# import xml.etree.ElementTree as ET
 
# tree = ET.parse("test.xml")
# # 根节点
# root = tree.getroot()
# # 标签名
# print('root_tag:',root.tag)
# for stu in root:
#     # 属性值
#     print ("stu_name:", stu.attrib["name"])
#     # 标签中内容
#     print ("id:", stu[0].text)
#     print ("name:", stu[1].text)
#     print("age:", stu[2].text)
#     print("gender:", stu[3].text)
