import platform

envir_system = platform.system()
print("[npc report] your system is ",envir_system)

batch_size = 10
read_data_num = 100

if envir_system == "Windows":
    envir_path = "."
else:
    envir_path = "/content/drive/MyDrive/FL-ABAS-TOPIC/"
    batch_size = 1000
    read_data_num = -1



import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")



vocLen = 10000

TopicModel_train_dataset = ""
