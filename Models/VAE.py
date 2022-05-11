import torch
from torch import layer_norm, nn
from torch.nn import functional as F
import Config

class MLP(nn.Module):
    
    def CreateMLP(dims=[200,20,2]):
        res = nn.ModuleList()
        lastOutput = dims[0]
        for dim in dims[1:]:
            res.append(torch.nn.Linear(lastOutput, dim))
            lastOutput = dim
        return res

class VAE(nn.Module):
    def __init__(self,
        inputdim = Config.vocLen, 
        topic_num = 3,
        encode_hidden_dims=[8000,3000,1000,1000],
        decode_hidden_dims=[],
        dtype = None):
        """
            decode_hidden_dims = []:解码器和编码器反对称
        """
        super(VAE, self).__init__()

        self.inputdim = inputdim
        self.deivce = None
        self.dtype = dtype
        self.decoder_output_acf = torch.sigmoid
        self.recon_loss_fc = F.mse_loss
        encode_dims = [inputdim] + encode_hidden_dims
        if len(decode_hidden_dims) == 0:
            decode_dims = [topic_num] + encode_dims[::-1]
        elif dtype == "NVMD":
            self.decoder_output_acf = VAE.__softmax__
            self.recon_loss_fc = F.binary_cross_entropy
            decode_dims = [topic_num,inputdim]

        self.encoder_mlp = MLP.CreateMLP(encode_dims)
        self.encoder_mu_logvar = torch.nn.Linear(encode_hidden_dims[-1], topic_num * 2)
        # self.encoder_mu = torch.nn.Linear(encode_hidden_dims[-1], topic_num)
        # self.encoder_logvar = torch.nn.Linear(encode_hidden_dims[-1], topic_num)

        self.decoder_mlp = MLP.CreateMLP(decode_dims)

    def __softmax__(x):
        return torch.softmax(x,dim = 1)

    def toInference(net,x):
        mu_logvar = net.encoder(x)
        # print(mu_logvar)
        try:
            mu,logvar = mu_logvar.chunk(2,dim=1)
        except:
            mu,logvar = mu_logvar[0].chunk(2,dim=1)

        z = net.reparameterise(mu, logvar)
        z = z.exp() / z.exp().sum()

        return z

    def inference(self,x):
        mu_logvar = self.encoder(x)
        mu,logvar = mu_logvar.chunk(2,dim=0)
        z = self.reparameterise(mu, logvar)
        z = F.softmax(z)
        return z

    def encoder(self,x):

        e = F.relu(self.encoder_mlp[0](x))
        for ln in self.encoder_mlp[1:]:
            e = F.relu(ln(e))

        return self.encoder_mu_logvar(e)

    def decoder(self,rez):

        e = F.relu(self.decoder_mlp[0](rez))
        for ln in self.decoder_mlp[1:-1]:
            e = F.relu(ln(e))
        e = self.decoder_output_acf(self.decoder_mlp[-1](e))
        return e

    def reparameterise(self, mu, logvar):
        epsilon = torch.randn_like(mu)
        # print(mu.shape,logvar.shape)
        return mu + epsilon * torch.exp(logvar / 2)

    def loss(x,recon_x, mu, logvar,recon_loss_fc = F.mse_loss,device = "cpu"):
        if device != "cpu":
            device = torch.device("cuda:0" if not torch.cuda.is_available() else "cpu")
        else:
            device = torch.device("cpu")

        kl_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - torch.exp(logvar)))
        # print(recon_x.size(),x.size())
        recon_loss = recon_loss_fc(recon_x,x)
        return ( recon_loss + kl_loss ) / Config.batch_size


    def forward(self, x):

        mu_logvar = self.encoder(x)
        # print(mu_logvar.shape)
        mu,logvar = mu_logvar.chunk(2,dim=1)
        # print(mu)
        # print(logvar)
        z = self.reparameterise(mu, logvar)
        recon_x = self.decoder(z)

        return recon_x, mu, logvar

    def save(self,info = ""):
        import pickle
        f_name = "%s/ModelPickle/topic_vae_net%s.pickle"%(Config.envir_path,info)
        if self.dtype == "NVMD":
            f_name = "%s/ModelPickle/topic_nvmd_net%s.pickle"%(Config.envir_path,info)

        self.to(torch.device("cpu"))
        with open(f_name, 'wb+') as net_file:
            pickle.dump(self,net_file)
        self.to(self.device)
