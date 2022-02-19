import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MixtureOfLogits(nn.Module):
    def __init__(self,
                 in_dim     = 64,   # input feature dimension
                 y_dim      = 10,   # number of classes
                 k          = 5,    # number of mixtures
                 mu_min     = -3,   # minimum mu (init)
                 mu_max     = 3,    # maximum mu (init)
                 sig_min    = 1e-4, # minimum sigma
                 sig_max    = None, # maximum sigma
                 share_sig  = True, # share sigma among mixture
                 multilabel = False,
                 ):
        super(MixtureOfLogits,self).__init__()
        self.in_dim     = in_dim    # Q
        self.y_dim      = y_dim     # D
        self.k          = k         # K
        self.mu_min     = mu_min
        self.mu_max     = mu_max
        self.sig_min    = sig_min
        self.sig_max    = sig_max
        self.share_sig  = share_sig
        self.multilabel = multilabel
        self.feature_selcetion = False
        self.build_graph()

    def build_graph(self):
        self.fc_pi      = nn.Linear(self.in_dim,self.k)
        self.fc_mu      = nn.Linear(self.in_dim,self.k*self.y_dim)
        if self.feature_selcetion:
            self.FSL = nn.Linear(self.in_dim, self.in_dim*self.k)
            self.mu  = nn.ModuleList([nn.Linear(self.in_dim, self.y_dim) for i in range(self.k)])
        if self.share_sig:
            self.fc_sigma   = nn.Linear(self.in_dim,self.k)
        else:
            self.fc_sigma   = nn.Linear(self.in_dim,self.k*self.y_dim)

    def forward(self,x):
        """
            :param x: [N x Q]
        """
        pi_logit        = self.fc_pi(x)                                 # [N x K]
        pi              = torch.softmax(pi_logit,dim=1)                 # [N x K]
        if self.feature_selcetion:
            sel_logit = self.FSL(x).view(-1,self.k,self.in_dim)
            sel = F.softmax(sel_logit/0.1, dim=1)                       # [N X K x Q]
            mu_ = list()
            for i in range(self.k):
                mu_.append(self.mu[i](sel[:,i]))
            mu = torch.stack(mu_, dim=1)

        else:
            mu              = self.fc_mu(x)                                 # [N x KD]
            mu              = torch.reshape(mu,(-1,self.k,self.y_dim))      # [N x K x D]
            #if not self.multilabel:
            #mu         = F.sigmoid(mu)
        mu          = F.softmax(mu, dim=-1)

        if self.share_sig:
            sigma       = self.fc_sigma(x)                              # [N x K]
            sigma       = sigma.unsqueeze(dim=-1)                       # [N x K x 1]
            sigma       = sigma.expand_as(mu)                           # [N x K x D]
        else:
            sigma       = self.fc_sigma(x)                              # [N x KD]
        sigma           = torch.reshape(sigma,(-1,self.k,self.y_dim))   # [N x K x D]

        if self.sig_max is None:
            sigma = self.sig_min + torch.exp(sigma)                     # [N x K x D]
        else:
            sig_range = (self.sig_max-self.sig_min)
            sigma = self.sig_min + sig_range*torch.sigmoid(sigma)       # [N x K x D]
            #sigma = torch.clamp(sigma, self.sig_min, self.sig_max)
        mol_out = {'pi':pi,'mu':mu,'sigma':sigma}
        return mol_out

    def init_param(self):
        for m in self.modules():
            if isinstance(m,nn.Conv2d): # init conv
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)
            if isinstance(m,nn.Linear): # lnit dense
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)
        # Heuristic: fc_mu.bias ~ Uniform(mu_min,mu_max)
        self.fc_mu.bias.data.uniform_(self.mu_min,self.mu_max)
        #self.fc_sigma.bias.data.uniform_(self.sig_min,self.sig_max)


class MixtureLogitNetwork(nn.Module):
    def __init__(self,
                 name       = 'mln',        # name
                 h_dim      = 512,          # itermediate feature dimension
                 y_dim      = 10,           # output dimension
                 use_bn     = True,         # whether to use batch-norm
                 k          = 5,            # number of mixtures
                 sig_min    = 1e-4,         # minimum sigma
                 sig_max    = 10,           # maximum sigma
                 mu_min     = -3,           # minimum mu (init)
                 mu_max     = +3,           # maximum mu (init)
                 share_sig  = True,
                 backbone   = None,
                 multilabel = False,
                 feature_extractor = None,
                 ):
        super(MixtureLogitNetwork,self).__init__()
        self.name       = name
        self.h_dim      = h_dim
        self.y_dim      = y_dim
        self.use_bn     = use_bn
        self.k          = k
        self.sig_min    = sig_min
        self.sig_max    = sig_max
        self.mu_min     = mu_min
        self.mu_max     = mu_max
        self.share_sig  = share_sig
        self.backbone   = backbone
        self.feature_extractor = feature_extractor
        #pretrain freeze
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.multilabel = multilabel
        self.build_graph()

    def build_graph(self):
        # Final mixture of logits layer
        self.mol = MixtureOfLogits(
            in_dim      = self.h_dim,
            y_dim       = self.y_dim,
            k           = self.k,
            mu_min      = self.mu_min,
            mu_max      = self.mu_max,
            sig_min     = self.sig_min,
            sig_max     = self.sig_max,
            share_sig   = self.share_sig,
            multilabel  = self.multilabel
        )
        self.mol.init_param()

    def forward(self,x):
        if self.backbone is not None:
            if self.feature_extractor is not None:
                #inputs = self.feature_extractor(images=x.cpu(), return_tensors="pt")
                #print(inputs)
                outputs = self.backbone(x)
                feat = outputs.pooler_output
            else:
                feat = self.backbone(x)
                if (feat.shape[-1] > 1):
                    feat = F.avg_pool2d(feat, 4)
                feat = feat.view(feat.size(0),-1)
        else:
            feat = x
        mln_out = self.mol(feat)
        return mln_out # mu:[N x K x D] / pi:[N x K] / sigma:[N x K x D]
