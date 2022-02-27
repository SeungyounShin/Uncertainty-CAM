import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss
import torch.nn.functional as F


class CrossEntropyLoss(_Loss):
    def __init__(self, weight=None, gamma=1., temp=1., reduction='mean', eps=1e-6):
        super(_Loss, self).__init__()
        self.weight = weight
        self.gamma = gamma
        self.temp = temp
        self.reduction = reduction
        self.eps = eps

    def forward(self, preds, labels):
        preds = preds / self.temp
        if self.gamma >= 1.:
            loss = F.cross_entropy(
                preds, labels, weight=self.weight, reduction=self.reduction)
        else:
            log_prob = preds - torch.logsumexp(preds, dim=1, keepdim=True)
            log_prob = log_prob * self.gamma
            loss = F.nll_loss(
                log_prob, labels, weight=self.weight, reduction=self.reduction)

        losses = {'loss': loss}
        return losses


class FocalLoss(_Loss):
    def __init__(self, weight=None, alpha=1., gamma=1., reduction='mean'):
        super(_Loss, self).__init__()
        self.weight = weight
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, preds, labels):
        log_prob = F.log_softmax(preds, dim=-1)
        prob = torch.exp(log_prob)
        loss = F.nll_loss(
            (self.alpha * (1 - prob) ** self.gamma) * log_prob, labels,
            weight=self.weight, reduction = self.reduction)
        losses = {'loss': loss}
        return losses


class CustomCriterion(_Loss):
    def __init__(self):
        super(_Loss, self).__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, output_dict):
        preds = output_dict['preds']
        labels = output_dict['labels']

        losses = {}
        loss = self.criterion(input=preds, target=labels)
        losses['loss'] = loss

        return losses

class MaceCriterion(_Loss):
    def __init__(self, num_classes, device, is_multilabel, epsilon=1e-6):
        super(_Loss, self).__init__()
        self.one_hot = torch.eye(num_classes).to(device)
        self.epsilon = epsilon
        self.is_multilabel = is_multilabel
        if self.is_multilabel:
            self.softmargin = torch.nn.MultiLabelSoftMarginLoss()

    def forward(self, output_dict):
        pi, mu_hat, sigma = output_dict['pi'], output_dict['mu'], output_dict['sigma']
        labels = output_dict['labels']
        if self.is_multilabel:
            targets = labels
        else:
            targets = self.one_hot[labels]

        #mu_hat = torch.softmax(mu, dim=2) # logit to prob [N x K x D]
        log_mu_hat = torch.log(mu_hat + self.epsilon) # [N x K x D]

        # pi
        pi_usq = torch.unsqueeze(pi, 2) # [N x K x 1]
        pi_exp = pi_usq.expand_as(mu_hat) # [N x K x D]

        ### self distributed labeling
        # target
        target_usq =  torch.unsqueeze(targets, 1) # [N x 1 x D]
        if target_usq.dim() == 4:
            target_usq = target_usq.squeeze(1)
        target_exp =  target_usq.expand_as(mu_hat) # [N x K x D]

        #if self.is_multilabel:
        if False:
            # using softmargin
            temp = 0.6
            term1 = target_exp*torch.log(torch.pow(1+torch.exp(-mu_hat/temp), -1))
            term2 = (1-target_exp)*(torch.exp(-mu_hat/temp)/(1+torch.exp(-mu_hat/temp)))
            cls_loss = - term1 - term2

        else:
            # CE loss
            cls_loss = -target_exp * log_mu_hat - (1-target_exp)*torch.log(1-mu_hat + self.epsilon) # CE [N x K x D]

        ace_exp = cls_loss #/sigma # attenuated CE [N x K x D]
        mace_exp = torch.mul(pi_exp, ace_exp) # mixtured attenuated CE [N x K x D]
        mace = torch.sum(mace_exp, dim=1)     # [N x D]
        mace = torch.sum(mace, dim=1)         # [N]
        mace_avg = torch.mean(mace)           # [1]

        # Compute uncertainties (epis and alea)
        unct_out = mln_uncertainties(pi, mu_hat, sigma)
        epis = unct_out['epis'] # [N]
        alea = unct_out['alea'] # [N]
        epis_avg = torch.mean(epis) # [1]
        #alea_avg = torch.mean(torch.log(alea)) # [1]
        alea_avg = torch.mean(alea)
        loss = mace_avg - epis_avg + alea_avg

        # Return
        losses = {
            'loss': loss,
            'mace':mace, # [N]
            'mace_avg':mace_avg, # [1]
            'epis':epis, # [N]
            'alea':alea, # [N]
            'epis_avg':epis_avg, # [1]
            'alea_avg':alea_avg, # [1]
        }
        return losses


def mln_uncertainties(pi, mu, sigma, alea_type='entropy'):
    """
        :param pi:      [N x K]
        :param mu:      [N x K x D]
        :param sigma:   [N x K x D]
    """
    # pi
    k = pi.size(-1)
    #mu_hat = torch.softmax(mu, dim=2) # logit to prob [N x K x D]
    pi_usq = torch.unsqueeze(pi, 2) # [N x K x 1]
    pi_exp = pi_usq.expand_as(sigma) # [N x K x D]

    # softmax(mu) average
    mu_hat_avg = torch.sum(torch.mul(pi_exp, mu), dim=1).unsqueeze(1) # [N x 1 x D]
    mu_hat_avg_exp = mu_hat_avg.expand_as(mu) # [N x K x D]
    mu_hat_diff_sq = torch.square(mu - mu_hat_avg_exp) # [N x K x D]

    # Epistemic uncertainty
    epis = torch.sum(torch.mul(pi_exp, mu_hat_diff_sq), dim=1)  # [N x D]
    epis = torch.sqrt(torch.sum(epis, dim=1) + 1e-4) # [N]

    # Aleatoric uncertainty
    if alea_type=='pi_entropy':
        alea = torch.nn.functional.cross_entropy(pi ,torch.argmax(pi,dim=1),reduce=False)
    elif alea_type=='pi_diff':
        mat = torch.topk(pi,k=2,dim=1)[0]
        alea = 1 - (mat[:,0] -mat[:,1])
    elif alea_type=='entropy':
        largest_pi_idx = torch.argmax(pi,dim=1) # [N]
        pad_ind = largest_pi_idx.unsqueeze(-1).repeat(1,1,mu.shape[-1]).permute(1,0,2)
        mu_gather = mu.gather(1, pad_ind).squeeze() # [N x 1 x D]
        if(mu_gather.dim()==1):
            mu_gather = mu_gather.unsqueeze(0)
        imp_label = torch.max(mu_gather, dim=-1)[-1]
        alea = torch.nn.functional.cross_entropy(mu_gather , imp_label ,reduce=False)
    elif alea_type == "mixture_entropy":
        pi_usq = torch.unsqueeze(pi, 2) # [N x K x 1]
        pi_exp = pi_usq.expand_as(mu) # [N x K x D]
        # mu [N x K x D]
        log_entropy = -mu*torch.log(mu + 1e-6) # [N x K x D]
        mace_exp = torch.mul(pi_exp, log_entropy)
        mace = torch.sum(mace_exp, dim=1)     # [N x D]
        mace = torch.sum(mace, dim=1)         # [N]
        alea = torch.mean(mace)               # [1]
    else:
        alea = torch.sum(torch.mul(pi_exp, sigma), dim=1)  # [N x D]
        alea = torch.sqrt(torch.mean(alea,dim=1) + 1e-4) # [N]
        #print(alea)

    # Return
    unct_out = {
        'epis':epis, # [N]
        'alea':alea  # [N]
    }
    return unct_out
