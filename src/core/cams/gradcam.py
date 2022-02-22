#
# Copyright (c) 2019 Idiap Research Institute, http://www.idiap.ch/
# Written by Suraj Srinivas <suraj.srinivas@idiap.ch>
#

"""
    Implement GradCAM

    Original Paper:
    Selvaraju, Ramprasaath R., et al. "Grad-cam: Visual explanations from deep networks
    via gradient-based localization." ICCV 2017.

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from math import isclose
from src.utils.util import mln_gather,DataParallel
from src.core.criterions import *

class GradCAMExtractor:
    #Extract tensors needed for Gradcam using hooks

    def __init__(self, model, criterion, loss_type='epis_avg', device='cuda'):
        if type(model)==DataParallel:
            self.model = model.module
        else:
            self.model = model
        self.criterion = criterion
        self.loss_type = loss_type
        self.mace_criterion = MaceCriterion(num_classes=1000, device=device,is_multilabel=False)
        self.device = device

        self.features = None
        self.feat_grad = None

        self.target_module = None

        self.feature_handles = []
        self.grad_handles = []
        prev_module = None

        # TODO: set the layer to extract a gradcam
        for name, module in self.model.named_modules():
            if name == 'mol':
                self.target_module = prev_module
                break
            if isinstance(module, nn.Conv2d):
                prev_module = module

    def _extract_layer_grads(self, module, in_grad, out_grad):
        # function to collect the gradient outputs
        self.feature_grads = out_grad[0]

    def _extract_layer_features(self, module, input, output):
        # function to collect the layer outputs
        self.features = output

    def _register_hooks(self):
        if self.target_module is not None:
            # Register feature-gradient and feature hooks for each layer
            self.handle_g = self.target_module.register_backward_hook(self._extract_layer_grads)
            self.handle_f = self.target_module.register_forward_hook(self._extract_layer_features)

    def _remove_hooks(self):

        self.handle_g.remove()
        self.handle_f.remove()

    def getFeaturesAndGrads(self, x, target_class, mixture=None):
        self.model.zero_grad()

        x.requires_grad = True
        output_dict = self.model(x)

        if target_class is None:
            mu_sel = mln_gather(output_dict)['mu_sel']
            target_class = mu_sel.data.max(1, keepdim=True)[1]
        output_dict['labels'] = target_class

        if(self.loss_type in ['alea_avg','epis_avg']):
            output_scalar = self.mace_criterion(output_dict)[self.loss_type]
            output_scalar = output_scalar
            #output_scalar = -1. * F.nll_loss(out, target_class.flatten(), reduction='sum')
        elif(self.loss_type=="second"):
            bs = output_dict['mu'].size(0)
            largest_pi = torch.argmax(output_dict['pi']).detach().cpu().numpy()
            secondind = torch.topk(output_dict['mu'][0,largest_pi], k=2)[-1][-1].detach().cpu()
            logit_sel = output_dict['mu'][0,largest_pi]# [N x D]
            one_hot = torch.eye(1000)[secondind] # CUB
            #one_hot = output_dict['labels'] # VOC
            output_scalar = torch.sum(one_hot.to(self.device) * logit_sel.to(self.device))

        else:
            bs = output_dict['mu'].size(0)
            num_classes = output_dict['mu'].size(-1)
            mixture_num = output_dict['pi'].size(-1)
            _, largest_pi_idx = torch.max(output_dict['pi'], dim=1)

            if mixture is None:
                #mu_hat = torch.softmax(output_dict['mu'], dim=2) # logit to prob [N x K x D]
                #logit_sel = output_dict['mu'][[i for i in range(bs)], largest_pi_idx] # [N x D]
                mu = output_dict['mu']          # [N x K x D]
                pi = output_dict['pi']          # [N x K]
                pi_usq = torch.unsqueeze(pi, 2) # [N x K x 1]
                pi_exp = pi_usq.expand_as(mu)   # [N x K x D]
                logit_sel = torch.sum(pi_exp * mu, dim=1)

            else:
                logit_sel = output_dict['mu'][[i for i in range(bs)], mixture] # [N x D]

            one_hot = torch.eye(num_classes)[output_dict['labels']] # CUB
            #one_hot = output_dict['labels'] # VOC

            output_scalar = torch.sum(one_hot.to(self.device) * logit_sel.to(self.device))

        # Compute gradients
        output_scalar.backward()

        return self.features, self.feature_grads


class GradCAM():
    """
    Compute GradCAM
    """

    def __init__(self, model, criterion, loss_type='alea_avg', device='cuda'):
        self.model = model
        self.model_ext = GradCAMExtractor(self.model, criterion, loss_type, device)
        self.loss_type = loss_type

    def register_hooks(self):
        self.model_ext._register_hooks()

    def remove_hooks(self):
        self.model_ext._remove_hooks()

    def localize(self, image, target_class=None, mixture=None, normalize=False):
        #Simple FullGrad saliency
        self.model.eval()
        features, intermed_grad = self.model_ext.getFeaturesAndGrads(image, target_class=target_class, mixture=mixture)

        # GradCAM computation
        if 'ViT' in type(self.model.backbone).__name__:
            # feature       [1, 768, 14, 14]
            # intermed_grad [1, 768, 14, 14]
            grads = intermed_grad.mean(dim=(2,3), keepdim=True)
            cam = (features* grads).sum(1, keepdim=True)
            #cam = (F.relu(features)* grads).sum(1, keepdim=True)
            #normalize
            if normalize:
                cam -= cam.min()
                cam /= cam.max()
            cam_resized = F.interpolate(F.relu(cam), size=image.size(2), mode='bilinear', align_corners=True)

        else:

            grads = intermed_grad.mean(dim=(2,3), keepdim=True)
            cam = (F.relu(features)* grads).sum(1, keepdim=True)
            cam_resized = F.interpolate(F.relu(cam), size=image.size(2), mode='bilinear', align_corners=True)
        return cam_resized
