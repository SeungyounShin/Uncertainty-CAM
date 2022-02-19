import torchvision
import torch
import numpy as np
from torchvision import transforms
import os,sys
sys.path.append('/data/private/Uncertainty-CAM/')
from src.core.models import resnet
from tqdm import tqdm
import torch.nn.functional as F
import torch.nn as nn

print("import")

trans = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),]
)

imagenet_train  = torchvision.datasets.ImageNet('/data/opensets/imagenet-pytorch', 
                                                split='train',
                                                transform = trans)


print("imagenet load")
data_loader = torch.utils.data.DataLoader(imagenet_train,
                                          batch_size=64,
                                          shuffle=True,
                                          num_workers=0)

model = resnet.resnet50(pretrained=True)
model = model.cuda()
print("model load")

label_saver = list()
feat_saver = list()

for imgs,labels in tqdm(data_loader):
    feat = model._forward_impl(imgs.cuda())
    feat = F.avg_pool2d(feat, 4).squeeze()
    
    label_saver += list(labels.cpu().numpy())
    feat_saver += list(feat.detach().cpu().numpy())

feat_torch = torch.tensor(feat_saver)
label_torch = torch.tensor(label_saver)
save_dict = {'feat' : feat_torch, 'label_torch':label_torch}

o = torch.save(save_dict, '/data/private/Uncertainty-CAM/ckpt_feat/feats.pt')