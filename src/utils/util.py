import torch
import getpass
import logging
import random, os, cv2
import yaml
import numpy as np
from PIL import Image
from colorlog import ColoredFormatter
from tqdm import tqdm

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from scipy.ndimage.filters import gaussian_filter
from scipy import io

path = os.getcwd()
if path.split('/')[-1] == 'notebooks':
    os.chdir('..')
from src.core.criterions import *
from notebooks.label_names import *


loss_types = ['mace_avg','alea_avg','epis_avg']
color_dict = {'head' : 'lightcyan' , 'lear':'red', 'rear':'red', 'leye':'blue','reye':'blue',
               'muzzle' : 'brown', 'torso':'purple','lblleg':'aqua','lbuleg':'aqua', 'nose' : 'darkred',
             'lfleg' :'aqua' , 'rfleg' : 'aqua', 'lfpa' : 'aqua','rfpa' : 'aqua', 'lbleg':'aqua',
             'lbpa':'navy','rbleg':'aqua','rbpa':'navy','tail' : 'darkslateblue','lflleg':'aqua','lfuleg':'aqua',
             'rblleg':'aqua','lblleg':'aqua','rbuleg':'aqua', 'screen' : 'aqua','frontside':'green',
             'rightside':'navy','door_1':'aqua','rightmirror':'brown','headlight_1':'red',
             'wheel_1':'silver','wheel_2':'silver','window_1':'lightcyan','window_2':'lightcyan','window_3':'lightcyan',
             'lebrow' : 'dimgray','rebrow':'dimgray','mouth':'lightcoral','luarm':'aqua','ruarm':'aqua',
             'body':'purple', 'cap' : 'tan','hair':'black','neck' : 'bisque', 'beak':'red','lleg':'aqua','rleg':'aqua'
             ,'rfoot':'navy','lfoot':'navy','llarm':'aqua','rlarm':'aqua','lhand':'navy','rhand':'navy',
             'llleg':'aqua','rlleg':'aqua','luleg':'navy','ruleg':'navy','rflleg':"aqua",'lflleg':"aqua",
             'rfuleg':'aqua','lfuleg':'aqua','pot' : 'yellow','plant':'green','bliplate':'dimgray','backside':'red',
             'lwing':'orange','rwing':'orange','boat':'purple','fwheel':'aqua','bwheel':'aqua','chainwheel':'silver',
             'handlebar':'navy','saddle':'red','lfho':'navy','rfho':'navy','lbho':'navy','rbho':'navy',
             'sofa':'purple','leftside':'aqua','fliplate':'purple','headlight_2':'red','headlight_3':'red',
             'headlight_4':'red','headlight_5':'red','headlight_6':'red','headlight_7':'red','motorbike':'purple',
             'coach_1':'orange','crightside_1':'purple','croofside_1':'gold','hfrontside':'green','hrightside':'blue',
             'hroofside':'lightcyan','bird':'purple','leftmirror':'brown','roofside':'gold'}

# Logging
# =======

def load_log(name):
    def _infov(self, msg, *args, **kwargs):
        self.log(logging.INFO + 1, msg, *args, **kwargs)

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    formatter = ColoredFormatter(
        "%(log_color)s[%(asctime)s - %(name)s] %(message)s",
        datefmt=None,
        reset=True,
        log_colors={
            'DEBUG':    'cyan',
            'INFO':     'white,bold',
            'INFOV':    'cyan,bold',
            'WARNING':  'yellow',
            'ERROR':    'red,bold',
            'CRITICAL': 'red,bg_white',
        },
        secondary_log_colors={},
        style='%'
    )
    ch.setFormatter(formatter)

    log = logging.getLogger(name)
    log.setLevel(logging.DEBUG)
    log.handlers = []       # No duplicated handlers
    log.propagate = False   # workaround for duplicated logs in ipython
    log.addHandler(ch)

    logging.addLevelName(logging.INFO + 1, 'INFOV')
    logging.Logger.infov = _infov
    return log


# General utils
# =============

def load_config(config_path):
    with open(config_path) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    return config

def cutout_func(img1, img2, bbox1, bbox2):
    patch = img2[bbox2[0]:bbox2[2], bbox2[1]:bbox2[3]]
    patch = cv2.resize(patch, dsize=(bbox1[3]-bbox1[1], bbox1[2]-bbox1[0]), interpolation=cv2.INTER_CUBIC)
    img1[bbox1[0]:bbox1[2], bbox1[1]:bbox1[3]] = patch

    #plt.subplot(2,2,1)
    #plt.imshow(img1[bbox1[0]:bbox1[2], bbox1[1]:bbox1[3]])
    #plt.subplot(2,2,2)
    #plt.imshow(patch)

    return img1

def blur_func(img,bbox,sigma=4):

    patch = img[bbox[0]:bbox[2],bbox[1]:bbox[3]]
    blurred = gaussian_filter(patch, sigma=sigma)

    img[bbox[0]:bbox[2],bbox[1]:bbox[3]] = blurred

    return img

def imagenet_sample(path):
    random_class = random.choice(os.listdir(path))
    sample_path = random.choice(os.listdir(path+'/'+random_class))
    #full_path = '/sdb/ImageNet1K/train/n02092002/n02092002_9633.JPEG'

    sample_img = Image.open(path + "/" + random_class + "/" + sample_path)
    sample_img = sample_img.resize((224, 224), Image.ANTIALIAS)

    return sample_img

def divide_mask(seg_label):
    space = 0
    foreground_idx = -1
    for i in list(np.where(seg_label==1)[0]):
        if(i==0):
            continue
        if( np.where(seg_label==i)[0].shape[0] > space):
            space = np.where(seg_label==i)[0].shape[0]
            foreground_idx = i

    foreground = (seg_label==foreground_idx)
    objects = ((seg_label!=foreground_idx) * (seg_label!=0))
    background = (voc_sample_mask==0)

    return foreground,objects,background

def iou_calc(cams, mask):
    iou = 0.
    union = np.bitwise_or(mask, cams > np.median(cams))
    intersection = np.bitwise_and(mask,cams > np.median(cams))
    if(np.sum(union)!=0):
        iou = np.sum(intersection)/np.sum(union)
    return iou

def ioa_calc(cams, mask, ratio=0.9):
    ioa = 0.
    area = cams > np.max(cams)*ratio
    intersection = np.bitwise_and(mask, cams > np.max(cams)*ratio)
    """
    plt.subplot(3,3,1)
    plt.title('cam')
    plt.imshow(cams)
    plt.subplot(3,3,2)
    plt.title('area')
    plt.imshow(area)
    plt.subplot(3,3,3)
    plt.title('intersection')
    plt.imshow(intersection)
    """
    if(np.sum(area) != 0):
        ioa = np.sum(intersection)/np.sum(area)
    return ioa

def matrix_calc(cams, seg_label, threshold=0.9, metric='ioa'):
    # cams      [[W x H], [W x H], [W x H]]
    # seg_label [W x H]
    # ====>
    # matrix [3x3]
    matrix = np.zeros((3,3))
    foreground_id = None
    foreground_macecam_iou = 0

    # determine foreground
    for idx,i in enumerate(list(np.unique(seg_label)[1:])):
        temp = ioa_calc(cams[0], seg_label==i, ratio=threshold)
        if(temp > foreground_macecam_iou):
            foreground_macecam_iou,foreground_id = temp,i

    # calculate iou matrix
    #      mace    alea   epig
    # fore
    # obj.
    # back

    object_map = np.multiply(seg_label!=foreground_id, seg_label!=0)
    background_map = (seg_label==0)
    foreground_ious = np.array([foreground_macecam_iou,
                                ioa_calc(cams[1], seg_label==foreground_id, ratio= threshold),
                                ioa_calc(cams[2], seg_label==foreground_id, ratio= threshold)])
    matrix[0,:] = foreground_ious
    objects_ious = np.array([ioa_calc(cams[i], object_map,ratio=threshold) for i in range(3)])
    matrix[1,:] = objects_ious
    background_ious = np.array([ioa_calc(cams[i], background_map,ratio=threshold) for i in range(3)])
    matrix[2,:] = background_ious

    return matrix

def matrix_calc_cutout(cams, masks, metric='ioa'):
    # cams      [[W x H], [W x H], [W x H]]
    # masks [[W x H], [W x H], [W x H]]
    # ====>
    # matrix [3x3]
    matrix = np.zeros((3,3))

    cutout_region_ious = np.array([ioa_calc(cams[0], masks[0]),
                                    ioa_calc(cams[1], masks[0]),
                                    ioa_calc(cams[2], masks[0])])
    matrix[0,:] = cutout_region_ious
    body_part_ious = np.array([ioa_calc(cams[0], masks[1]),
                                    ioa_calc(cams[1], masks[1]),
                                    ioa_calc(cams[2], masks[1])])
    matrix[1,:] = body_part_ious
    background_ious = np.array([ioa_calc(cams[0], masks[2]),
                                    ioa_calc(cams[1], masks[2]),
                                    ioa_calc(cams[2], masks[2])])
    matrix[2,:] = background_ious

    return matrix

def calc_mat33(engine, voc_dataset,threshold=0.9):
    confusion_mat = np.zeros((3,3))

    return_list = list()
    sim = 1000

    for idx,(voc_sample_img, voc_sample_label, voc_sample_mask, voc_id) in enumerate(voc_dataset):
        voc_sample_img = (voc_sample_img.transpose(1,2,0)/255.+ voc_dataset.mean_bgr/255.)[:,:,::-1]
        inp = np.array(voc_sample_img*255) - np.array((104.008, 116.669, 122.675))
        inp = torch.tensor(inp)/255.
        voc_sample_mask = np.where(voc_sample_mask==255,0,voc_sample_mask)

        inpt = inp.permute(2,0,1).unsqueeze(0)

        voc_labels = list(np.where(voc_sample_label==1)[0])

        # calc prediction, uncertainty
        with torch.no_grad():
            output_dict = engine.model(inpt.cuda().float())
        pi, mu, sigma = output_dict['pi'],output_dict['mu'],output_dict['sigma']
        largest_pi_ind = torch.argmax(pi)
        unct_out = mln_uncertainties(pi, mu, sigma)
        alea , epis = float(unct_out['alea']), float(unct_out['epis'])
        sel_out = mln_gather(output_dict)
        mu_sel = sel_out['mu_sel'].cpu()
        ind_sel = torch.topk(mu_sel,5)[-1][0]
        pred_label_name = label_name[int(ind_sel[0].cpu().numpy())].split(',')[0]

        # calc cam
        cam_saver_list = list()
        for i,loss_type in enumerate(loss_types):

            engine.localizer.register_hooks()
            engine.localizer.model_ext.loss_type= loss_type

            pred_label = torch.tensor([ind_sel[0]]).long()

            cams = engine.localizer.localize(inpt.to('cuda').float(), pred_label.to('cuda'),largest_pi_ind)
            cams = cams.cpu().detach().squeeze().numpy()

            engine.localizer.remove_hooks()

            cam_saver_list.append(cams)

        # calc mat
        sample_mat_result = matrix_calc(cam_saver_list, np.where(voc_sample_mask==255,0,voc_sample_mask), threshold)
        best = True

        if(np.unique(np.where(voc_sample_mask==255,0,voc_sample_mask)).shape[0] >= 3):
            if best:
                save_mat = list()
                #save_mat.append(sample_mat_result)
                iden = np.eye(3)
                iden[-1,-1] = 0.
                sim_tmp = np.abs(iden-sample_mat_result).sum()
                print(sim_tmp)
                if(sim_tmp < 2.5):
                    print("Updated")
                    sim = sim_tmp
                    return_list.append([voc_sample_img, cam_saver_list,sample_mat_result,np.where(voc_sample_mask==255,0,voc_sample_mask),pred_label_name])

        confusion_mat +=sample_mat_result
    return confusion_mat, return_list

def parts_preproc(voc_parts, img_size = 224):
    num_objects = len(voc_parts['anno'][0,0][1][0])
    object_dict = dict()
    #print(voc_parts['anno'])
    for nobj in range(num_objects):
        part_dict = dict()
        obj_name = voc_parts['anno'][0,0][1][0][nobj][0][0]
        part = voc_parts['anno'][0,0][1][0][nobj][-1]
        num_parts= part.shape[-1]
        for i in range(num_parts):
            part_dict[part[0][i][0][0]] = cv2.resize(part[0][i][-1], dsize=(img_size, img_size), interpolation=cv2.INTER_NEAREST)
        object_dict[obj_name] = part_dict
    return object_dict

def part_vis(image, obj_dict, color_dict=color_dict):
    plt.imshow(np.array(image))

    for obj in obj_dict:
        for idx,part in enumerate(obj_dict[obj]):
            cmap = matplotlib.colors.ListedColormap(['none', color_dict[part]])
            plt.imshow(obj_dict[obj][part], cmap=cmap, alpha=0.5)

def filterOut(parts_path, voc_ids, voc_images, only=None):
    # voc_ids :: []
    # voc_images :: []
    new_ids, new_images, new_parts = list(),list(),list()
    for i in range(len(voc_ids)):
        voc_parts = io.loadmat(parts_path + "/" + voc_ids[i] + ".mat")
        obj_dict = parts_preproc(voc_parts)
        if(len(obj_dict)==1):
            if(len(obj_dict[list(obj_dict.keys())[0]]) > 0):
                if((only is not None) and (list(obj_dict.keys())[0] in only)):
                    new_ids.append(voc_ids[i])
                    new_images.append(voc_images[i])
                    new_parts.append(obj_dict)
                if(only is None):
                    new_ids.append(voc_ids[i])
                    new_images.append(voc_images[i])
                    new_parts.append(obj_dict)

    return new_ids, new_images, new_parts

def partsBlur(img, voc_part, vis=False, sigma=2):
    obj_name = None
    face_mask = voc_part[list(voc_part.keys())[0]]['head'] # 224 x 224

    cmap = matplotlib.colors.ListedColormap(['none', 'blue'])
    xx,yy= np.where(face_mask==1)
    bbox = [min(xx),min(yy),max(xx),max(yy)]
    blurred = blur_func(np.array(img),bbox,sigma=sigma)
    blurred = np.where(np.stack([face_mask==1,face_mask==1,face_mask==1],axis=-1), blurred, np.array(img))
    if(vis):
        fig, ax = plt.subplots()
        ax.imshow(np.array(img))
        #ax.imshow(face_mask, alpha=0.3, cmap=cmap)
        rect = patches.Rectangle((min(yy), min(xx)), max(yy)-min(yy), max(xx)-min(xx), linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

    return blurred

def partsCutout(img, voc_part, img2, voc_part2, vis=False):
    obj_name = None
    face_mask1 = voc_part[list(voc_part.keys())[0]]['head'] # 224 x 224
    face_mask2 = voc_part2[list(voc_part2.keys())[0]]['head'] # 224 x 224

    cmap = matplotlib.colors.ListedColormap(['none', 'blue'])
    xx,yy= np.where(face_mask1==1)
    xx2,yy2= np.where(face_mask2==1)
    bbox1 = [min(xx),min(yy),max(xx),max(yy)]
    bbox2 = [min(xx2),min(yy2),max(xx2),max(yy2)]
    cutout_img = cutout_func(np.array(img),np.array(img2),bbox1, bbox2)

    #print(bbox1)
    swap_region_h,swap_region_w = bbox1[2] - bbox1[0], bbox1[3] - bbox1[1]
    mask_patch = face_mask2[bbox2[0]:bbox2[2],bbox2[1]:bbox2[3]]
    mask_patch = cv2.resize(mask_patch, dsize=(swap_region_w, swap_region_h), interpolation=cv2.INTER_CUBIC)
    mask_map = np.zeros_like(face_mask2)
    mask_map[bbox1[0]:bbox1[2], bbox1[1]:bbox1[3]] = mask_patch

    cutout_img = np.where(np.stack([mask_map==1,mask_map==1,mask_map==1],axis=-1)
                          ,np.array(cutout_img), np.array(img))
    if(vis):
        plt.imshow(np.array(cutout_img))
        #plt.imshow(mask_map, alpha=0.5, cmap=cmap)

    return cutout_img

def plot_func(engine, image ,save_path = None, dataset='imagenet', vis=True):
    # engine :: Engine()
    # image  :: numpy array
    list_of_cams = list()

    img = np.array(image)
    if dataset=='imagenet':
        inp = img - np.array((104.008, 116.669, 122.675))
        inp = torch.tensor(inp)/255.
    elif dataset=='oxfordpets':
        mean, std = normalization_params()
        inp = (img/225.-np.array(mean))/np.array(std)
        inp = torch.tensor(inp)
    inpt = inp.permute(2,0,1).unsqueeze(0)

    loss_types = ['mace_avg','epis_avg','alea_avg']

    output_dict = engine.model(inpt.cuda().float())
    pi, mu, sigma = output_dict['pi'],output_dict['mu'],output_dict['sigma']

    pi_entropies_inds = list(torch.argsort(pi[0],descending=True).cpu().numpy())
    unct_out = mln_uncertainties(pi, mu, sigma)
    sel_out = mln_gather(output_dict)
    mu_sel = sel_out['mu_sel'].cpu()
    ind_sel = torch.topk(mu_sel,3)[-1][0]
    pred_label_name = None

    largest_pi_ind = torch.argmax(pi)
    mixture_k = engine.model.mol.fc_pi.weight.size(0)

    if vis:
        fig, ax = plt.subplots(3, 5, figsize=(15, 15))
        fig.subplots_adjust(hspace=0, wspace=0.2)
        font = {'family' : 'normal',
                'weight' : 'bold',
                'size'   : 10}
        plt.rc('font', **font)

        # Grad,Epis, Alea

        ax[0,-1].imshow(img)
        ax[0,-2].imshow(img)
    for i,loss_type in enumerate(loss_types):
        engine.localizer.register_hooks()

        engine.localizer.model_ext.loss_type= loss_type

        max_label = torch.tensor([ind_sel[0]]).long()

        cams = engine.localizer.localize(inpt.to('cuda').float(), max_label.to('cuda'))
        engine.localizer.remove_hooks()
        cams = cams.cpu().detach().squeeze().numpy()
        list_of_cams.append(cams)

        epis = unct_out['epis'][0]
        alea = unct_out['alea'][0]

        if vis:
            if loss_type=='mace_avg':
                if dataset=='imagenet':
                    pred_label_name = label_name[int(ind_sel[0].cpu().numpy())].split(',')[0]
                elif dataset=='oxfordpets':
                    classmap = list(engine.dataloaders['train'].dataset.class2idx.keys())
                    pred_label_name = classmap[int(ind_sel[0].cpu().numpy())]
                ax[0,i].set_title('Grad-CAM')
            else:
                if loss_type=='epis_avg':
                    ax[0,i].set_title('Epis-CAM')
                    ax[0,i].set_xlabel('σₑ : ' + str(round(float(epis),3)) )
                elif loss_type=='alea_avg':
                    ax[0,i].set_title('Alea-CAM')
                    ax[0,i].set_xlabel('σₐ : ' + str(round(float(alea),3)) )
            ax[0,i].imshow(img)
            cams = np.ma.masked_where(cams < np.mean(cams), cams)
            ax[0,i].imshow(cams, alpha=0.5, cmap='jet')
            ax[0,i].xaxis.set_major_locator(plt.NullLocator())
            ax[0,i].yaxis.set_major_locator(plt.NullLocator())
            #ax[0,i].text(60, 15, "a:{:.3f} e:{:.3f}".format(alea,epis), fontsize=10,color='red')

    # mixture plot

    mixture_cams = list()
    for i,k in enumerate(pi_entropies_inds[:5]):

        engine.localizer.model_ext.loss_type= 'mace_avg'
        if dataset=='imagenet':
            mixture_pred_name = label_name[int(torch.argmax(mu[0,k]).cpu().numpy())].split(',')[0]
        elif dataset=='oxfordpets':
            classmap = list(engine.dataloaders['train'].dataset.class2idx.keys())
            mixture_pred_name = classmap[int(torch.argmax(mu[0,k]).cpu().numpy())]

        max_label = torch.tensor([torch.argmax(mu[0,largest_pi_ind])]).long()

        engine.localizer.register_hooks()
        cams = engine.localizer.localize(inpt.to('cuda').float(), max_label.to('cuda'), mixture=k)
        engine.localizer.remove_hooks()
        cams = cams.cpu().detach().squeeze().numpy()
        list_of_cams.append(cams)
        mixture_cams.append(cams)

        if vis:
            if(largest_pi_ind==k):
                ax[1,i].set_title(mixture_pred_name,color='red')
            else:
                ax[1,i].set_title(mixture_pred_name,color='blue')

            ax[1,i].imshow(img)
            cams = np.ma.masked_where(cams < np.mean(cams), cams)
            ax[1,i].imshow(cams, alpha=0.5, cmap='jet')
            ax[1, i].xaxis.set_major_locator(plt.NullLocator())
            ax[1,i].yaxis.set_major_locator(plt.NullLocator())
            ax[1,i].set_xlabel('π : ' + str(round(float(pi[0,k].detach().cpu().numpy()),3)) )



    # top class pred
    for i,k in enumerate(pi_entropies_inds[:5]):
        engine.localizer.register_hooks()
        engine.localizer.model_ext.loss_type= 'mace_avg'
        if dataset=='imagenet':
            mixture_pred_name = label_name[int(torch.argmax(mu[0,k]).cpu().numpy())].split(',')[0]
        elif dataset=='oxfordpets':
            classmap = list(engine.dataloaders['train'].dataset.class2idx.keys())
            mixture_pred_name = classmap[int(torch.argmax(mu[0,k]).cpu().numpy())]

        max_label = torch.tensor([torch.argmax(mu[0,k])]).long()

        cams = engine.localizer.localize(inpt.to('cuda').float(), max_label.to('cuda'), mixture=k)
        cams = cams.cpu().detach().squeeze().numpy()
        list_of_cams.append(cams)

        if vis:
            if(largest_pi_ind==k):
                ax[2,i].set_title(mixture_pred_name,color='red')
            else:
                ax[2,i].set_title(mixture_pred_name,color='blue')

            ax[2,i].imshow(img)
            cams = np.ma.masked_where(cams < np.mean(cams), cams)
            ax[2,i].imshow(cams, alpha=0.5, cmap='jet')
            ax[2, i].xaxis.set_major_locator(plt.NullLocator())
            ax[2,i].yaxis.set_major_locator(plt.NullLocator())
            ax[2,i].set_xlabel('μ : ' + str(round(float(mu[0,k,max_label].detach().cpu().numpy()),3)) )

        engine.localizer.remove_hooks()

    # Mean, Var Mixtures
    mixture_cams = np.stack(mixture_cams)
    mean_cam = np.mean(mixture_cams,axis=0)
    var_cam = np.var(mixture_cams,axis=0)

    if vis:
        ax[0,-2].xaxis.set_major_locator(plt.NullLocator())
        ax[0,-2].yaxis.set_major_locator(plt.NullLocator())
        ax[0,-1].xaxis.set_major_locator(plt.NullLocator())
        ax[0,-1].yaxis.set_major_locator(plt.NullLocator())
        ax[0,-2].set_title('Mean-CAM')
        mean_cam = np.ma.masked_where(mean_cam < np.mean(mean_cam), mean_cam)
        ax[0,-2].imshow(mean_cam, alpha=0.5, cmap='jet')
        ax[0,-1].set_title('Var-CAM')
        var_cam = np.ma.masked_where(var_cam < np.mean(var_cam), var_cam)
        ax[0,-1].imshow(var_cam, alpha=0.5, cmap='jet')

        if save_path is not None:
            plt.savefig(save_path)

    return list_of_cams,alea,epis

def getUncertainyDistBlur(engine,new_images,new_parts):
    sigma_max = 6

    unc_dist_alea = [[] for i in range(sigma_max+1)]
    unc_dist_epis = [[] for i in range(sigma_max+1)]

    for i in range(0,sigma_max+1):
        for test_idx in tqdm(range(len(new_images))):
            try:
                blur_img = partsBlur(new_images[test_idx], new_parts[test_idx], sigma=i)
            except:
                continue
            _, alea,epis = plot_func(engine, blur_img, vis=False)
            unc_dist_alea[i].append(float(alea.detach().cpu().numpy()))
            unc_dist_epis[i].append(float(epis.detach().cpu().numpy()))

    return unc_dist_alea,unc_dist_epis

def getUncertainyDistCutOut(engine,new_images,new_parts):

    cutout_ln=1
    unc_dist_alea = [[] for i in range(cutout_ln+1)]
    unc_dist_epis = [[] for i in range(cutout_ln+1)]

    for test_idx in tqdm(range(len(new_images))):
        randIdx = random.choice([i for i in range(len(new_images))])
        while(test_idx== randIdx):
            randIdx = random.choice([i for i in range(len(new_images))]) # choose index randomly
        try:
            cutout_img = partsCutout(new_images[idx], new_parts[idx],
                                     new_images[randIdx], new_parts[randIdx])
        except:
            continue
        _, aleaCutout,episCutout = plot_func(engine, cutout_img, vis=False)
        _, alea,epis = plot_func(engine, new_images[idx], vis=False)

        unc_dist_alea[0].append(float(alea.detach().cpu().numpy()))
        unc_dist_epis[0].append(float(epis.detach().cpu().numpy()))
        unc_dist_alea[-1].append(float(aleaCutout.detach().cpu().numpy()))
        unc_dist_epis[-1].append(float(episCutout.detach().cpu().numpy()))

    return unc_dist_alea,unc_dist_epis

def calc_mat33_noise(engine, new_images, new_parts, target='cutout', sigma=2):
    confusion_mat_cutout = np.zeros((3,3))

    for idx in range(len(new_parts)):
        randIdx = idx
        class_name = list(new_parts[idx].keys())[0]
        while(idx== randIdx):
            randIdx = random.choice([i for i in range(len(new_images))]) # choose index randomly
        try:
            if(target=='cutout'):
                cutout_img = partsCutout(new_images[idx], new_parts[idx],
                                         new_images[randIdx], new_parts[randIdx])
            elif(target=='blur'):
                cutout_img = partsBlur(new_images[idx], new_parts[idx],sigma=sigma)
        except:
            continue
        list_of_cams, alea, epis = plot_func(engine, cutout_img, vis=False) # get CAMs, uncertainty
        parts_names = list(new_parts[idx][class_name].keys())

        mask = np.sum([new_parts[idx][class_name][part] for part in parts_names], axis=0)
        head_mask = np.sum([new_parts[idx][class_name][part] for part in parts_names if part == 'head'], axis=0)
        nothead_mask = np.sum([new_parts[idx][class_name][part] for part in parts_names if part != 'head'], axis=0)
        nothead_mask = np.clip(nothead_mask, 0, 1)
        mask = np.clip(mask, 0, 1)
        head_mask = np.clip(head_mask, 0, 1)
        nothead_mask = np.where(head_mask==nothead_mask, 0, nothead_mask)
        background_mask = 1-mask

        masks = [head_mask,nothead_mask,background_mask]
        cam_saver_list = [list_of_cams[0],list_of_cams[1],list_of_cams[2]]

        confusion_mat_cutout += matrix_calc_cutout(cam_saver_list,masks)
    return confusion_mat_cutout

# Path utils
# ==========

def mkdir_p(path):
    os.makedirs(path, exist_ok=True)
    return path


# MultiGPU
# ========

class DataParallel(torch.nn.DataParallel):
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


# Data
# ====

def getBlurAlea(cams, origin_img, ratio=7, mask_ratio=0.2):
    results = list()

    for idx in range(cams.shape[0]):
        alea_cam = cams[idx]
        blur_img = cv2.GaussianBlur(origin_img[idx].numpy(),(0,0),ratio)
        alea_mask = alea_cam> mask_ratio*np.max(alea_cam)
        blur_part_alea = blur_img*(np.stack([alea_mask,alea_mask,alea_mask],axis=-1))
        nonblur_part_alea = origin_img[idx]*(1-np.stack([alea_mask,alea_mask,alea_mask],axis=-1))
        blur_alea = nonblur_part_alea + blur_part_alea
        results.append(blur_alea)
    return results

def normalization_params():
#CUB200
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    #mean = [0.5, 0.5, 0.5]
    #std = [0.5, 0.5, 0.5]

    #mean = np.array((104.008, 116.669, 122.675))
    #std = [1,1,1]

    return (mean, std)

def normalization_params2():
    #openImages30k
    mean = [0.5, 0.5 , 0.5]
    std = [1,1,1]

    return (mean, std)

def unnormalize_images(images):
    mean, std = normalization_params()
    mean = torch.reshape(torch.tensor(mean), (1, 3, 1, 1))
    std = torch.reshape(torch.tensor(std), (1, 3, 1, 1))
    unnormalized_images = images.clone().detach().cpu() * std + mean
    return unnormalized_images


# MLN
# ===

def mln_gather(output_dict):
    """
        :param pi:      [N x K]
        :param mu:      [N x K x D]
        :param sigma:   [N x K x D]
    """
    pi, mu, sigma = output_dict['pi'], output_dict['mu'], output_dict['sigma']
    max_idx = torch.argmax(pi, dim=1) # [N]
    idx_gather = max_idx.unsqueeze(dim=-1).repeat(
        1, mu.shape[2]).unsqueeze(1) # [N x 1 x D]
    mu_sel = torch.gather(mu, dim=1, index=idx_gather).squeeze(dim=1) # [N x D]
    sigma_sel = torch.gather(sigma, dim=1, index=idx_gather).squeeze(dim=1) # [N x D]
    out = {
        'max_idx':max_idx, # [N]
        'idx_gather':idx_gather, # [N x 1 x D]
        'mu_sel':mu_sel, # [N x D]
        'sigma_sel':sigma_sel # [N x D]
    }
    return out

# CAM
# ===

def extract_bbox(images, cams, gt_boxes, loc_threshold=0.2, color=[(0, 255, 0)]):
    # Convert the format of threshold and percentile
    if not isinstance(loc_threshold, list):
        loc_threshold = [loc_threshold]

    # Generate colors
    gt_color = (0, 0, 255) # (0, 0, 255)
    line_thickness = 2
    from itertools import cycle, islice
    color = list(islice(cycle(color), len(loc_threshold)))

    # Convert a data format
    images = images.clone().numpy().transpose(0, 2, 3, 1)
    images = images[:, :, :, ::-1] * 255 # reverse the color representation(RGB -> BGR) and Opencv format
    cams = cams.clone().detach().cpu().numpy().transpose(0, 2, 3, 1)

    bboxes = []
    blended_bboxes = []
    for i in range(images.shape[0]):
        image, cam, gt_box = images[i].astype('uint8'), cams[i], gt_boxes[i]

        # Normalize a cam
        cam_max, cam_min = np.amax(cam), np.amin(cam)
        normalized_cam = (cam - cam_min) / (cam_max - cam_min) * 255
        normalized_cam = normalized_cam.astype('uint8')

        # Generate a heatmap using jet colormap
        heatmap_jet = cv2.applyColorMap(normalized_cam, cv2.COLORMAP_JET)
        blend = cv2.addWeighted(heatmap_jet, 0.5, image, 0.5, 0)
        blended_bbox = blend.copy()

        try:
            cv2.rectangle(blended_bbox,
                    pt1=(gt_box[0], gt_box[1]),
                    pt2=(gt_box[2], gt_box[3]),
                    color=gt_color, thickness=line_thickness)
        except:
            pass

        for _threshold, _color in zip(loc_threshold, color):
            _, thresholded_gray_heatmap = cv2.threshold(
                normalized_cam, _threshold, maxval=255, type=cv2.THRESH_BINARY)

            try:
                _, contours, _ = cv2.findContours(thresholded_gray_heatmap,
                                                  cv2.RETR_TREE,
                                                  cv2.CHAIN_APPROX_SIMPLE)
            except:
                contours, _ = cv2.findContours(thresholded_gray_heatmap,
                                            cv2.RETR_TREE,
                                            cv2.CHAIN_APPROX_SIMPLE)

            bbox = [0, 0, 224, 224]
            if len(contours) > 0:
                max_contour = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(max_contour)
                bbox = [x, y, x + w, y + h]
                cv2.rectangle(blended_bbox,
                              pt1=(x, y), pt2=(x + w, y + h),
                              color=_color, thickness=line_thickness)

        blended_bbox = blended_bbox[:,:,::-1] / 255.0
        blended_bbox = blended_bbox.transpose(2, 0, 1)
        bboxes.append(torch.tensor(bbox))
        blended_bboxes.append(torch.tensor(blended_bbox))

    bboxes = torch.stack(bboxes)
    blended_bboxes = torch.stack(blended_bboxes)
    return bboxes, blended_bboxes
