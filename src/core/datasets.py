import os
import cv2
import torch
import random
import numpy as np
import pandas as pd
from glob import glob

from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as F
from torchvision import datasets, transforms
from torchvision.datasets import MNIST
import torch

from numpy.testing import assert_array_almost_equal

def load_image(filename) :
    img = Image.open(filename)
    img = img.convert('RGB')
    return img

# basic function
def multiclass_noisify(y, P, random_state=0):
    """ Flip classes according to transition probability matrix T.
    It expects a number between 0 and the number of classes - 1.
    """
    assert P.shape[0] == P.shape[1]
    assert np.max(y) < P.shape[0]

    # row stochastic matrix
    assert_array_almost_equal(P.sum(axis=1), np.ones(P.shape[1]))
    assert (P >= 0.0).all()

    m = y.shape[0]
    new_y = y.copy()
    flipper = np.random.RandomState(random_state)

    for idx in np.arange(m):
        i = y[idx]
        # draw a vector with only an 1
        flipped = flipper.multinomial(1, P[i, :][0], 1)[0]
        new_y[idx] = np.where(flipped == 1)[0]

    return new_y


# noisify_pairflip call the function "multiclass_noisify"
def noisify_pairflip(y_train, noise, random_state=None, nb_classes=10):
    """mistakes:
        flip in the pair
    """
    P = np.eye(nb_classes)
    n = noise

    if n > 0.0:
        # 0 -> 1
        P[0, 0], P[0, 1] = 1. - n, n
        for i in range(1, nb_classes-1):
            P[i, i], P[i, i + 1] = 1. - n, n
        P[nb_classes-1, nb_classes-1], P[nb_classes-1, 0] = 1. - n, n

        y_train_noisy = multiclass_noisify(y_train, P=P,
                                           random_state=random_state)
        actual_noise = (y_train_noisy != y_train).mean()
        assert actual_noise > 0.0
        print('Actual noise %.2f' % actual_noise)
        y_train = y_train_noisy
    #print(P)

    return y_train, actual_noise

def noisify_multiclass_symmetric(y_train, noise, random_state=None, nb_classes=10):
    """mistakes:
        flip in the symmetric way
    """
    P = np.ones((nb_classes, nb_classes))
    n = noise
    P = (n / (nb_classes - 1)) * P

    if n > 0.0:
        # 0 -> 1
        P[0, 0] = 1. - n
        for i in range(1, nb_classes-1):
            P[i, i] = 1. - n
        P[nb_classes-1, nb_classes-1] = 1. - n

        y_train_noisy = multiclass_noisify(y_train, P=P,
                                           random_state=random_state)
        actual_noise = (y_train_noisy != y_train).mean()
        assert actual_noise > 0.0
        print('Actual noise %.2f' % actual_noise)
        y_train = y_train_noisy
    print(P)

    return y_train, P

class CUB200(Dataset):
    def __init__(self, root, log, mode, transform=None, transform_config=None):
        self.root = root
        self.log = log
        self.mode = mode
        if transform_config is not None:
            image_size = float(transform_config.get('image_size', 256))
            crop_size = transform_config.get('crop_size', 224)
            shift = (image_size - crop_size) // 2
        self.data = self._load_data(image_size, crop_size, shift)
        self.transform = transform

    def _load_data(self, image_size, crop_size, shift):
        self._labelmap_path = os.path.join(self.root, 'CUB_200_2011', 'classes.txt')

        paths = pd.read_csv(
            os.path.join(self.root, 'CUB_200_2011', 'images.txt'),
            sep=' ', names=['id', 'path'])
        labels = pd.read_csv(
            os.path.join(self.root, 'CUB_200_2011', 'image_class_labels.txt'),
            sep=' ', names=['id', 'label'])
        splits = pd.read_csv(
            os.path.join(self.root, 'CUB_200_2011', 'train_test_split.txt'),
            sep=' ', names=['id', 'is_train'])
        orig_image_sizes = pd.read_csv(
            os.path.join(self.root, 'CUB_200_2011', 'image_sizes.txt'),
            sep=' ', names=['id', 'width', 'height'])
        bboxes = pd.read_csv(
            os.path.join(self.root, 'CUB_200_2011', 'bounding_boxes.txt'),
            sep=' ', names=['id', 'x', 'y', 'w', 'h'])

        resized_xmin = np.maximum(
            (bboxes.x / orig_image_sizes.width * image_size - shift).astype(int), 0)
        resized_ymin = np.maximum(
            (bboxes.y / orig_image_sizes.height * image_size - shift).astype(int), 0)
        resized_xmax = np.minimum(
            ((bboxes.x + bboxes.w - 1) / orig_image_sizes.width * image_size - shift).astype(int),
            crop_size - 1)
        resized_ymax = np.minimum(
            ((bboxes.y + bboxes.h - 1) / orig_image_sizes.height * image_size - shift).astype(int),
            crop_size - 1)

        resized_bboxes = pd.DataFrame({'id': paths.id,
                                       'xmin': resized_xmin.values,
                                       'ymin': resized_ymin.values,
                                       'xmax': resized_xmax.values,
                                       'ymax': resized_ymax.values})

        data = paths.merge(labels, on='id')\
                    .merge(splits, on='id')\
                    .merge(resized_bboxes, on='id')

        if self.mode == 'train':
            data = data[data.is_train == 1]
        else:
            data = data[data.is_train == 0]
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        path = os.path.join(self.root, 'CUB_200_2011/images', sample.path)
        image = Image.open(path).convert('RGB')
        label = sample.label - 1 # label starts from 1
        gt_box = torch.tensor(
            [sample.xmin, sample.ymin, sample.xmax, sample.ymax])

        if self.transform is not None:
            image = self.transform(image)

        return (image, label, gt_box)

class VOC2012N(Dataset):
    def __init__(self, root, log, mode, transform=None, transform_config=None):
        self.log = log
        self.mode = mode
        self.transform = transform
        self.voc = datasets.voc.VOCSegmentation(root=root)

    def __len__(self):
        return len(self.voc)

    def __getitem__(self,idx):
        img, target = self.voc[idx]
        img = img.convert('RGB').astype(np.float32)

        if self.transform is not None:
            img = self.transform(img)

        return (img, target, None)

class VOC2012(Dataset):
    def __init__(self, root, crop_size=321, num_classes=21, metadata_split='train_aug',
                 label_split='SegmentationClassAug', train_full=None, scale=True, flip=True, ood=False, outdist=False, together=False, noise=True):
        self.data_dir = os.path.join(root, 'VOC2012')
        self.image_dir_path = os.path.join(self.data_dir, 'JPEGImages')
        self.label_dir_path = os.path.join(self.data_dir, label_split)
        self.id_path = os.path.join('metadata', 'VOC2012', metadata_split + '.txt')
        self.metadata_split = metadata_split

        self.image_ids = [i.strip() for i in open(self.id_path) if not i.strip() == ' ']

        self.train_full_ids = None
        if train_full is not None:
            train_full_path = os.path.join('metadata', 'VOC2012', train_full + '.txt')
            train_full_ids = [i.strip() for i in open(train_full_path) if not i.strip() == ' ']
            self.image_ids = train_full_ids

        self.crop_size = crop_size
        self.mean_bgr = np.array((104.008, 116.669, 122.675))

        self.scale = scale
        self.flip = flip
        self.num_classes = num_classes

        self.scale_ratio = [0.5, 0.75, 1.0, 1.25, 1.5]
        #self.scale_ratio = [1.0]
        self.ignore_label = 255
        self.base_size = None

        self.ood = ood
        self.outdist = outdist
        self.together = together
        self.noise = noise
        if(self.ood and self.metadata_split!='test'):
            self.preprocess()

    def __len__(self):
        return len(self.image_ids)

    def preprocess(self):
        new_image_idx_list = list()

        indist_list  = [8,9,10,11,12,13,14,15,16,17]
        outdist_list = [1,2,3,4,5,6,7,18,19,20]

        for idx in range(len(self.image_ids)):
            image_id = self.image_ids[idx]
            image_path = os.path.join(self.image_dir_path, image_id + '.jpg')
            label_path = os.path.join(self.label_dir_path, image_id + '.png')

            # Load an image
            image = cv2.imread(image_path, cv2.IMREAD_COLOR).astype(np.float32)
            if self.metadata_split == 'test':
                seg_label = np.zeros_like(image[:,:,0], dtype=np.int32)
            else:
                seg_label = np.asarray(Image.open(label_path), dtype=np.int32)

            if self.metadata_split=='val':
                image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_LINEAR)
                seg_label = Image.fromarray(seg_label).resize((224, 224), resample=Image.NEAREST)
                seg_label = np.asarray(seg_label, dtype=np.int64)
            else:
                image, seg_label = self._augmentation(image, seg_label)

            label = np.zeros(self.num_classes)
            unique_labels = np.array([elem for elem in np.unique(seg_label) if elem != self.ignore_label])
            flag = False

            if not self.together:
                for label in unique_labels:
                    #indist
                    if(not self.outdist):
                        if(label in outdist_list):
                            flag = True
                            break
                    #outdist
                    else:
                        if(label in indist_list):
                            flag = True
                            break
            else:
                # in-dist and out-dist together
                is_indist = False
                is_outdist = False
                for label in unique_labels:
                    if(label in indist_list):
                        is_indist = True
                    if(label in outdist_list):
                        is_outdist = True
                flag = not (is_indist and is_outdist)


            if(not flag):
                new_image_idx_list.append(image_id)

        self.image_ids = new_image_idx_list

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_path = os.path.join(self.image_dir_path, image_id + '.jpg')
        label_path = os.path.join(self.label_dir_path, image_id + '.png')


        # Load an image
        image = cv2.imread(image_path, cv2.IMREAD_COLOR).astype(np.float32)
        if self.metadata_split == 'test':
            seg_label = np.zeros_like(image[:,:,0], dtype=np.int32)
        else:
            seg_label = np.asarray(Image.open(label_path), dtype=np.int32)

        if self.metadata_split=='val':
            image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_LINEAR)
            seg_label = Image.fromarray(seg_label).resize((224, 224), resample=Image.NEAREST)
            seg_label = np.asarray(seg_label, dtype=np.int64)
        else:
            image, seg_label = self._augmentation(image, seg_label)

        image -= self.mean_bgr
        image = image.transpose(2, 0, 1).astype(np.float32)

        seg_label = seg_label.astype(np.int64)
        label = np.zeros(self.num_classes)
        unique_labels = np.array([elem for elem in np.unique(seg_label) if elem != self.ignore_label])
        if len(unique_labels) >= 1:
            label[unique_labels] = 1
        label = label.astype(np.int64)
        if(self.noise and (random.uniform(0,1) <=0.2)):
            label[unique_labels[1]] = 0
            label[random.randint(1,20)] = 1

        return (image, label, seg_label, image_id)

    def _augmentation(self, image, label):
        # Scaling
        if self.scale:
            h, w = label.shape
            if self.base_size:
                if h > w:
                    h, w = (self.base_size, int(self.base_size * w / h))
                else:
                    h, w = (int(self.base_size * h / w), self.base_size)
            scale_factor = random.choice(self.scale_ratio)
            h, w = (int(h * scale_factor), int(w * scale_factor))
            image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
            label = Image.fromarray(label).resize((w, h), resample=Image.NEAREST)
            label = np.asarray(label, dtype=np.int64)

        # Padding to fit for crop_size
        h, w = label.shape
        pad_h = max(self.crop_size - h, 0)
        pad_w = max(self.crop_size - w, 0)
        pad_kwargs = {
            "top": 0,
            "bottom": pad_h,
            "left": 0,
            "right": pad_w,
            "borderType": cv2.BORDER_CONSTANT,
        }
        if pad_h > 0 or pad_w > 0:
            image = cv2.copyMakeBorder(image, value=self.mean_bgr, **pad_kwargs)
            label = cv2.copyMakeBorder(label, value=self.ignore_label, **pad_kwargs)

        # Cropping
        h, w = label.shape
        start_h = random.randint(0, h - self.crop_size)
        start_w = random.randint(0, w - self.crop_size)
        #start_h = 0
        #start_w = 0
        end_h = start_h + self.crop_size
        end_w = start_w + self.crop_size
        image = image[start_h:end_h, start_w:end_w]
        label = label[start_h:end_h, start_w:end_w]

        # Random flipping
        if self.flip:
            if random.random() < 0.5:
                image = np.fliplr(image).copy()  # HWC
                label = np.fliplr(label).copy()  # HW

        return image, label

class OpenImages30k(Dataset):
    def __init__(self, root, log, mode, transform=None):
        self.mode =mode
        self.root = root
        self.img_size = 224
        self.dataset = datasets.ImageFolder(self.root + "/" + self.mode)
        self.transform = transform
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        if self.transform is not None:
            image = self.transform(image)
        return image, label, 0

class ImageNet(Dataset):
    def __init__(self, root, log, mode, transform=None, noise=False):
        self.mode =mode

        self.root = root
        with open('./src/utils/imagenet1000_clsidx_to_labels.txt', "r") as f:
            self.clsidx_to_labels =  eval(f.read())
        self.map_clsloc = dict()
        with open('./src/utils/map_clsloc.txt', "r") as f:
            tmp =  f.read().split('\n')
        tmp.sort()
        for idx,line in enumerate(tmp):
            fname = line.split(' ')[0]
            self.map_clsloc[fname] = idx

        self.mode = 'train'
        self.relabel_path = "/sdb/relabel"
        self.img_size = 224
        #self.dataset = datasets.ImageFolder(self.root + "/" + self.mode)
        #self.dataset = datasets.ImageFolder(self.root + "/train")
        self.dataset = self.imagenet_dataset()
        self.noise= noise
        if self.noise:
            self.relabel_classes = os.listdir(self.relabel_path)
            self.classes = self.relabel_classes
        self.transform = transform
    def __len__(self):
        return len(self.dataset)

    def imagenet_dataset(self, mode='train'):
        self.label_classes = os.listdir(self.root+"/" + mode)
        self.list_ = list()
        for i in self.label_classes:
            self.list_ += os.listdir(self.root+'/'+mode+'/'+i)
        return self.list_

    def get_sample_from_imagename(self,path):
        # imagenet :: n01930112_3994.JPEG
        # relabel  :: n01930112_23621.pt
        seg = torch.zeros(2,5,15,15) # [2,5,15,15]
        name = path.split('.')[0]
        class_name = path.split('_')[0]
        class_num = self.map_clsloc[class_name]
        image = Image.open(self.root + "/" + self.mode + "/" + class_name + "/" + name + ".JPEG")
        if self.noise:
            seg   = torch.load(self.relabel_path + "/" + class_name + "/" + name + ".pt")

        return image, class_num ,seg

    def imagenet_relabel_transform(self, input, target, mode):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        crop_size = (224,224)

        input = F.resize(input, (256,256))
        if mode=='train':
            i, j, h, w = transforms.RandomCrop.get_params(input, crop_size)
            org_h, org_w = input.size[0],input.size[1]

            input = F.crop(input, i, j, h, w)
            target = torch.nn.functional.upsample_nearest(target, size=(org_w,org_h), scale_factor=None)
            target = target[:,:,i:i+h,j:j+w]

        else:
            i, j, h, w = transforms.CetnerCrop.get_params(input, crop_size)
            input = F.crop(input, i, j, h, w)
            target = torch.nn.functional.upsample_nearest(target, size=(org_w,org_h), scale_factor=None)
            target = target[:,:,i:i+h,j:j+w]

        if random.random() > 0.5:
            input = F.hflip(input)
            target = F.hflip(target)

        input = F.to_tensor(input)
        #print(input.shape)
        if input.shape[0] == 1:
            input = torch.cat([input,input,input],dim=0)
        elif input.shape[0] == 4:
            input = input[:3,:,:]
        input = F.normalize(input, mean, std)
        return {'image':input, 'seg_label': target}

    def __getitem__(self, idx):
        path = self.dataset[idx]
        image, label, seg_label = self.get_sample_from_imagename(path)

        sample = {'image': image, 'seg_label': seg_label}
        sample = self.imagenet_relabel_transform(image, seg_label, self.mode)
        pd = list()
        if self.noise:

            label_map_up = sample['seg_label'] # [2, 5, 224, 224])

            counts = torch.unique(label_map_up[1,0], return_counts=True)
            rank = counts[0][torch.argsort(counts[1],descending=True)]

            for k in rank:
                pd.append(torch.sum(seg_label[1,0]==int(k))/(224.*224.))

            eps = 1-np.sum(np.array([float(p) for p in pd]))
            pd[0] += eps

            draw = np.random.choice([int(i) for i in rank], 1, p=np.array([p for p in pd]))
            label = int(draw)

        image = sample['image']
        seg_label = sample['seg_label']

        return image, label, seg_label

class AmbiguousMNIST(Dataset):
    def __init__(self, root, train=True, device=None, transform = None):
        # Scale data to [0,1]
        self.data = torch.load(os.path.join(root, "amnist_samples.pt")).to(device)
        self.targets = torch.load(os.path.join(root, "amnist_labels.pt")).to(device)

        # Normalize it with the usual MNIST mean and std
        self.data = self.data.sub_(0.1307).div_(0.3081)

        num_multi_labels = self.targets.shape[1]

        self.data = self.data.expand(-1, num_multi_labels, 28, 28).reshape(-1, 1, 28, 28)
        self.targets = self.targets.reshape(-1)

        data_range = slice(None, 60000) if train else slice(60000, None)
        self.data = self.data[data_range]
        self.targets = self.targets[data_range]
        self.transform = transform

    # The number of items in the dataset
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        if self.transform is not None:
            img = self.transform(img)

        return img, target, target

class MNIST_Wrapper(MNIST):
    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, target

class OxfordPets(Dataset):
    def __init__(self, root, log, mode, transform=None):
        filenames = glob(root + '/*.jpg')

        classes = set()
        data = []
        labels = []

        # Load the images and get the classnames from the image path
        for idx,image in enumerate(filenames):
            class_name = image.rsplit("/", 1)[1].rsplit('_', 1)[0]
            classes.add(class_name)

            data.append(image)
            labels.append(class_name)

        # convert classnames to indices
        class2idx = {cl: idx for idx, cl in enumerate(classes)}
        labels = torch.Tensor(list(map(lambda x: class2idx[x], labels))).long()

        data = list(zip(data, labels))

        self.data = data
        self.len = len(data)
        self.transforms = transform
        self.class2idx = class2idx

    def __getitem__(self, index):
        img_path, label = self.data[index]
        img = load_image(img_path)

        if self.transforms is not None:
            img = self.transforms(img)

        return img, label, label

    def __len__(self):
        return self.len

if __name__=='__main__':
    import matplotlib.pyplot as plt
    import random
    import matplotlib
    import numpy as np
    matplotlib.use('tkagg')

    #dataset = ImageNet("/sdb/ILSVRC/Data/CLS-LOC", None, 'train')
    #dataset = AmbiguousMNIST("/home/yo0n/바탕화면/RIL/ucam/data/",True)
    dataset = OxfordPets("/home/yo0n/바탕화면/RIL/ucam/data/oxford_pets", None, 'train', None)

    image , label, _ = dataset[random.randint(0,len(dataset))]

    plt.title(list(dataset.class2idx.keys())[label])
    plt.imshow(np.array(image))
    plt.show()
