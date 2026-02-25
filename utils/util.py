import torch
import os
import errno

import math
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms


def generate_instancedependent_candidate_labels(model, train_X, train_Y, 
                                                device='cuda',
                                                RATE=0.4, batch_size=2000):
    with torch.no_grad():
        k = int(torch.max(train_Y) - torch.min(train_Y) + 1)
        n = train_Y.shape[0]
        model = model.to(device)
        train_Y = torch.nn.functional.one_hot(train_Y, num_classes=k)
        avg_C = 0
        partialY_list = []
        rate = RATE
        step = math.ceil(n / batch_size)

        for i in range(0, step):
            b_end = min((i + 1) * batch_size, n)

            train_X_part = train_X[i * batch_size: b_end].to(device)

            outputs = model(train_X_part)

            train_p_Y = train_Y[i * batch_size: b_end].clone().detach()

            partial_rate_array = F.softmax(outputs, dim=1).clone().detach()
            partial_rate_array[torch.where(train_p_Y == 1)] = 0
            partial_rate_array = partial_rate_array / torch.max(partial_rate_array, dim=1, keepdim=True)[0]
            partial_rate_array = partial_rate_array / partial_rate_array.mean(dim=1, keepdim=True) * rate
            partial_rate_array[partial_rate_array > 1.0] = 1.0

            m = torch.distributions.binomial.Binomial(total_count=1, probs=partial_rate_array)
            z = m.sample()

            train_p_Y[torch.where(z == 1)] = 1.0
            partialY_list.append(train_p_Y)

        partialY = torch.cat(partialY_list, dim=0).float()

        assert partialY.shape[0] == train_X.shape[0]

    avg_C = torch.sum(partialY) / partialY.size(0)
    print(avg_C)

    return partialY


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape((-1, )).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res[0]

def test(args, epoch, test_loader, model):
    with torch.no_grad():
        model.eval()
        top1_acc = AverageMeter("Top1")

        for batch_idx, (images, labels) in enumerate(test_loader):
            images, labels = images.cuda(), labels.cuda()
            outputs = model(images,images,eval_only=True)
            acc1 = accuracy(outputs, labels)
            top1_acc.update(acc1[0])

    return top1_acc.avg


def mkdir_if_missing(directory):
    if not os.path.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


class Partialize_Dataset(Dataset):
    def __init__(self, images, given_partial_label_matrix, true_labels, images_diff, diff_targets,
                 classify_transform, weak_transform, strong_transform, nomalize, normalize_diff=None,
                 image_path=False):
        
        self.ori_images = images
        self.given_partial_label_matrix = given_partial_label_matrix
        self.true_labels = true_labels
        self.images_diff = images_diff
        label_count = self.given_partial_label_matrix.sum(1)
        label_cumsum = label_count.cumsum(0)
        self.diff_endidx = label_cumsum.int()
        diff_beginidx = torch.zeros_like(self.diff_endidx)
        diff_beginidx[1:] = label_cumsum[:-1]
        self.diff_beginidx = diff_beginidx.int()
        self.padding_count = int(label_count.max().item())
        self.diff_targets = diff_targets
        self.diff_count = given_partial_label_matrix.sum(1)
        self.img_diff_cls = []
        self.image_path = image_path
        
        for i in range(given_partial_label_matrix.shape[1]):
            self.img_diff_cls.append(self.images_diff[self.diff_targets==i])
        
        self.classify_transform = classify_transform
        self.weak_transform = weak_transform
        self.strong_transform = strong_transform
        self.normalize_diff = self.normalize = nomalize
        if normalize_diff is not None:
            self.normalize_diff = normalize_diff
    
    
    def __len__(self):
        return len(self.true_labels)
    
        
    def __getitem__(self, index):
        if self.image_path:
            images = Image.open(self.ori_images[index]).convert("RGB")
        else:
            images = self.ori_images[index]
        
        each_image_w = self.normalize(self.weak_transform(images))
        each_image_cls= self.normalize(self.classify_transform(images))
        each_label = self.given_partial_label_matrix[index]
        each_true_label = self.true_labels[index]
        
        each_image_diff = self.images_diff[self.diff_beginidx[index]:self.diff_endidx[index]]
        size_tuple = tuple(int(dim) for dim in each_image_w.shape)
        img_diff_w = torch.randn((self.padding_count, *size_tuple))
        img_diff_s = torch.randn((self.padding_count, *size_tuple))
        img_diff_d = torch.randn((self.padding_count, *size_tuple))
        diff_idx = torch.zeros(self.padding_count, dtype=torch.bool)
        for i, img_diff in enumerate(each_image_diff):
            if self.image_path:
                img_diff = transforms.ToPILImage()(img_diff)
            img_diff_w[i] = self.normalize_diff(self.weak_transform(img_diff))
            img_diff_s[i] = self.normalize_diff(self.strong_transform(img_diff))
            img_diff_d[i] = self.normalize_diff(self.classify_transform(img_diff))
        diff_idx[:len(each_image_diff)] = True
    
        return each_image_cls, each_image_w, img_diff_w, img_diff_s, img_diff_d, each_label, each_true_label, index, diff_idx


class Ori_Partialize(Dataset):
    def __init__(self, images, given_partial_label_matrix, true_labels, classes, transform=None):
        self.ori_images = images
        self.given_partial_label_matrix = given_partial_label_matrix
        self.true_labels = true_labels
        self.classes = classes
        self.transform=transform

    def __len__(self):
        return len(self.true_labels)
        
    def __getitem__(self, index):
        
        images = self.ori_images[index]
        if self.transform is not None:
            images = self.transform(images)
        each_label = self.given_partial_label_matrix[index]
        each_true_label = self.true_labels[index]
        
        return images, each_label, each_true_label, index


def labels_multi2single(multi_labels):
    num_labels = multi_labels.size(1)
    expanded_tensor = multi_labels.unsqueeze(2).expand(-1, -1, num_labels)
    identity_matrix = torch.eye(num_labels)
    identity_matrix = identity_matrix.to(multi_labels.device)
    single_labels_oh = expanded_tensor * identity_matrix
    single_labels_oh = single_labels_oh[multi_labels == 1]
    single_labels_oh = single_labels_oh.view(-1, num_labels)
    return single_labels_oh
