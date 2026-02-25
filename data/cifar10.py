import torch
import os
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from partial_models.wide_resnet import WideResNet
from augment.randaugment import RandomAugment
from augment.cutout import Cutout
from augment.autoaugment_extra import CIFAR10Policy
from utils.util import Ori_Partialize, Partialize_Dataset, generate_instancedependent_candidate_labels, mkdir_if_missing


def load_cifar10(args):
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    ])
    
    original_train = dsets.CIFAR10(root='./data/CIFAR10', train=True, download=True, transform=transforms.ToTensor())
    ori_data, ori_labels = original_train.data, torch.Tensor(original_train.targets).long()
    
    test_dataset = dsets.CIFAR10(root='./data/CIFAR10', train=False, transform=test_transform)
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=args.batch_size, \
        shuffle=False, num_workers=6, pin_memory=False
    )
    
    if os.path.exists(f'plabels/partialY_{args.dataset}_{args.seed}.pt'):
        print('loading labels')
        partialY_matrix = torch.load(f'plabels/partialY_{args.dataset}_{args.seed}.pt', map_location='cpu')
    else:
        print('generating labels')
        mkdir_if_missing('plabels')
        ori_data = torch.Tensor(original_train.data)
        model = WideResNet(depth=28, num_classes=10, widen_factor=10, dropRate=0.3)
        model.load_state_dict(torch.load(os.path.expanduser('./partial_models/weights/cifar10.pt')))
        ori_data = ori_data.permute(0, 3, 1, 2)
        partialY_matrix = generate_instancedependent_candidate_labels(model, ori_data, ori_labels, args.rate)
        torch.save(partialY_matrix, f'plabels/partialY_{args.dataset}_{args.seed}.pt')
    ori_data = original_train.data

    temp = torch.zeros(partialY_matrix.shape)
    temp[torch.arange(partialY_matrix.shape[0]), ori_labels] = 1
    
    if torch.sum(partialY_matrix * temp) == partialY_matrix.shape[0]:
        print('data loading done !')
    
    images_diff = torch.load(f'res_diff/diff_{args.dataset}.pt', map_location='cpu')
    idx_diff = torch.load(f'res_diff/idx_fuge_{args.dataset}.pt', map_location='cpu').long()
    idx_diff = idx_diff[partialY_matrix == 1]
    images_diff = images_diff[idx_diff].squeeze(1)
    diff_targets = torch.nonzero(partialY_matrix, as_tuple=False)[:, 1]
    
    partial_training_dataset = cifar10_partialize(ori_data, partialY_matrix.float(), ori_labels.float(), 
                                                  images_diff, diff_targets)

    partial_training_dataloader = torch.utils.data.DataLoader(
        dataset=partial_training_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=6,
        pin_memory=True,
        drop_last=True
    )
    
    return partial_training_dataloader, test_loader


def cifar10_partialize(images, given_partial_label_matrix, true_labels, images_diff, diff_targets):
    
    classify_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, 4, padding_mode='reflect'),
        transforms.ToTensor(),
        Cutout(n_holes=1, length=16),
        transforms.ToPILImage(),
        CIFAR10Policy(),
        transforms.ToTensor()
    ])
        
    weak_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor()
    ])

    strong_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        RandomAugment(3, 5),
        transforms.ToTensor()
    ])
    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    
    return Partialize_Dataset(images, given_partial_label_matrix, true_labels, images_diff, diff_targets,
                              classify_transform, weak_transform, strong_transform, normalize)


def load_cifar10_diff(args, diff_batch_size=8):
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    ])
    ori_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ]) 
    back_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ])
    original_train = dsets.CIFAR10(root='./data/CIFAR10', train=True, download=True, transform=None)
    ori_data, ori_labels = original_train.data, torch.Tensor(original_train.targets).long()
    
    if os.path.exists(f'plabels/partialY_{args.dataset}_{args.seed}.pt'):
        print('loading labels')
        partialY_matrix = torch.load(f'plabels/partialY_{args.dataset}_{args.seed}.pt', map_location='cpu')
    else:
        print('generating labels')
        mkdir_if_missing('plabels')
        ori_data = torch.Tensor(original_train.data)
        model = WideResNet(depth=28, num_classes=10, widen_factor=10, dropRate=0.3)
        model.load_state_dict(torch.load(os.path.expanduser('./partial_models/weights/cifar10.pt')))
        ori_data = ori_data.permute(0, 3, 1, 2)
        partialY_matrix = generate_instancedependent_candidate_labels(model, ori_data, ori_labels, 
                                                                      args.device, args.rate)
        torch.save(partialY_matrix, f'plabels/partialY_{args.dataset}_{args.seed}.pt')
    
    ones = torch.ones_like(partialY_matrix)
    ones[partialY_matrix == 0] = 0
    idx_fuge = ones.flatten().cumsum(dim=0) - 1
    
    mkdir_if_missing('res_diff')
    torch.save(idx_fuge.reshape_as(partialY_matrix).long(), f'res_diff/idx_fuge_{args.dataset}.pt',)
        
    ori_data = original_train.data
    temp = torch.zeros(partialY_matrix.shape)
    temp[torch.arange(partialY_matrix.shape[0]), ori_labels] = 1
    
    if torch.sum(partialY_matrix * temp) == partialY_matrix.shape[0]:
        print('data loading done !')

    
    partial_ori_dataset = Ori_Partialize(ori_data, partialY_matrix.float(), ori_labels.float(), 
                                         classes=original_train.classes, transform=ori_transform)
    
    partial_ori_dataloader = torch.utils.data.DataLoader(
        dataset=partial_ori_dataset, 
        batch_size=diff_batch_size, 
        shuffle=False, 
        num_workers=20,
        pin_memory=True
    )

    return partial_ori_dataloader, back_transform
