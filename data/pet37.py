import torch
import os
import torchvision.transforms as transforms
from torchvision import models
from augment.randaugment import RandomAugment
from utils.util import Ori_Partialize, Partialize_Dataset, generate_instancedependent_candidate_labels, mkdir_if_missing
import torch.nn as nn
from augment.autoaugment_extra import ImageNetPolicy
from data.dataset_pet import OxfordIIITPet


def load_pet37(args):
    
    test_transform = transforms.Compose([
        transforms.Resize(int(224 / 0.875)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),std=(0.229, 0.224, 0.225))
    ])
    
    test_dataset = OxfordIIITPet(root='./data/PET37', split='test', transform=test_transform)
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=args.batch_size, \
        shuffle=False, num_workers=4, pin_memory=False
    )
    
    original_train = OxfordIIITPet(root='./data/PET37', split='trainval', transform=test_transform, download=False)
    original_full_loader = torch.utils.data.DataLoader(dataset=original_train, 
                                                       batch_size=len(original_train),
                                                       shuffle=False, 
                                                       num_workers=20)
    ori_data, ori_labels = next(iter(original_full_loader))
    ori_labels = ori_labels.long()

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
    if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()
    if os.path.exists(f'plabels/partialY_{args.dataset}_{args.seed}.pt'):
        print(f'loading {args.dataset} labels')
        partialY_matrix = torch.load(f'plabels/partialY_{args.dataset}_{args.seed}.pt', map_location='cpu')
    else:
        print('generating labels')
        mkdir_if_missing('plabels')
        model = models.wide_resnet50_2()
        model.fc = nn.Linear(model.fc.in_features, max(ori_labels) + 1)
        model = torch.nn.DataParallel(model)
        model.load_state_dict(torch.load(os.path.expanduser('./partial_models/weights/pet37.pt')))
        partialY_matrix = generate_instancedependent_candidate_labels(model.module, ori_data, ori_labels, args.rate, 512)
        torch.save(partialY_matrix, f'plabels/partialY_{args.dataset}_{args.seed}.pt')
        
    temp = torch.zeros(partialY_matrix.shape)
    temp[torch.arange(partialY_matrix.shape[0]), ori_labels] = 1

    if torch.sum(partialY_matrix * temp) == partialY_matrix.shape[0]:
        print('data loading done !')
        
    images_diff = torch.load(f'res_diff/diff_{args.dataset}.pt', map_location='cpu')
    idx_diff = torch.load(f'res_diff/idx_fuge_{args.dataset}.pt', map_location='cpu').long()
    idx_diff = idx_diff[partialY_matrix == 1]
    images_diff = images_diff[idx_diff]
    diff_targets = torch.nonzero(partialY_matrix, as_tuple=False)[:, 1]
    
    partial_training_dataset = pet37_partialize(original_train._images, partialY_matrix.float(), 
                                                original_train._labels, images_diff, diff_targets)

    partial_training_dataloader = torch.utils.data.DataLoader(
        dataset=partial_training_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=8,
        prefetch_factor=16,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True
    )

    return partial_training_dataloader, test_loader


def pet37_partialize(images, given_partial_label_matrix, true_labels, images_diff, diff_targets):
    
    classify_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        ImageNetPolicy(),
        transforms.ToTensor()])
        
    weak_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor()])

    strong_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        RandomAugment(3, 5),
        transforms.ToTensor()])
    
    normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    
    return Partialize_Dataset(images, given_partial_label_matrix, true_labels, images_diff, diff_targets,
                              classify_transform, weak_transform, strong_transform, normalize, image_path=True)


def load_pet37_diff(args, diff_batch_size=8):
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:3950"
    test_transform = transforms.Compose([
        transforms.Resize(int(224 / 0.875)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),std=(0.229, 0.224, 0.225))
    ])
    
    back_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    original_train = OxfordIIITPet(root='./data/PET37', split='trainval', transform=test_transform, download=True)
    original_full_loader = torch.utils.data.DataLoader(dataset=original_train, batch_size=len(original_train),shuffle=False, num_workers=20)
    ori_data, ori_labels = next(iter(original_full_loader))
    ori_labels = ori_labels.long()

    if os.path.exists(f'plabels/partialY_{args.dataset}_{args.seed}.pt'):
        print(f'loading {args.dataset} labels')
        partialY_matrix = torch.load(f'plabels/partialY_{args.dataset}_{args.seed}.pt', map_location='cpu')
    else:
        print('generating labels')
        mkdir_if_missing('plabels')
        model = models.wide_resnet50_2()
        model.fc = nn.Linear(model.fc.in_features, max(ori_labels) + 1)
        model = torch.nn.DataParallel(model)
        model.load_state_dict(torch.load(os.path.expanduser('./partial_models/weights/pet37.pt')))
        partialY_matrix = generate_instancedependent_candidate_labels(model.module, ori_data, ori_labels, 
                                                                      args.device, args.rate, 512)
        torch.save(partialY_matrix, f'plabels/partialY_{args.dataset}_{args.seed}.pt')
         
    ones = torch.ones_like(partialY_matrix)
    ones[partialY_matrix == 0] = 0
    idx_fuge = ones.flatten().cumsum(dim=0) - 1
    mkdir_if_missing('res_diff')
    torch.save(idx_fuge.reshape_as(partialY_matrix).long(), f'res_diff/idx_fuge_{args.dataset}.pt',)

    temp = torch.zeros(partialY_matrix.shape)
    temp[torch.arange(partialY_matrix.shape[0]), original_train._labels] = 1

    if torch.sum(partialY_matrix * temp) == partialY_matrix.shape[0]:
        print('data loading done !')
    
    ori_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    
    partial_ori_dataset = Ori_Partialize(ori_data, partialY_matrix.float(), 
                                         original_train._labels, 
                                         classes=original_train.classes, transform=ori_transform)
    
    partial_ori_dataloader = torch.utils.data.DataLoader(
        dataset=partial_ori_dataset, 
        batch_size=diff_batch_size, 
        shuffle=False, 
        num_workers=20,
        pin_memory=True
    )

    return partial_ori_dataloader, back_transform
