import torch
import os
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from augment.cutout import Cutout
from partial_models.mlp import mlp_partialize
from utils.util import Ori_Partialize, Partialize_Dataset, generate_instancedependent_candidate_labels, mkdir_if_missing


def load_fmnist(args):
    original_train = dsets.FashionMNIST(root='./data/FMNIST', train=True, download=True, transform=transforms.ToTensor())
    ori_data, ori_labels = original_train.data, original_train.targets.long()
    
    if os.path.exists(f'plabels/partialY_{args.dataset}_{args.seed}.pt'):
        print('loading labels')
        partialY_matrix = torch.load(f'plabels/partialY_{args.dataset}_{args.seed}.pt', map_location='cpu')
    else:
        ori_data = ori_data.view((ori_data.shape[0], -1)).float()
        num_features = 28 * 28
        partialize_net = mlp_partialize(n_inputs=num_features, n_outputs=max(ori_labels)+1)
        partialize_net.load_state_dict(torch.load(os.path.expanduser('./partial_models/weights/fmnist.pt'), map_location='cpu'))
        partialY_matrix = generate_instancedependent_candidate_labels(partialize_net, ori_data, ori_labels, args.rate)
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
    
    partial_training_dataset = fmnist_partialize(ori_data, partialY_matrix.float(), ori_labels.float(), 
                                                 images_diff, diff_targets)
    
    partial_training_dataloader = torch.utils.data.DataLoader(
        dataset=partial_training_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=6,
        pin_memory=True,
        drop_last=True
    )
    
    test_transform = transforms.Compose([
        transforms.Grayscale(3),
        transforms.ToTensor(),
        partial_training_dataset.normalize
    ])
    
    test_dataset = dsets.FashionMNIST(root='./data/FMNIST', train=False, download=True, transform=test_transform)
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=args.batch_size, \
        shuffle=False, num_workers=6, pin_memory=False
    )
    
    return partial_training_dataloader, test_loader


def fmnist_partialize(images, given_partial_label_matrix, true_labels, images_diff, diff_targets):
    
    classify_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(28, 4, padding_mode='reflect'),
        transforms.ToTensor(),
        Cutout(n_holes=1, length=16),
        transforms.ToPILImage(),
        transforms.Grayscale(3),
        transforms.ToTensor()
    ])
        
    weak_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomResizedCrop(size=28, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.Grayscale(3),
        transforms.ToTensor()
    ])

    strong_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomResizedCrop(size=28, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.Grayscale(3),
        transforms.ToTensor()
    ])
    
    normalize = transforms.Normalize((0.1307, 0.1307, 0.1307), (0.3081, 0.3081, 0.3081))
    
    return Partialize_Dataset(images, given_partial_label_matrix, true_labels, images_diff, diff_targets,
                              classify_transform, weak_transform, strong_transform, normalize)


def load_fmnist_diff(args, diff_batch_size=8):

    test_transform = transforms.Compose([
        transforms.Grayscale(3),
        transforms.ToTensor(),
        transforms.Normalize((0.1307, 0.1307, 0.1307), (0.3081, 0.3081, 0.3081))
    ])
    
    
    back_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
    ])
    
    original_train = dsets.FashionMNIST(root='./data/FMNIST', train=True, download=True, transform=transforms.ToTensor())
    ori_data, ori_labels = original_train.data, original_train.targets.long()
      
    if os.path.exists(f'plabels/partialY_{args.dataset}_{args.seed}.pt'):
        print('loading labels')
        partialY_matrix = torch.load(f'plabels/partialY_{args.dataset}_{args.seed}.pt', map_location='cpu')
    else:
        ori_data = ori_data.view((ori_data.shape[0], -1)).float()
        num_features = 28 * 28
        partialize_net = mlp_partialize(n_inputs=num_features, n_outputs=max(ori_labels)+1)
        partialize_net.load_state_dict(torch.load(os.path.expanduser('./partial_models/weights/fmnist.pt'), map_location='cpu'))
        partialY_matrix = generate_instancedependent_candidate_labels(partialize_net, ori_data, ori_labels, 
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
    
    ori_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.Grayscale(3),
        transforms.ToTensor(),
    ])
    partial_ori_dataset = Ori_Partialize(ori_data, partialY_matrix.float(), 
                                         ori_labels.float(), 
                                         classes=original_train.classes, transform=ori_transform)
    
    partial_ori_dataloader = torch.utils.data.DataLoader(
        dataset=partial_ori_dataset, 
        batch_size=diff_batch_size, 
        shuffle=False, 
        num_workers=20,
        pin_memory=True
    )
    
    return partial_ori_dataloader, back_transform

