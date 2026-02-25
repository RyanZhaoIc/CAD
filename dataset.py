from data.fmnist import load_fmnist, load_fmnist_diff
from data.cifar10 import load_cifar10, load_cifar10_diff
from data.cifar100 import load_cifar100, load_cifar100_diff
from data.flower102 import load_flower102, load_flower102_diff
from data.pet37 import load_pet37, load_pet37_diff

def get_loader(args):
    if args.dataset == 'fmnist':
        num_classes = 10
        train_loader, test_loader = load_fmnist(args)
        
    elif args.dataset == "cifar10":
        num_classes = 10
        train_loader, test_loader = load_cifar10(args)
        
    elif args.dataset == 'cifar100':
        num_classes = 100
        train_loader, test_loader = load_cifar100(args)

    elif args.dataset == 'flower102':
        num_classes = 102
        train_loader, test_loader = load_flower102(args)

    elif args.dataset == 'pet37':
        num_classes = 37
        train_loader, test_loader = load_pet37(args)

    return train_loader, test_loader, num_classes


def get_loader_diff(args):
    if args.dataset == 'fmnist':
        num_classes = 10
        train_loader, back_transform = load_fmnist_diff(args, diff_batch_size=args.diff_batch_size)
        
    elif args.dataset == "cifar10":
        num_classes = 10
        train_loader, back_transform = load_cifar10_diff(args, diff_batch_size=args.diff_batch_size)
        
    elif args.dataset == 'cifar100':
        num_classes = 100
        train_loader, back_transform = load_cifar100_diff(args, diff_batch_size=args.diff_batch_size)

    elif args.dataset == 'flower102':
        num_classes = 102
        train_loader, back_transform = load_flower102_diff(args, diff_batch_size=args.diff_batch_size)

    elif args.dataset == 'pet37':
        num_classes = 37
        train_loader, back_transform = load_pet37_diff(args, diff_batch_size=args.diff_batch_size)

    return train_loader, back_transform
