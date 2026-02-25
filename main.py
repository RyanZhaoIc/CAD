import os
import argparse
import torch
import random
import logging
from torch.backends import cudnn
import torch.nn.functional as F

from dataset import *
from model import *
from resnet import *
from utils.utils_loss import *
from utils.util import *
    
def main(args, logging):
    cudnn.benchmark = True
    cudnn.deterministic = True
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # load dataloader
    logging.info("=> creating loader '{}'".format(args.dataset))
    train_loader, test_loader, num_class = get_loader(args)
    args.num_class=num_class
    logging.info('=> Average number of partial labels: {}'.format(
        train_loader.dataset.given_partial_label_matrix.sum() / len(train_loader.dataset)))

    # load model
    logging.info("=> Creating model '{}'".format(args.arch))
    estimator = Estimator(args,SupConResNet).cuda()
    classifier = Classifier(args, SupConResNet).cuda()
    optimizer = torch.optim.SGD(classifier.parameters(), args.lr, momentum=0.9, weight_decay=args.wd)

    # set contrastive loss function
    loss_con_fn = WeightedConLoss(temperature=args.feat_temp,dist_temprature=args.dist_temp)

    logging.info('=> Start Training')
    best_acc1 = 0
    
    for epoch in range(args.epochs):
        adjust_learning_rate(args, optimizer, epoch)
        start_upd_prot = epoch >= args.prot_start
        teach_loss, cont_loss = train(train_loader,estimator, classifier, optimizer,loss_con_fn,epoch,args,start_upd_prot)
        logging.info("[Training-Epoch {}]:teach_loss:{}\tcont_loss:{}".format(epoch,teach_loss,cont_loss))
        val_acc = test(args, epoch, test_loader, classifier)
        best_acc1 = max(best_acc1, val_acc)
        logging.info("[Testing-Epoch {}]:val_acc: {}".format(epoch, val_acc))
    
    torch.save(classifier.cpu().state_dict(), f'{args.output_dir}/{args.dataset}-{args.arch}-seed{args.seed}.pth')


def train(train_loader, estimator, classifier, optimizer, loss_con_fn, epoch, args, start_upd_prot=False):
    teach_losses=AverageMeter('Teaching_Loss',':.2f')
    con_losses=AverageMeter('Con_Loss',':.2f')

    # switch to train mode
    classifier.train()
    estimator.train()

    for i, (img_cls, img_w, img_diff_w, img_diff_s, img_diff_cls, partY, target, index, diff_idx) in enumerate(train_loader):
         
        img_diff_w = img_diff_w.flatten(0, 1)
        img_diff_s = img_diff_s.flatten(0, 1)
        img_diff_cls = img_diff_cls.flatten(0, 1)
        diff_idx = diff_idx.flatten()
        img_diff_w, img_diff_s, img_diff_cls = img_diff_w[diff_idx], img_diff_s[diff_idx], img_diff_cls[diff_idx]
        assert partY.sum() == img_diff_w.shape[0] == img_diff_s.shape[0] == img_diff_cls.shape[0]
        diff_target_w = torch.nonzero(partY, as_tuple=False)[:, 1]
        
        img_diff_w, img_diff_s, img_diff_cls = img_diff_w.cuda(non_blocking=True), img_diff_s.cuda(non_blocking=True), img_diff_cls.cuda(non_blocking=True)
        img_w, img_cls = img_w.cuda(non_blocking=True), img_cls.cuda(non_blocking=True)
        partY, index = partY.cuda(non_blocking=True), index.cuda(non_blocking=True)
        diff_target_w = diff_target_w.cuda(non_blocking=True)

        features, dists, diff_targets, output_diff, omega = estimator(img_cls, img_diff_w, img_diff_cls, partY, diff_target_w)
        output_s, feat_s = classifier(img_diff_s, img_cls)

        features_cont = torch.cat((feat_s, features), dim=0)
        partY_cont = torch.cat((diff_target_w, diff_targets), dim=0)
        dist_cont = torch.cat((output_diff, dists), dim=0)

        batch_size = feat_s.shape[0]

        if start_upd_prot:
            mask = torch.eq(partY_cont.view(-1, 1)[:batch_size], partY_cont.view(-1, 1).T).float()
        else:
            mask = None

        # L_{cons}
        if args.weight != 0:
            loss_con = loss_con_fn(features=features_cont, dist=dist_cont, partY=partY_cont,
                                     mask=mask, epoch=epoch, args=args, batch_size=batch_size)
        else:
            loss_con = torch.tensor(0.0).cuda()

        # L_{cls}
        probs = F.softmax(output_s, dim=1)
        loss_cls = Sym_CE_loss(probs, omega, partY)
        
        # total loss
        loss = loss_cls + args.weight * loss_con

        teach_losses.update(loss_cls.item(),partY.size(0))
        con_losses.update(loss_con.item(),partY.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        momentum_model(estimator, classifier, args.m)

    return teach_losses.avg, con_losses.avg


def momentum_model(model_tea, model_stu, momentum=0.5):
    for param_tea, param_stu in zip(model_tea.parameters(), model_stu.parameters()):
        param_tea.data = param_tea.data * momentum + param_stu.data * (1 - momentum)


def adjust_learning_rate(args, optimizer, epoch):
    import math
    lr = args.lr
    eta_min=lr * (args.lr_decay_rate**3)
    lr=eta_min+(lr-eta_min)*(
        1+math.cos(math.pi*epoch/args.epochs))/2

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    logging.info('LR: {}'.format(lr))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='cifar10',choices=['cifar10', 'fmnist', 'cifar100', 'pet37', 'flower102'], type=str)
    parser.add_argument('--arch', default='resnet18', type=str)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epochs', default=500, type=int)
    parser.add_argument('--num_workers', default=12, type=int)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--lr_decay_rate', type=float, default=0.1)
    parser.add_argument('--lr', default=1e-2, type=float)
    parser.add_argument('--wd', default=1e-3, type=float)
    parser.add_argument('--m', type=float, default=0.99)
    parser.add_argument('--rate', default=1.0, type=float)
    parser.add_argument('--queue',type=int,default=4096)
    # parser.add_argument('--weight',type=float, default=1.0)
    # parser.add_argument('--dist_temp',type=float,default=0.4)
    # parser.add_argument('--feat_temp',type=float,default=0.07)
    
    parser.add_argument('--weight',type=float, default=1.5)
    parser.add_argument('--dist_temp',type=float,default=0.1)
    parser.add_argument('--feat_temp',type=float,default=0.12)
    
    parser.add_argument('--prot_start',type=int,default=1)
    parser.add_argument('--gpu', default='0', type=str)
    parser.add_argument('--output_dir', default='results', type=str)
    args = parser.parse_args()
    
    if args.dataset == 'fmnist':
        args.arch = 'resnet18'
        args.rate = 1.0
    elif args.dataset == 'cifar10':
        args.arch = 'resnet34'
        args.rate = 1.0
    elif args.dataset == 'cifar100':
        args.arch = 'resnet34'
        args.rate = 0.1
    elif args.dataset == 'pet37':
        args.arch = 'resnet34'
        args.rate = 0.1
        args.batch_size = 16
    elif args.dataset == 'flower102':
        args.arch = 'resnet34'
        args.rate = 0.05
        args.batch_size = 16
    
    mkdir_if_missing(args.output_dir)
    logging.basicConfig(format='[%(asctime)s] - %(message)s',
                        datefmt='%Y/%m/%d %H:%M:%S',
                        level=logging.INFO,
                        filename=f'{args.output_dir}/{args.dataset}-{args.arch}-seed{args.seed}.log',
                        filemode='w'
                        )
    torch.set_printoptions(linewidth=2000)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    logging.info(args.__dict__)
    
    main(args, logging)
