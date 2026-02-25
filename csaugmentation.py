import os
import argparse
import torch
import random
from torch.backends import cudnn

from diffusers import StableDiffusionInstructPix2PixPipeline, DPMSolverMultistepScheduler
from dataset import get_loader_diff
from utils.util import labels_multi2single
import numpy as np

def main(args):
    cudnn.benchmark = True
    cudnn.deterministic = True
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    partial_ori_dataloader, back_transform = get_loader_diff(args)
    label2classes = np.array(partial_ori_dataloader.dataset.classes)

    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        "timbrooks/instruct-pix2pix", torch_dtype=torch.float16, safety_checker=None, use_safetensors=True
    )

    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    print(args.device)
    pipe = pipe.to(args.device)

    res_list = []

    with torch.no_grad():
        for i, (images, labels, true_labels, index) in enumerate(partial_ori_dataloader):
            single_labels_oh = labels_multi2single(labels)
            prompts_idx = np.argmax(single_labels_oh, axis=1)
            prompts = label2classes[prompts_idx]
            
            count_labels = labels.sum(dim=1).int()
            images = images.repeat_interleave(count_labels, dim=0)
            
            images = pipe(prompt=prompts.tolist(), image=images, guidance_scale=args.guidance_scale, num_inference_steps=args.num_inference_steps, output_type='pt').images
            images = [back_transform(img) for img in images]
            res_list.extend(images)
            
    results = torch.stack(res_list)
    torch.save(results, f'res_diff/diff_{args.dataset}.pt',)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # global set
    parser.add_argument('--dataset', default='cifar10',choices=['cifar10','cifar10n','kmnist','fmnist','fmnist3','mnist','cifar100', 'pet37',
                                                                'flower102', 'cub200'], type=str)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--diff_batch_size', default=8, type=int)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--gpu', default='0', type=str)
    parser.add_argument('--guidance_scale', type=int, default=5)
    parser.add_argument('--num_inference_steps', type=int, default=10)

    # PLL setting
    parser.add_argument('--rate', default=1.0, type=float)

    args = parser.parse_args()
    if args.dataset in ['fmnist', 'cifar10']:
        args.rate = 1.0
    elif args.dataset == 'cifar100':
        args.rate = 0.1
    elif args.dataset == 'pet37':
        args.rate = 0.1
    elif args.dataset == 'flower102':
        args.rate = 0.05

    args.device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:4096'
    
    main(args)