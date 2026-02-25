# CAD
This is the implementation of Mitigating Instance Entanglement in Instance-Dependent Partial Label Learning [CVPR 2026].

## Requirements: 
Python 3.8.13, 
numpy 1.22.3, 
torch 1.10.0,
torchvision 0.11.0,
diffusers 0.28.2.


## Training
### data preparing
To synthesize candidate labels, the annotation model weights should be downloaded from [this link](https://drive.google.com/drive/folders/1N6ZASfKQZkIu9t5l91ojzhUSVQZgRRgX?usp=drive_link) and place them into the `./partial_models/weights/` directory.

### demo
First, generate class-specific augmentations:
```sh
python -u csaugmentation.py --dataset cifar10
python -u csaugmentation.py --dataset cifar100
python -u csaugmentation.py --dataset pet37
python -u csaugmentation.py --dataset flower102
python -u csaugmentation.py --dataset fmnist
```

Then, train the model:
```sh
python -u main.py --dataset cifar10
python -u main.py --dataset cifar100
python -u main.py --dataset pet37
python -u main.py --dataset flower102
python -u main.py --dataset fmnist
```

## Reference
https://github.com/wu-dd/DIRK
