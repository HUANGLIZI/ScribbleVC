# ScribbleVC (ACM MM 2023)

This repository is the official implementation of the paper ScribbleVC: Scribble-supervised Medical Image Segmentation with Vision-Class Embedding (ACM MM 2023). [Paper](https://dl.acm.org/doi/10.1145/3581783.3612056), [Arxiv](https://arxiv.org/abs/2307.16226), [ResearchGate](https://www.researchgate.net/publication/372761587_ScribbleVC_Scribble-supervised_Medical_Image_Segmentation_with_Vision-Class_Embedding)

![image](https://github.com/HUANGLIZI/ScribbleVC/blob/main/ScribbleVC.png)

The ScribbleVC has been compatible with [WSL4MIS](https://github.com/HiLab-git/WSL4MIS) Codebase, feel free to use it.

## Datasets

### ACDC
1. The ACDC dataset with mask annotations can be downloaded from [ACDC](https://www.creatis.insa-lyon.fr/Challenge/acdc/).
2. The scribble annotations of ACDC have been released in [ACDC scribbles](https://vios-s.github.io/multiscale-adversarial-attention-gates/data). 
3. The pre-processed ACDC data used for training could be directly downloaded from [ACDC_dataset](https://github.com/HiLab-git/WSL4MIS/tree/main/data/ACDC).

### MSCMR
1. The MSCMR dataset with mask annotations can be downloaded from [MSCMRseg](https://zmiclab.github.io/zxh/0/mscmrseg19/data.html). 
2. The scribble annotations of MSCMRseg have been released in [MSCMR_scribbles](https://github.com/BWGZK/CycleMix/tree/main/MSCMR_scribbles). 
3. The scribble-annotated MSCMR dataset used for training could be directly downloaded from [MSCMR_dataset](https://github.com/BWGZK/CycleMix/tree/main/MSCMR_dataset).

**The slice classfication files have been available.**

## Requirements

Some important required packages include:
* Python 3.8
* CUDA 11.7
* [Pytorch](https://pytorch.org) 1.13.1
* torchvision 0.14.1
* Some basic python packages such as Numpy, Scikit-image, SimpleITK, Scipy.

Follow official guidance to install [Pytorch](https://pytorch.org).

## Training

To train the model, run this command:

```train
python train_ACDC.py --root_path <dataset_path> --exp <path_to_save_model> --bilinear --linear_layer --max_epoches 100 --pretrain_weights cnnTransformer.pth 
```

## Evaluation

To evaluate the model, run this command:

```eval
python test_ACDC.py --bilinear --linear_layer --exp <path_to_save_model> --save_prediction
```

# Citation

```bash
@inproceedings{li2023scribblevc,
  title={ScribbleVC: Scribble-supervised Medical Image Segmentation with Vision-Class Embedding},
  author={Li, Zihan and Zheng, Yuan and Luo, Xiangde and Shan, Dandan and Hong, Qingqi},
  booktitle={Proceedings of the 31st ACM International Conference on Multimedia},
  year={2023}
}
```
