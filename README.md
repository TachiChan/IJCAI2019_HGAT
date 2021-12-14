# Heterogeneous Graph Attention Networks
This is our Pytorch implementation for the paper:

>Weijian Chen, Yulong Gu, Zhaochun Ren, Xiangnan He, Hongtao Xie, Tong Guo, Dawei Yin and Yongdong Zhang (2019). [Semi-supervised User Profiling with Heterogeneous Graph Attention Networks](https://www.ijcai.org/proceedings/2019/0293.pdf). In IJCAI'19, Macao, China, August 10-16, 2019.

This work was done during my internship at [JD Data Science Lab](http://datascience.jd.com/).

## Citation 
If you want to use our codes and dataset in your research, please cite:
```
@inproceedings{DBLP:conf/ijcai/ChenGRHXGYZ19,
  author    = {Weijian Chen and
               Yulong Gu and
               Zhaochun Ren and
               Xiangnan He and
               Hongtao Xie and
               Tong Guo and
               Dawei Yin and
               Yongdong Zhang},
  title     = {Semi-supervised User Profiling with Heterogeneous Graph Attention
               Networks},
  booktitle = {IJCAI},
  pages     = {2116--2122},
  year      = {2019}
}
```
## Environment Requirement
The code has been tested running under Python 3.6.9. The required packages are as follows:
* pytorch == 1.0.1
* numpy == 1.17.2
* scikit-learn == 0.21.3

## Example to Run the Codes
The instruction of commands has been clearly stated in the codes (see the parser function in train.py).
* HGAT, Gender Prediction
```
CUDA_VISIBLE_DEVICES=0 python train.py --pkl-dir 00 --data-dir data --model gat --hidden-units 16,16 --heads 8,8,1 --train-ratio 75 --valid-ratio 12.5 --instance-normalization --weight-decay 5e-4 --class-weight-balanced --patience 10 --epochs 100 --task gender --use-word-feature --lr 0.005 --dropout 0.6 --batch 64
```

* HGCN, Age Prediction
```
CUDA_VISIBLE_DEVICES=1 python train.py --pkl-dir 01 --data-dir data --model gcn --hidden-units 128,128 --train-ratio 75 --valid-ratio 12.5 --instance-normalization --weight-decay 5e-4 --class-weight-balanced --patience 10 --epochs 100 --task age --use-word-feature --lr 0.1 --dropout 0.2 --batch 32
```

## Dataset
The dataset used in our paper has been provided by JD Data Science Lab, which can be downloaded here: https://github.com/guyulongcs/IJCAI2019_HGAT.

## Related Links
* [DeepInf](https://github.com/xptree/DeepInf): The main reference for our code implementation.
* [RHGN](https://github.com/CRIPAC-DIG/RHGN): This work selects our work as the benchmark and provides another data set.
* [CatGCN](https://github.com/TachiChan/CatGCN): Our latest work involves user profiling.
