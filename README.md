# Implementation of "Understanding the Robustness of Randomized Feature Defense Against Query-Based Adversarial Attacks"
## Setup the environment
Install a conda environment with the requirements in ```requirements.txt```.
## Repository artifacts
- ```models```: contains models and implementation of randomized feature defense. You can add your own model and the corresponding defense in ```models/utils.py:add_defense``` function.
- ```attack```: implementation of attacks.
- ```configs```: configurations of each attack.

## Supported attacks
We are currently implementing the following attacks. Our code is modified from the original implementation of each method and [BlackBoxBench](https://github.com/SCLBD/BlackboxBench). 
- Score-based attacks:
  - [Square attack](https://link.springer.com/chapter/10.1007/978-3-030-58592-1_29)
  - [NES](https://proceedings.mlr.press/v80/ilyas18a.html)
  - [SimBA](https://proceedings.mlr.press/v97/guo19a.html)
  - [SignHunt](https://openreview.net/forum?id=SygW0TEFwH)
  - [ZO-signSGD](https://openreview.net/forum?id=BJe-DsC5Fm)
  - [Parsimonious attack](https://proceedings.mlr.press/v97/moon19a.html).
- Decision-based attacks:
  - [GeoDA](https://openaccess.thecvf.com/content_CVPR_2020/papers/Rahmati_GeoDA_A_Geometric_Framework_for_Black-Box_Adversarial_Attacks_CVPR_2020_paper.pdf)
  - [HopSkipJumpAttack](https://ieeexplore.ieee.org/abstract/document/9152788/)
  - [SignFlip](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123600273.pdf)
  - [RayS](https://dl.acm.org/doi/abs/10.1145/3394486.3403225)
  - [OPT attack](https://openreview.net/forum?id=rJlk6iRqKX)
  - [SignOPT](https://openreview.net/forum?id=SklTQCNtvS).
    
## Running the experiments
Run the following command to perform the defense

```python main.py --attack $attack_method --eps $lp_bound --defense random_noise --def_position hidden_feature --dataset $dataset_name --data-path $data_directory --n_ex $number_of_examples --n_iter $number_of_attack_iterations --noise_list $noise_scale```

For example:

```python main.py --attack square_linf --eps 0.05 --model resnet50 --defense random_noise --def_position hidden_feature --dataset imagenet --data-path ~\data --n_ex 1000 --n_iter 10000 --noise_list 0```

Please cite the paper, as below, when using this repository:
```
@inproceedings{nguyen2024understanding,
  title={Understanding the Robustness of Randomized Feature Defense Against Query-Based Adversarial Attacks},
  author={Nguyen, Quang H and Lao, Yingjie and Pham, Tung and Wong, Kok-Seng and Doan, Khoa D},
  booktitle={The Twelfth International Conference on Learning Representations},
  year={2024}
}
```
