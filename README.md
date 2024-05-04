# Implementation of "Understanding the Robustness of Randomized Feature Defense Against Query-Based Adversarial Attacks"
## Setup the environment
Install a conda environment with the requirements in ```requirements.txt```.
## Repository artifacts
- ```models```: contains models and implementation of randomized feature defense. You can add your own model and the corresponding defense in ```models/utils.py:add_defense``` function.
- ```attack```: implementation of attacks.
- ```configs```: configurations of each attack.

## Running the experiments
Run the following command to perform the defense

```python main.py --attack $attack_method --eps $lp_bound --defense random_noise --def_position hidden_feature --dataset $dataset_name --data-path $data_directory --n_ex $number_of_examples --n_iter $number_of_attack_iterations --noise_list $noise_scale```

For example:

```python main.py --attack square_linf --eps 0.05 --model resnet50 --defense random_noise --def_position hidden_feature --dataset imagenet --data-path ~\data --n_ex 1000 --n_iter 10000 --noise_list 0```

Please cite the paper, as below, when using this repository:
```
@article{quang2024understanding,
  title={Understanding the Robustness of Randomized Feature Defense Against Query-Based Adversarial Attacks},
  author={Nguyen, Quang H and Lao, Yingjie and Pham, Tung and Wong, Kok-Seng and Doan, Khoa D},
  year={2024}
}
```
