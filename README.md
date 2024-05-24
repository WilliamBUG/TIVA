# Provably Invariance Learning without Domain Information 🚀

Welcome to the implementation of the **TIVA** (Through Independent Variables Automatically) for Provably Invariance Learning without Domain Information (ICML 2023)! 🎉

📝 **Note:** This implementation is the synthetic experiment in Section 4.

This work builds upon the **ZIN** project. Don't forget to check out the [ZIN repository](https://github.com/linyongver/ZIN_official/tree/main) for more details. 👀

## Requirements 🛠️

To run the code, make sure you have the following environment set up (generally the newest version would be fine):

- PyTorch (newest version should work)
- torch==1.14
- torchvision==0.15.0

## Parameters 🎛️

Here are some important parameters you need to consider:

- `l2_regularizer_weight`: L2 regularization weight.
- `lr`: learning rate.
- `steps`: training steps.
- `dataset`: which dataset to use.
- `penalty_anneal_iters`: the ERM producer before imposing the IRM penalty on the model. It also serves as the environmental inference procedure.

In addition, the following parameters are used for the synthetic simulation:

- `noise_ratio`: noise ratio when generating Y from invariant features.
- `cons_train`: the correlation between spurious features and Y in the training domains.
- `cons_test`: the correlation between spurious features and Y in the testing domains.
- `dim_inv`: dimension of invariant features.
- `dim_spu`: dimension of spurious features.
- `data_num_train`: number of training samples.
- `data_num_test`: number of testing samples.

## Run Synthetic Simulation with TIVA 🏃‍♂️

To run synthetic simulation with TIVA, you can set the simulation parameters and use other default values:

For example, if you want to run a synthetic simulation with the following settings:
- `noise_ratio` set to 0.1
- `cons_train` set to 0.999_0.7

You can execute the following command:
```
python synthetic_sample.py --noise_ratio 0.1 --cons_train 0.999_0.7
```
