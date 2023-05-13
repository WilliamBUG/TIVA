# Provably Invariance Learning without Domain Information
The implementation of synthetic experiment of learning invariance Through Independent Variables Automatically (TIVA) for Provably Invariance Learning without Domain Information
The full code is currently being organized and will be released in the future.

# Requirements
## Environment
The code is implemented by Pytorch, the newest version should work, our implementation uses the following environment:
- pytorch-transformers==1.2.0
- torch==1.3.1
- torchvision==0.4.2

# Parameters
These are some important parameters:
* `l2_regularizer_weight`: L2 regularization weight.
* `lr`: learning rate
* `steps`: training steps
* `dataset`: which dataset to use
* `penalty_anneal_iters`: the ERM proceducer befor imposing the IRM penalty on the model, this is also the environmental inference procedure.

The following parameters are used for the synthetic simulation:
* `noise_ratio`: noise ratio when generation Y from invariant features
* `cons_train`: the correlation  between spurious features and Y in the training domains
* `cons_test`: the correlation  between spurious features and Y in the testing domains
* `dim_inv`: dimension of invariant features
* `dim_spu`: dimension of spurious features
* `data_num_train`: number of training samples
* `data_num_test`: number of testing samples

# Run TIVA
You can set the simulation parameter and use other default parameter:

For exmaple, if you want to run synthetic simulation with setting p_s=(0.999, 0.7) and p_v=0.9:
```
python synthetic_sample.py --noise_ratio 0.1 --cons_train 0.999_0.7
```
