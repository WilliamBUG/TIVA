import argparse
import numpy as np
import torch
import pandas as pd
import os

import sys
from torch import nn, optim, autograd

from model import InferEnv
from model import FeatureSelector

from utils import eval_acc_class,mean_nll_class,mean_accuracy_class
from utils_z import MetaAcc, pretty_print_ly
from global_utils import args2header, save_args, save_cmd, LYCSVLogger
from utils_z import LOGITZ

parser = argparse.ArgumentParser(description='Colored MNIST')
parser.add_argument('--aux_num', type=int, default=7)
parser.add_argument('--batch_size', type=int, default=1024)# 1024
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--classes_num', type=int, default=2)
# 2
parser.add_argument('--dataset', type=str, default="logit_z", choices=["logit_z"])
parser.add_argument('--opt', type=str, default="adam", choices=["adam", "sgd"])
parser.add_argument('--l2_regularizer_weight', type=float,default=0.001)
parser.add_argument('--print_every', type=int,default=100)
parser.add_argument('--prior_sd_coef', type=float,default=50)
parser.add_argument('--dim_inv', type=int, default=5)
parser.add_argument('--variance_gamma', type=float, default=1.0)
parser.add_argument('--data_num_train', type=int, default=5000)
parser.add_argument('--data_num_test', type=int, default=5000)
parser.add_argument('--dim_spu', type=int, default=5)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--vib_lr', type=float, default=0.005)
# 0.005
parser.add_argument('--sfx', default="adhfoadf", type=str)
parser.add_argument('--env_type', default="linear", type=str, choices=["2_group", "cos", "linear"])
parser.add_argument('--irm_type', default="infer_adver", type=str, choices=["infer_adver"])
parser.add_argument('--n_restarts', type=int, default=3)
parser.add_argument('--save_last', default=0, type=int, choices=[0, 1])
parser.add_argument('--image_scale', type=int, default=64)
parser.add_argument('--hidden_dim', type=int, default=16)
parser.add_argument('--hidden_dim_infer', type=int, default=16)
parser.add_argument('--cons_train', type=str, default="0.999_0.7")
parser.add_argument('--cons_test', type=str, default="0.01_0.2_0.8_0.999")
parser.add_argument('--num_classes', type=int, default=2)
# 2
parser.add_argument('--z_class_num', type=int, default=7)
parser.add_argument('--noise_ratio', type=float, default=0.1)
parser.add_argument('--step_gamma', type=float, default=1.0)
parser.add_argument('--step_round', type=int, default=3)
parser.add_argument('--inner_steps', type=int, default=1)
parser.add_argument('--penalty_anneal_iters', type=int, default=4500)
parser.add_argument('--steps', type=int, default=5000)
parser.add_argument('--penalty_weight', type=float, default=10000)
parser.add_argument('--grayscale_model', type=int, default=0)
parser.add_argument('--stg_reg', type=float, default=0.001)
parser.add_argument('--ib_reg', type=float, default=0.005)
parser.add_argument('--radius', type=float, default=1)
parser.add_argument('--gpu', type=int, default=0)

flags = parser.parse_args()
print("batch_size is", flags.batch_size)

torch.manual_seed(flags.seed)
np.random.seed(flags.seed)


default_dict = {"inner_steps": 1, "step_round":3, "step_gamma":0.1, "hidden_dim":390,  "data_num_train":2000, "data_num_test":2000, "grayscale_model": False, "l2_regularizer_weight":0.001, "penalty_anneal_iters":200, "lr": 0.0004, "steps":10000, "envs_num_train":2, "dim_inv":2, "penalty_weight":10000, "cons_ratio": "0.9_0.8_0.1", "noise_ratio":0.1}
exclude_names = [
    "print_every",
    "variance_gamma",
    "cons_ratio",
    "cons_test",
    "envs_num_train",
    "envs_num_test",
    "env_type",
    "data_num_train",
    "data_num_test",
    "hidden_dim",
    "step_round",
    "inner_steps",
    "grayscale_model",
    "save_last"
]

os.environ["CUDA_VISIBLE_DEVICES"] = str(flags.gpu)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

logger_key= args2header(
    flags, default_dict=default_dict, exclude_names=exclude_names)
print('Flags:')

autograd.set_detect_anomaly(True)

flags.cons_ratio = "_".join([flags.cons_train, flags.cons_test])
flags.envs_num_train = len(flags.cons_train.split("_"))
flags.envs_num_test = len(flags.cons_test.split("_"))
assert flags.envs_num_test + flags.envs_num_train == len(flags.cons_ratio.split("_"))
irm_type = flags.irm_type

logger_key = logger_key.replace(':', '_')
logger_key = logger_key[:20]

logger_path = "logs/%s" % logger_key
if not os.path.exists(logger_path):
    os.makedirs(logger_path)
save_args(flags, logger_path)
save_cmd(sys.argv, logger_path)
mode='w'
csv_logger = LYCSVLogger(os.path.join(logger_path, 'res.csv'),  mode=mode)

for k,v in sorted(vars(flags).items()):
    print("\t{}: {}".format(k, v))


final_train_accs = []
final_test_accs = []

for restart in range(flags.n_restarts):
    print("Restart", restart)
    test_acc = {}
    test_acc['test'] = []
    test_acc["test_e0"] = []
    test_acc["test_e1"] = []
    test_acc["test_e2"] = []
    test_acc["test_e3"] = []

    dp = LOGITZ(flags, device)
    print(dp)

    test_batch_num = 1
    test_batch_fetcher = dp.fetch_test
    mean_nll = mean_nll_class
    mean_accuracy = mean_accuracy_class
    eval_acc = eval_acc_class
    scale = torch.tensor(1.).to(device).requires_grad_()

    z_dim_ex = flags.dim_spu
    stg = FeatureSelector(flags.dim_inv + flags.dim_spu + z_dim_ex).to(device)
    stg_mlp = nn.Linear(in_features=flags.dim_inv + flags.dim_spu + z_dim_ex, out_features=1).to(device)
    infer_env = InferEnv(flags, z_dim=flags.dim_inv + flags.dim_spu + z_dim_ex).to(device)

    mlp = nn.Linear(in_features=flags.dim_inv + flags.dim_spu + z_dim_ex, out_features=1).to(device)

    optimizer = optim.Adam(
        list(mlp.parameters()),
        lr=flags.lr)
    optimizer_infer_adver = optim.Adam(
        list(stg_mlp.parameters())+list(stg.parameters()),
        lr=flags.lr)
    optimizer_infer_env = optim.Adam(
        infer_env.parameters(),
        lr=flags.lr)


    meta_acc_test = MetaAcc(env=flags.envs_num_test, acc_measure=mean_accuracy, acc_type="test")
    pretty_print_ly(['step', 'train penalty'] + ["train_acc"] + meta_acc_test.acc_fields)
    for step in range(flags.steps):
        mlp.train()
        train_x, train_y, train_z, train_g, train_c, train_invnoise= dp.fetch_train()

        normed_z = (train_z.float() - train_z.float().mean(dim=0)) / train_z.float().std(dim=0)

        normed_z_input = torch.reshape(normed_z, (-1, z_dim_ex))
        all_input = torch.cat([train_x, normed_z_input], dim=1)
        for _ in range(1):
            train_y_target = train_y.squeeze().to(torch.int32)
            all_input_stg = stg(all_input)
            all_input_inverse = stg.inverse(all_input)
            learn_y = stg_mlp(all_input_stg)
            train_logits = scale * mlp(all_input)
            train_nll = nn.functional.binary_cross_entropy_with_logits(train_logits, train_y, reduction="none")
            infered_envs = infer_env(all_input_inverse)
            loss_fun = nn.MSELoss(reduction='none')

            stg_y_loss = nn.functional.binary_cross_entropy_with_logits(learn_y, train_y, reduction="none")
            infer_loss = stg_y_loss.mean() + flags.stg_reg * torch.mean(stg.regularizer((stg.mu + 0.5)/stg.sigma))

            env1_loss = (train_nll * infered_envs).mean()
            env2_loss = (train_nll * (1 - infered_envs)).mean()

            grad1 = autograd.grad(
                env1_loss,
                [scale],
                create_graph=True)[0]
            grad2 = autograd.grad(
                env2_loss,
                [scale],
                create_graph=True)[0]
            train_penalty = grad1 ** 2 + grad2 ** 2
            train_nll = train_nll.mean()

            if step < flags.penalty_anneal_iters:

                optimizer_infer_env.zero_grad()
                (-train_penalty).backward(retain_graph=True)
                optimizer_infer_env.step()

                optimizer_infer_adver.zero_grad()
                infer_loss.backward(retain_graph=True)
                optimizer_infer_adver.step()

                if step % 100 == 0:
                    print("SP agree", infered_envs[train_c == 0].mean().detach().cpu().numpy(),
                          "SP disagree", infered_envs[train_c == 1].mean().detach().cpu().numpy())

            if step == flags.penalty_anneal_iters:
                out_df = pd.DataFrame({
                    "infered_env": infered_envs.view(-1).detach().cpu().numpy(),
                    "train_g": train_g.view(-1).detach().cpu().numpy(),
                    "train_c": train_c.view(-1).detach().cpu().numpy(),
                    "train_y": train_y.view(-1).detach().cpu().numpy()})
                out_df.to_csv("outs/%s_inferz_%s.csv" % (flags.cons_train, step))


        train_acc, train_minacc, train_majacc = eval_acc(train_logits, train_y, train_c)
        weight_norm = torch.tensor(0.).to(device)
        for w in mlp.parameters():
            weight_norm += w.norm().pow(2)
        loss = train_nll.clone()
        loss += flags.l2_regularizer_weight * weight_norm

        penalty_weight = (flags.penalty_weight if step >= flags.penalty_anneal_iters else 0.0)
        penalty_loss = penalty_weight * train_penalty

        if penalty_weight > 1.0:
            total_loss_penalty = (loss + penalty_loss)/(1. + penalty_weight)
            optimizer.zero_grad()
            total_loss_penalty.backward()
        else:
            total_loss = loss
            optimizer.zero_grad()
            total_loss.backward()

        optimizer.step()

        if step % flags.print_every == 0:
            mlp.eval()
            meta_acc_test.clear()
            for ii in range(test_batch_num):
                test_x, test_y, test_z, test_g, test_c, test_invnoise = test_batch_fetcher()
                normed_z_test = (test_z.float() - train_z.float().mean(dim=0)) / train_z.float().std(dim=0)
                all_input_stg_test = torch.cat([test_x, normed_z_test], dim=1)
                test_inverse = stg.inverse(all_input_stg_test)
                test_logits = mlp(all_input_stg_test)
                meta_acc_test.process_batch(test_y, test_logits, test_g)

            meta_acc_test_res = meta_acc_test.meta_acc
            pretty_print_ly(
                [np.int32(step),
                train_penalty.detach().cpu().numpy(),
                train_acc.detach().cpu().numpy()] +
                [meta_acc_test_res[fd].detach().cpu().numpy() for fd in meta_acc_test.acc_fields])
            stats_dict = {
                "train_nll": train_nll.detach().cpu().numpy(),
                "train_acc": train_acc.detach().cpu().numpy(),
                "train_minacc": train_minacc.detach().cpu().numpy(),
                "train_majacc": train_majacc.detach().cpu().numpy(),
                "train_penalty": train_penalty.detach().cpu().numpy(),
            }
            test_acc['test'].append(meta_acc_test_res['test_acc'].detach().cpu().numpy())
            test_acc['test_e0'].append(meta_acc_test_res['test_e0'].detach().cpu().numpy())
            test_acc['test_e1'].append(meta_acc_test_res['test_e1'].detach().cpu().numpy())
            test_acc['test_e2'].append(meta_acc_test_res['test_e2'].detach().cpu().numpy())
            test_acc['test_e3'].append(meta_acc_test_res['test_e3'].detach().cpu().numpy())

            stats_dict.update(
                dict(zip(
                    meta_acc_test.acc_fields,
                    [meta_acc_test_res[fd].detach().cpu().numpy() for fd in meta_acc_test.acc_fields]
                ))
            )
            csv_logger.log(
                epoch=step,
                batch=step,
                stats_dict=stats_dict,
                restart=restart)
    final_train_accs.append(train_acc.detach().cpu().numpy())
    final_test_accs.append(max(test_acc['test'][flags.penalty_anneal_iters//flags.print_every:]))

print('Final train acc (mean/std across restarts so far):')
print(np.mean(final_train_accs), np.std(final_train_accs))
print('Final test acc (mean/std across restarts so far):')
print(np.mean(final_test_accs), np.std(final_test_accs))

torch.cuda.empty_cache()
csv_logger.close()