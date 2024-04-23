from Model.Model import PINN,count_parameters
from Model.Compare_Models import MLP,CNN
import os
import argparse
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='XJTU', help='XJTU, HUST, MIT, TJU')
    parser.add_argument('--batch', type=int, default=10, help='1,2,3')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size')
    parser.add_argument('--normalization_method', type=str, default='z-score', help='min-max,z-score')

    # scheduler 相关
    parser.add_argument('--epochs', type=int, default=1, help='epoch')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--warmup_epochs', type=int, default=10, help='warmup epoch')
    parser.add_argument('--warmup_lr', type=float, default=5e-4, help='warmup lr')
    parser.add_argument('--final_lr', type=float, default=1e-4, help='final lr')
    parser.add_argument('--lr_F', type=float, default=1e-3, help='learning rate of F')
    parser.add_argument('--iter_per_epoch', type=int, default=1, help='iter per epoch')
    parser.add_argument('--F_layers_num', type=int, default=3, help='the layers num of F')
    parser.add_argument('--F_hidden_dim', type=int, default=60, help='the hidden dim of F')

    parser.add_argument('--alpha', type=float, default=1, help='loss = l_data + alpha * l_PDE + beta * l_physics')
    parser.add_argument('--beta', type=float, default=1, help='loss = l_data + alpha * l_PDE + beta * l_physics')

    parser.add_argument('--save_folder', type=str, default=None, help='save folder')
    parser.add_argument('--log_dir', type=str, default=None, help='log dir, if None, do not save')

    return parser.parse_args()

args = get_args()
pinn = PINN(args)
print('pinn:')
count_parameters(pinn.solution_u)

mlp = MLP()
cnn = CNN()
print('mlp:')
count_parameters(mlp)
print('cnn:')
count_parameters(cnn)


