
from dataloader.dataloader import XJTUdata,MITdata,HUSTdata,TJUdata
from Model.Model import PINN
import argparse
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def get_args():
    parser = argparse.ArgumentParser('Hyper Parameters for MIT dataset')
    parser.add_argument('--data', type=str, default='MIT', help='XJTU, HUST, MIT, TJU')
    parser.add_argument('--batch_size', type=int, default=512, help='batch size')
    parser.add_argument('--normalization_method', type=str, default='min-max', help='min-max,z-score')

    # scheduler related
    parser.add_argument('--epochs', type=int, default=200, help='epoch')
    parser.add_argument('--early_stop', type=int, default=10, help='early stop')
    parser.add_argument('--warmup_epochs', type=int, default=30, help='warmup epoch')
    parser.add_argument('--warmup_lr', type=float, default=2e-3, help='warmup lr')
    parser.add_argument('--lr', type=float, default=1e-2, help='learning rate')
    parser.add_argument('--final_lr', type=float, default=2e-4, help='final lr')
    parser.add_argument('--lr_F', type=float, default=1e-3, help='learning rate of F')

    # model related
    parser.add_argument('--u_layers_num', type=int, default=3, help='the layers num of u')
    parser.add_argument('--u_hidden_dim', type=int, default=60, help='the hidden dim of u')
    parser.add_argument('--F_layers_num', type=int, default=3, help='the layers num of F')
    parser.add_argument('--F_hidden_dim', type=int, default=60, help='the hidden dim of F')

    # loss related
    parser.add_argument('--alpha', type=float, default=1, help='loss = l_data + alpha * l_PDE + beta * l_physics')
    parser.add_argument('--beta', type=float, default=0.02, help='loss = l_data + alpha * l_PDE + beta * l_physics')

    parser.add_argument('--log_dir', type=str, default='logging.txt', help='log dir, if None, do not save')
    parser.add_argument('--save_folder', type=str, default='results/MIT results', help='save folder')

    args = parser.parse_args()

    return args

def load_MIT_data(args,small_sample=None):
    root = 'data/MIT data'
    train_list = []
    test_list = []
    for batch in ['2017-05-12','2017-06-30','2018-04-12']:
        batch_root = os.path.join(root,batch)
        files = os.listdir(batch_root)
        for f in files:
            id = int(f.split('-')[-1].split('.')[0])
            if id % 5 == 0:
                test_list.append(os.path.join(batch_root,f))
            else:
                train_list.append(os.path.join(batch_root,f))
    if small_sample is not None:
        train_list = train_list[:small_sample]
    data = MITdata(root=root,args=args)
    trainloader = data.read_all(specific_path_list=train_list)
    testloader = data.read_all(specific_path_list=test_list)
    dataloader = {'train':trainloader['train_2'],'valid':trainloader['valid_2'],'test':testloader['test_3']}

    return dataloader


def main():
    args = get_args()
    for e in range(10):
        setattr(args, 'save_folder', f'revise_results/MIT results/Experiment{e + 1}')
        if not os.path.exists(args.save_folder):
            os.makedirs(args.save_folder)

        dataloader = load_MIT_data(args)
        pinn = PINN(args)
        pinn.Train(trainloader=dataloader['train'],validloader=dataloader['valid'],testloader=dataloader['test'])

def small_sample():
    args = get_args()
    num_battery = 2
    for e in range(10):
        setattr(args, 'save_folder',
                f'results/MIT results (small sample {num_battery})/Experiment{e + 1}')
        setattr(args, 'batch_size', 128)
        if not os.path.exists(args.save_folder):
            os.makedirs(args.save_folder)
        dataloader = load_MIT_data(args, small_sample=num_battery)
        pinn = PINN(args)
        pinn.Train(trainloader=dataloader['train'], validloader=dataloader['valid'], testloader=dataloader['test'])


if __name__ == '__main__':
    pass
    # main()
