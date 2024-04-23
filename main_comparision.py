import torch
import torch.nn as nn
import numpy as np
import os
from utils.util import AverageMeter,get_logger
from Model.Compare_Models import MLP,CNN
from Model.Model import LR_Scheduler
from dataloader.dataloader import XJTUdata,HUSTdata,MITdata,TJUdata
import argparse

class Trainer():
    def __init__(self,model,train_loader,valid_loader,test_loader,args):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.args = args
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader

        self.save_dir = args.save_folder
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        self.epochs = args.epochs
        self.logger = get_logger(os.path.join(args.save_folder,args.log_dir))


        self.loss_meter = AverageMeter()
        self.loss_func = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(),lr=args.warmup_lr)
        self.scheduler = LR_Scheduler(optimizer=self.optimizer,
                                      warmup_epochs=args.warmup_epochs,
                                      warmup_lr=args.warmup_lr,
                                      num_epochs=args.epochs,
                                      base_lr=args.lr,
                                      final_lr=args.final_lr)

    def clear_logger(self):
        self.logger.removeHandler(self.logger.handlers[0])
        self.logger.handlers.clear()

    def train_one_epoch(self,epoch):
        self.model.train()
        self.loss_meter.reset()
        for (x1,_,y1,_) in self.train_loader:
            x1 = x1.to(self.device)
            y1 = y1.to(self.device)


            y_pred = self.model(x1)
            loss = self.loss_func(y_pred,y1)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.loss_meter.update(loss.item())
        info = '[Train] epoch:{:0>3d}, data loss:{:.6f}'.format(epoch,self.loss_meter.avg)
        self.logger.info(info)
        return self.loss_meter.avg

    def valid(self,epoch):
        self.model.eval()
        self.loss_meter.reset()
        with torch.no_grad():
            for (x1,_,y1,_) in self.valid_loader:
                x1 = x1.to(self.device)
                y1 = y1.to(self.device)

                y_pred = self.model(x1)
                loss = self.loss_func(y_pred,y1)
                self.loss_meter.update(loss.item())
        info = '[Valid] epoch:{:0>3d}, data loss:{:.6f}'.format(epoch,self.loss_meter.avg)
        self.logger.info(info)
        return self.loss_meter.avg

    def test(self):
        self.model.eval()
        self.loss_meter.reset()
        true_label = []
        pred_label = []
        with torch.no_grad():
            for (x1,_,y1,_) in self.test_loader:
                x1 = x1.to(self.device)
                y_pred = self.model(x1)

                true_label.append(y1.cpu().detach().numpy())
                pred_label.append(y_pred.cpu().detach().numpy())
        true_label = np.concatenate(true_label,axis=0)
        pred_label = np.concatenate(pred_label,axis=0)
        if self.save_dir is not None:
            np.save(os.path.join(self.save_dir,'true_label.npy'),true_label)
            np.save(os.path.join(self.save_dir,'pred_label.npy'),pred_label)
        return true_label,pred_label

    def train(self):
        min_loss = 100
        early_stop = 0
        for epoch in range(1,self.epochs+1):
            early_stop += 1
            train_loss = self.train_one_epoch(epoch)
            current_lr = self.scheduler.step()
            valid_loss = self.valid(epoch)
            if valid_loss < min_loss and self.test_loader is not None:
                min_loss = valid_loss
                true_label,pred_label = self.test()
                early_stop = 0
            if early_stop > 10:
                break
        self.clear_logger()


def load_model(args):
    if args.model == 'MLP':
        model = MLP()
    elif args.model == 'CNN':
        model = CNN()
    return model

def load_XJTU_data(args,small_sample=None):
    root = 'data/XJTU data'
    data = XJTUdata(root=root, args=args)
    train_list = []
    test_list = []
    files = os.listdir(root)
    for file in files:
        if args.xjtu_batch in file:
            if '4' in file or '8' in file:
                test_list.append(os.path.join(root, file))
            else:
                train_list.append(os.path.join(root, file))
    if small_sample is not None:
        train_list = train_list[:small_sample]

    train_loader = data.read_all(specific_path_list=train_list)
    test_loader = data.read_all(specific_path_list=test_list)
    dataloader = {'train': train_loader['train_2'],
                  'valid': train_loader['valid_2'],
                  'test': test_loader['test_3']}
    return dataloader


def get_args():
    parser = argparse.ArgumentParser('The parameters of Comparision methods')
    parser.add_argument('--model',type=str,default='CNN',choices=['MLP','CNN'])
    parser.add_argument('--dataset',type=str,default='XJTU',choices=['XJTU','HUST','MIT','TJU'])
    parser.add_argument('--normalization_method',type=str, default='min-max', help='min-max,z-score')

    # XJTU data
    parser.add_argument('--xjtu_batch',type=str,default='2C',choices=['2C','3C','R2.5','R3','RW','satellite'])

    # TJU data
    parser.add_argument('--in_same_batch',type=bool,default=True)
    parser.add_argument('--tju_batch',type=int,default=0,choices=[0,1,2])
    parser.add_argument('--tju_train_batch',type=int,default=-1, choices=[-1,0,1,2])
    parser.add_argument('--tju_test_batch',type=int,default=-1, choices=[-1,0,1,2])

    # scheduler related
    parser.add_argument('--epochs', type=int, default=200, help='epoch')
    parser.add_argument('--early_stop', type=int, default=20, help='early stop')
    parser.add_argument('--warmup_epochs', type=int, default=30, help='warmup epoch')
    parser.add_argument('--warmup_lr', type=float, default=2e-3, help='warmup lr')
    parser.add_argument('--lr', type=float, default=1e-2, help='learning rate')
    parser.add_argument('--final_lr', type=float, default=2e-4, help='final lr')
    parser.add_argument('--lr_F', type=float, default=5e-4, help='lr of F')


    parser.add_argument('--save_folder',type=str,default='./results of reviewer/')
    parser.add_argument('--log_dir',type=str,default='logging.txt')
    parser.add_argument('--batch_size',type=int,default=512)

    args = parser.parse_args()
    if not os.path.exists(args.save_folder):
        os.makedirs(args.save_folder)
    return args


if __name__ == '__main__':
    args = get_args()
    xjtu_batch_names = ['2C','3C','R2.5','R3','RW','satellite']
    # tju_batch = [0,1,2]
    setattr(args,'model','MLP') # select model: MLP or CNN
    for i in range(len(xjtu_batch_names)):
        setattr(args,'xjtu_batch',xjtu_batch_names[i])
        # setattr(args,'tju_batch',tju_batch[i])
        for e in range(10):
            setattr(args,'save_folder',os.path.join('./results of reviewer/',f'{args.dataset}-{args.model} results/{i}-{i}/Experiment{e+1}'))
            if not os.path.exists(args.save_folder):
                os.makedirs(args.save_folder)

            model = load_model(args)
            data_loader = load_XJTU_data(args)
            trainer = Trainer(model,data_loader['train'],data_loader['valid'],data_loader['test'],args)
            trainer.train()








