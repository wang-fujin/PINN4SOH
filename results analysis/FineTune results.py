'''
解析fine-tuning的结果

English:
Parse the fine-tuning results
'''

import pandas as pd
import numpy as np
import os
from utils.util import eval_metrix
import matplotlib.pyplot as plt
import scienceplots
plt.style.use('science')

class Results:
    def __init__(self,root='../results_fine-tuning/HUST-XJTU/',gap=0.07):
        self.root = root
        self.experiments = os.listdir(root)
        self.dataset = root.split('/')[-2]
        self.source = root.split('tuning/')[-1].split('-')[0]
        self.target = root.split('tuning/')[-1].split('-')[1][:-1]
        self.gap = gap
        self.log_dir = None
        self.pred_label = None
        self.true_label = None
        self._update_experiments(1)

    def _update_experiments(self,batch,experiment=1):
        if  'XJTU' in self.dataset and 'TJU' in self.dataset:
            subfolder = f'batch{batch}/Experiment{experiment}'
        else:
            subfolder = f'Experiment{experiment}'
        self.log_dir = os.path.join(self.root, subfolder,'logging.txt')
        self.pred_label = os.path.join(self.root, subfolder,'pred_label.npy')
        self.true_label = os.path.join(self.root, subfolder,'true_label.npy')

    def parser_log(self):
        '''
        解析train过程中产生的log文件，获取里面的数据
        English:
        parse the log file generated during the training process to obtain the data
        :return: dict
        '''
        data_dict = {}

        with open(self.log_dir, 'r') as f:
            lines = f.readlines()

        # 解析超参数，logging等级为CRITICAL
        # Parse hyperparameters, logging level is CRITICAL
        for line in lines:
            if 'CRITICAL' in line:
                params = line.split('\t')[-1].split('\n')[0]
                k, v = params.split(':')
                data_dict[k] = v

            # source only
            if 'Source only:' in line:
                source_only = {}
                text = line.split('Source only:')[-1]
                source_only['task'] = text.split('|')[0]
                source_only['mse'] = float(text.split('MSE:')[1].split(',')[0])
                source_only['mae'] = float(text.split('MAE:')[1].split(',')[0])
                source_only['mape'] = float(text.split('MAPE:')[1].split(',')[0])
                source_only['rmse'] = float(text.split('RMSE:')[1].split('\n')[0])
                self.source_only = source_only

        # 解析train/valid/test过程中的loss
        # Parse the loss during the train/valid/test process
        train_data_loss = []
        train_PDE_loss = []
        train_phy_loss = []
        train_total_loss = []
        valid_data_loss = []

        test_mse = []
        test_epoch = []
        # 第一个iter的损失 (the loss of the first iter)
        for i in range(len(lines)):
            line = lines[i]
            if '[train] epoch:1 iter:1 data' in line:
                train_data_loss.append(float(line.split('data loss:')[1].split(',')[0]))

            elif '[Train]' in line:
                train_data_loss.append(float(line.split('data loss:')[1].split(',')[0]))
            elif '[Valid]' in line:
                valid_data_loss.append(float(line.split('MSE:')[1].split('\n')[0]))

        data_dict['train_data_loss'] = train_data_loss

        data_dict['valid_data_loss'] = valid_data_loss


        # 解析数据路径 (Parse the data path)
        line1 = lines[1]
        if '.csv' in line1:
            line = line1[1:-2]
            line_list = line.replace(f'data/{self.dataset} data/', '').replace('.csv','').replace('\'','').split(', ')
            data_dict['IDs_1'] = line_list

        line2 = lines[3]
        if '.csv' in line2:
            line = line2[1:-2]
            line_list = line.replace(f'data/{self.dataset} data/', '').replace('.csv', '').replace('\'', '').split(', ')
            for i in range(len(line_list)):
                line_list[i] = line_list[i].split('\\')[-1]
            data_dict['IDs_2'] = line_list

        return data_dict

    def parser_label(self):
        '''
        解析预测结果
        English:
        Parse the prediction results
        :return:
        '''
        pred_label = np.load(self.pred_label).reshape(-1)
        true_label = np.load(self.true_label).reshape(-1)
        [MAE, MAPE, MSE, RMSE, R2] = eval_metrix(pred_label, true_label)
        # fig = plt.figure(figsize=(6, 4))
        # plt.plot(true_label, label='true label')
        # plt.plot(pred_label, label='pred label')
        # plt.legend()
        # plt.show()
        # plt.close(fig)

        # 用来保存每个电池的预测结果
        # To save the prediction results of each battery
        pred_label_list = []
        true_label_list = []
        MAE_list = []
        MAPE_list = []
        MSE_list = []
        RMSE_list = []
        R2_list = []

        diff = np.diff(true_label)
        split_point = np.where(diff > self.gap)[0]
        local_minima = np.concatenate((split_point, [len(true_label)]))

        start = 0
        end = 0
        for i in range(len(local_minima)):
            end = local_minima[i]
            pred_i = pred_label[start:end]
            true_i = true_label[start:end]
            [MAE_i, MAPE_i, MSE_i, RMSE_i, R2_i] = eval_metrix(pred_i, true_i)
            # print('battery {} MAE:{:.4f}, MAPE:{:.4f}, MSE:{:.6f}, RMSE:{:.4f}, R2:{:.4f}'.format(i + 1, MAE_i, MAPE_i,
            #                                                                                       MSE_i, RMSE_i, R2_i))
            start = end + 1

            pred_label_list.append(pred_i)
            true_label_list.append(true_i)
            MAE_list.append(MAE_i)
            MAPE_list.append(MAPE_i)
            MSE_list.append(MSE_i)
            RMSE_list.append(RMSE_i)
            R2_list.append(R2_i)
        #print('Mean  MAE:{:.4f}, MAPE:{:.4f}, MSE:{:.6f}, RMSE:{:.4f}, R2:{:.4f}'.format(MAE, MAPE, MSE, RMSE, R2))
        results_dict = {}
        results_dict['pred_label'] = pred_label_list
        results_dict['true_label'] = true_label_list
        results_dict['MAE'] = MAE_list
        results_dict['MAPE'] = MAPE_list
        # results_dict['MSE'] = MSE_list
        results_dict['RMSE'] = RMSE_list
        results_dict['R2'] = R2_list
        return results_dict


    def get_test_results(self,batch=1,e=1):
        '''
        解析训练和测试数据中的电池id
        English:
        Parse the battery id in the training and test data
        :param e: experiment id
        :return:
        '''
        self._update_experiments(batch=batch,experiment=e)
        log_dict = self.parser_log()
        results_dict = self.parser_label()
        results_dict['channel'] = log_dict['IDs_2']
        return results_dict

    def get_experiments_mean(self,batch=0,num=10):
        '''
        分别获取每个测试电池在所有实验中的平均值
        English:
        Get the average value of each test battery in all experiments
        :return: dataframe，每一行是一个电池在10次实验中的平均值 (each row is the average value of a battery in 10 experiments)
        '''
        df_value_list = []
        for i in range(1,1+num):
            res = self.get_test_results(batch,i)
            df = pd.DataFrame(res)
            df = df[['channel','MAE', 'MAPE', 'RMSE', 'R2']]
            df = df.sort_values(by='channel')
            df.reset_index(drop=True, inplace=True)
            df_value_list.append(df[['MAE','MAPE','RMSE','R2']].values)
        channel = df['channel']
        columns = ['MAE', 'MAPE', 'RMSE', 'R2']

        np_array = np.array(df_value_list)
        np_mean = np.mean(np_array,axis=0)
        df_mean = pd.DataFrame(np_mean,columns=columns)
        df_mean.insert(0,column='channel',value=channel)
        #df_mean['channel'] = df_mean['channel'].astype(str)
        #df_mean['channel'] = df_mean['channel'].apply(lambda x: x[-9:])
        print(df_mean)
        return df_mean

    def get_battery_average(self,batch,num=10):
        '''
        计算每次实验中所有电池的平均值
        English:
        Calculate the average value of all batteries in each experiment
        :param train_batch:
        :param test_batch:
        :return: dataframe，每一行是一个实验中所有电池的平均值 (each row is the average value of all batteries in an experiment)
        '''
        df_mean_values = []
        for i in range(1,1+num):
            res = self.get_test_results(batch,i)
            df_i = pd.DataFrame(res)
            df_i = df_i[['MAE', 'MAPE', 'RMSE', 'R2']]
            df_mean_values.append(df_i.mean(axis=0).values)
            #print(df_i.mean(axis=0).values)
        df_mean_values = np.array(df_mean_values)
        df_mean = pd.DataFrame(df_mean_values,columns=['MAE', 'MAPE', 'RMSE', 'R2'])
        df_mean.insert(0,'experiment',range(1,1+num))
        print(df_mean)
        return df_mean

    def get_source_only(self):
        '''
        获取source only的结果
        English:
        Get the results of 'source only'
        :return:
        '''
        print(self.source_only)
        df = pd.DataFrame(self.source_only,index=[1])
        return df




if __name__ == '__main__':
    source = 'XJTU'
    target = 'HUST'

    com_root = f'../results_fine-tuning/{source}-{target}/'
    writer = pd.ExcelWriter(f'processed results (fine tuning)/Ours-FineTune-{source}-{target}.xlsx')
    xjtu_gap = 0.05
    tju_gap = 0.07
    results = Results(com_root,gap=tju_gap)
    batch = 0

    for batch in range(3):
        #df_experiment_mean = results.get_experiments_mean(train_batch=batch,test_batch=batch)
        df_battery_mean = results.get_battery_average(batch=batch)
        df_source_only = results.get_source_only()


        # # 最后两行添加所有样本的均值和方差 (add the mean and variance of all samples in the last two rows)
        # mean = df_battery_mean.mean(axis=0)
        # std = df_battery_mean.std(axis=0)
        # df_battery_mean = df_battery_mean.append(mean, ignore_index=True)
        # df_battery_mean = df_battery_mean.append(std, ignore_index=True)

        df_battery_mean.to_excel(writer, f'battery_mean_{batch}', index=False)
        df_source_only.to_excel(writer, f'source_only_{batch}', index=False)
        #df_experiment_mean.to_excel(writer,f'experiment_mean_{batch}',index=False)
    writer.save()



