'''
TJU数据集的结果分析

English:
    This file is used to analyze the results of the TJU dataset.
'''
import pandas as pd
import numpy as np
import os
from utils.util import eval_metrix
import matplotlib.pyplot as plt
import scienceplots
plt.style.use('science')

class Results:
    def __init__(self,root='../results/TJU results/',gap=0.07):
        self.root = root
        self.experiments = os.listdir(root)
        self.gap = gap
        self.log_dir = None
        self.pred_label = None
        self.true_label = None
        self._update_experiments(1)

    def _update_experiments(self,train_batch=0,test_batch=1,experiment=1):
        subfolder = f'{train_batch}-{test_batch}/Experiment{experiment}'
        self.log_dir = os.path.join(self.root, subfolder,'logging.txt')
        self.pred_label = os.path.join(self.root, subfolder,'pred_label.npy')
        self.true_label = os.path.join(self.root, subfolder,'true_label.npy')

    def parser_log(self):
        '''
        解析train过程中产生的log文件，获取里面的数据
        English:
            Parse the log file generated during the training process to obtain the data
        :return: dict
        '''
        data_dict = {}

        with open(self.log_dir, 'r') as f:
            lines = f.readlines()

        for line in lines:
            if 'CRITICAL' in line:
                params = line.split('\t')[-1].split('\n')[0]
                k, v = params.split(':')
                data_dict[k] = v

        train_data_loss = []
        train_PDE_loss = []
        train_phy_loss = []
        train_total_loss = []
        valid_data_loss = []

        test_mse = []
        test_epoch = []

        for i in range(len(lines)):
            line = lines[i]
            if '[train] epoch:1 iter:1 data' in line:
                train_data_loss.append(float(line.split('data loss:')[1].split(',')[0]))
                train_PDE_loss.append(float(line.split('PDE loss:')[1].split(',')[0]))
                train_phy_loss.append(float(line.split('physics loss:')[1].split(',')[0]))
                train_total_loss.append(float(line.split('total loss:')[1].split('\n')[0]))
            elif '[Train]' in line:
                train_data_loss.append(float(line.split('data loss:')[1].split(',')[0]))
                train_PDE_loss.append(float(line.split('PDE loss:')[1].split(',')[0]))
                train_phy_loss.append(float(line.split('physics loss:')[1].split(',')[0]))
                train_total_loss.append(float(line.split('total loss:')[1].split('\n')[0]))
            elif '[Valid]' in line:
                valid_data_loss.append(float(line.split('MSE:')[1].split('\n')[0]))
            elif '[Test]' in line:
                test_mse.append(float(line.split('MSE:')[1].split(',')[0]))
                test_epoch.append(int(lines[i - 1].split('epoch:')[1].split(',')[0]))

        data_dict['train_data_loss'] = train_data_loss
        data_dict['train_PDE_loss'] = train_PDE_loss
        data_dict['train_phy_loss'] = train_phy_loss
        data_dict['train_total_loss'] = train_total_loss
        data_dict['valid_data_loss'] = valid_data_loss
        data_dict['test_mse'] = test_mse
        data_dict['test_epoch'] = test_epoch


        line1 = lines[1]
        if '.csv' in line1:
            line = line1[1:-2]
            line_list = line.replace('data/TJU data/', '').replace('.csv','').replace('\'','').split(', ')
            data_dict['IDs_1'] = line_list

        line2 = lines[3]
        if '.csv' in line2:
            line = line2[1:-2]
            line_list = line.replace('data/TJU data/', '').replace('.csv', '').replace('\'', '').split(', ')
            for i in range(len(line_list)):
                line_list[i] = line_list[i].split('\\')[-1]
            data_dict['IDs_2'] = line_list
            #print('test ID length:',len(line_list))

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
        # plt.figure(figsize=(6, 3),dpi=200)
        # plt.plot(true_label, label='true label')
        # plt.plot(pred_label, label='pred label')
        # plt.legend()
        # plt.show()

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
            print('battery {} MAE:{:.4f}, MAPE:{:.4f}, MSE:{:.6f}, RMSE:{:.4f}, R2:{:.4f}'.format(i + 1, MAE_i, MAPE_i,
                                                                                                 MSE_i, RMSE_i, R2_i))
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


    def get_test_results(self,train=0,test=1,e=1):
        '''
        解析训练和测试数据中的电池id
        English:
            Parse the battery id in the training and test sets
        :param e: experiment id
        :return:
        '''
        self._update_experiments(train_batch=train,test_batch=test,experiment=e)
        log_dict = self.parser_log()
        results_dict = self.parser_label()
        results_dict['channel'] = log_dict['IDs_2']
        return results_dict

    def get_battery_average(self,train_batch=0,test_batch=0):
        '''
        计算每次实验中所有电池的平均值
        English:
            Calculate the average value of all batteries in each experiment
        :param train_batch:
        :param test_batch:
        :return: dataframe，每一行是一个实验中所有电池的平均值 (each row is the average value of all batteries in an experiment)
        '''
        df_mean_values = []
        for i in range(1,11):
            res = self.get_test_results(train_batch,test_batch,i)
            df_i = pd.DataFrame(res)
            df_i = df_i[['MAE', 'MAPE', 'RMSE', 'R2']]
            df_mean_values.append(df_i.mean(axis=0).values)
            #print(df_i.mean(axis=0).values)
        df_mean_values = np.array(df_mean_values)
        df_mean = pd.DataFrame(df_mean_values,columns=['MAE', 'MAPE', 'RMSE', 'R2'])
        df_mean.insert(0,'experiment',range(1,11))
        print(df_mean)
        return df_mean


    def get_experiments_mean(self,train_batch=0,test_batch=0):
        '''
        分别获取每个测试电池在所有实验中的平均值
        English:
            Get the average value of each test battery in all experiments
        :return: dataframe，每一行是一个电池在10次实验中的平均值 (each row is the average value of a battery in 10 experiments)
        '''
        df_value_list = []
        for i in range(1,11):
            res = self.get_test_results(train_batch,test_batch,i)
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




if __name__ == '__main__':

    root = '../results/TJU results (small sample 2)/'
    writer = pd.ExcelWriter('processed results (small sample)/Ours-TJU_results (small sample 2).xlsx')
    batch = 0
    results = Results(root, gap=0.07)
    for batch in [2]:
        df_battery_mean = results.get_battery_average(train_batch=batch,test_batch=batch)
        df_experiment_mean = results.get_experiments_mean(test_batch=batch,train_batch=batch)
        df_battery_mean.to_excel(writer,sheet_name='battery_mean_{}'.format(batch),index=False)
        df_experiment_mean.to_excel(writer,sheet_name='experiment_mean_{}'.format(batch),index=False)
    writer.save()
    print(df_experiment_mean.mean())





