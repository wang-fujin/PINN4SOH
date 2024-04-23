'''
绘制小提琴图,并且把所有的数据集都绘制在一张图上
在4个数据集上的常规实验，绘制指标[MAE,MAPE,RMSE]的小提琴图，
在同一幅图上并对比Ours，MLP和CNN

English:
Draw a violin plot and plot all datasets on one figure
Common experiments on 4 data sets, plotting violin plots of indicators [MAE, MAPE, RMSE],
and comparing Ours, MLP, and CNN on the same figure
'''
import matplotlib.pyplot as plt
import scienceplots
plt.style.use(['science','nature'])
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.ticker as mtick
from matplotlib.ticker import MaxNLocator
from matplotlib.lines import Line2D




# pdf = PdfPages(f'violin error.pdf')
fig, axs = plt.subplots(3,4,figsize=(2.3*4,1.8*3),dpi=200)
colors = ['#b8dff2','#abeadb','#ffb2b4']
count = 0
for data in ['XJTU','TJU','MIT','HUST']:
    if data == 'XJTU':
        batches = [0,1,2,3,4,5]
    elif data == 'TJU':
        batches = [0,1,2]
    else:
        batches = [0]
    for batch in batches:
        ############################
        df_list = []
        for model in ['Ours', 'MLP', 'CNN']:

            df1 = pd.read_excel(f'../results analysis/processed results/{model}-{data}-results.xlsx',
                                engine='openpyxl',
                                sheet_name=f'battery_mean_{batch}')

            df1['model'] = [model] * df1.shape[0]
            melted_df1 = pd.melt(df1, id_vars=['model'],
                                 value_vars=['MAE','MAPE','RMSE'],
                                 var_name='metric', value_name='error')
            df_list.append(melted_df1)
        if data in ['MIT', 'HUST']:
            title = data + ' dataset'
        else:
            title = data + f' batch {batch+1}'
        merge_df_keys = ['Ours', 'MLP', 'CNN']

        # 把三个DataFrame拼接起来
        # Concatenate three DataFrames
        df = pd.concat(df_list, axis=0)
        df = df.reset_index()
        df.drop('index', axis=1, inplace=True)
        df['metric'] = df['metric'].astype('category').cat.codes


        # 绘制小提琴图
        # Draw a violin plot
        col = count % 4
        row = count // 4
        print(data, batch, row, col)
        ax = axs[row, col]
        sns.violinplot(x='metric',y='error',hue='model',data=df,
                       scale='count',
                       inner='point',
                       dodge=True,
                       saturation=1,
                       palette=colors,
                       linewidth=0,
                       ax=ax)

        # 在绘制小提琴图后，添加以下代码来计算并绘制均值线和均值加减标准差线
        # After drawing the violin plot, add the following code to calculate and draw the mean line and the mean plus or minus the standard deviation line
        for i, metric in enumerate(['MAE', 'MAPE', 'RMSE']):
            for model in ['Ours', 'MLP', 'CNN']:
                model_mean = df[(df['model'] == model) & (df['metric'] == i)]['error'].mean()  # 计算每个模型的均值 (mean)
                model_std = df[(df['model'] == model) & (df['metric'] == i)]['error'].std()  # 计算每个模型的标准差 (standard deviation)
                # 计算均值线和标准差线的横坐标位置 (x position of the standard deviation line and mean line )
                offset = 0.27
                x_pos = i + (model == 'CNN') * offset - (model == 'Ours') * offset
                # 绘制标准差线 (draw the standard deviation line)
                ax.plot([x_pos, x_pos], [model_mean - model_std, model_mean + model_std], color='black', linestyle='-',
                        linewidth=0.5)
                # 绘制均值线 (draw the mean line)
                ax.plot([x_pos - 0.1, x_pos + 0.1], [model_mean, model_mean], color='red', linestyle='-', linewidth=0.6)

        # 设置x轴范围和标签位置 (set the x-axis range and label position)
        ax.set_xticklabels(['MAE', 'MAPE', 'RMSE'])
        ax.set_xlabel(None)
        # 设置y轴范围 (set the y-axis range)

        # ax.set_ylim(0, 0.055)

        # 在y轴顶上加上百分号 (add a percentage sign on the top of the y-axis)
        def percentage(x, pos):
            if x >= 0.2:
                return '{:.0f}%'.format(x * 100)
            else:
                return '{:.1f}%'.format(x * 100)


        ax.yaxis.set_major_formatter(mtick.FuncFormatter(percentage))
        # 在y轴顶端添加百分号文本标签 (add a percentage text at the top of the y-axis)
        x_min, x_max = ax.get_xlim()
        y_max = ax.get_ylim()[1]
        ax.annotate('(\%)', xy=(x_min, y_max), xytext=(-2, 3),
                    textcoords='offset points', ha='center', fontsize=8)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))

        # 添加标题和标签 (add title and label)
        ax.set_title(title)
        ax.set_ylabel("Error")

        # 关闭图例 (remove the legend)
        ax.get_legend().remove()

        if count >= 11:
            break
        count += 1

    if count >= 11:
        break
# 添加图例 (add legend)
#axs[2, 3].set_visible(False)
# 关闭axs[2,3]的坐标轴 (remove the axis of axs[2,3])
axs[2, 3].axis('off')
boxs = []
for c in colors:
    box = plt.Rectangle((0, 0), 1, 1, fc=c)
    boxs.append(box)
mean_line = Line2D([0], [0], color='red', linestyle='-', linewidth=1)
std_line = Line2D([0, 0], [0, 1], color='black', linestyle='-', linewidth=1)
boxs.append(mean_line)
boxs.append(std_line)
legend_labes = ['Ours', 'MLP', 'CNN', 'Mean', 'Mean $\pm$ Std']
axs[2, 3].legend(handles=boxs,labels=legend_labes, loc=[0.2, 0],
                 handlelength=4,
                 handleheight=2.5,
                 fontsize=8)

plt.tight_layout()
#plt.savefig('violin error.svg',format='svg')
#pdf.savefig()
#pdf.close()
plt.show()
