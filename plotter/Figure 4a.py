'''
绘制不同数据集的预测结果
English:
Plot the prediction results of different datasets
'''
import numpy as np
import matplotlib.pyplot as plt
import scienceplots
plt.style.use(['science','nature'])
from matplotlib.backends.backend_pdf import PdfPages



# 画一个3行4列的图
# Plot a figure with 3 rows and 4 columns
# pdf = PdfPages(f'estimation results.pdf')
fig, axs = plt.subplots(3,4,figsize=(8,6),dpi=150)
count = 0
color_list = [
        '#74AED4',
        '#7BDFF2',
        '#FBDD85',
        '#F46F43',
        '#CF3D3E'
        ]
colors = plt.cm.colors.LinearSegmentedColormap.from_list(
    'custom_cmap', color_list, N=256
)
for data in ['XJTU','TJU','MIT','HUST']:
    if data == 'XJTU':
        batches = [0,1,2,3,4,5]
    elif data == 'TJU':
        batches = [0,1,2]
    else:
        batches = [0]
    for batch in batches:
        # 读取数据
        if data in ['HUST','MIT']:
            root = f'../results/Ours/{data} results/Experiment1/'
            title = f'{data} dataset'
        else:
            root = f'../results/Ours/{data} results/{batch}-{batch}/Experiment1/'
            title = f'{data} batch {batch+1}'
        try:
            pred_label = np.load(root+'pred_label.npy')
            true_label = np.load(root+'true_label.npy')
        except:
            continue
        error = np.abs(pred_label-true_label)
        vmin, vmax = error.min(), error.max()

        lims = {'HUST':[0.79,1.105],
                'MIT':[0.79,1.005],
                'XJTU':[0.79,1.005],
                'TJU':[0.69,0.97],
                }
        # plot
        #fig = plt.figure(figsize=(3,2.6),dpi=200)
        #ax = fig.add_subplot(111)
        col = count%4
        row = count//4
        print(data,batch,row,col)
        ax = axs[row,col]
        ax.scatter(true_label,pred_label,c=error, cmap=colors,s=3,alpha=0.7, vmin=0, vmax=0.1)
        ax.plot([0.65,1.15],[0.65,1.15],'--',c='#ff4d4e',alpha=1,linewidth=1)
        ax.set_aspect('equal')
        ax.set_xlabel('True SOH')
        ax.set_ylabel('Prediction')

        ax.set_xticks([0.7,0.75,0.8,0.85,0.9,0.95,1.0,1.05,1.1])
        ax.set_yticks([0.7,0.75,0.8,0.85,0.9,0.95,1.0,1.05,1.1])
        ax.set_xlim(lims[data])
        ax.set_ylim(lims[data])
        #plt.suptitle(title)
        # 每个子图的标题 (set the title of each subplot)
        ax.set_title(title)

        if count >=11:
            break
        count += 1

    if count >= 11:
        break
# 在最后一个子图上画colorbar (draw a colorbar on the last subplot)
fig.colorbar(plt.cm.ScalarMappable(cmap=colors,norm=plt.Normalize(vmin=0, vmax=0.1)),
             ax=axs[2,3],
             label='Absolute error',

             # 指定colorbar的位置 (set the position of the colorbar)
            fraction=0.46, pad=0.4
             )
# 关闭左后一个子图的坐标轴 (turn off the axis of the last subplot)
axs[2,3].axis('off')

#axs[2,3].set_visible(False)
plt.tight_layout()
# plt.savefig('estimation results.svg',format='svg')
# pdf.savefig(fig)
# pdf.close()
plt.show()