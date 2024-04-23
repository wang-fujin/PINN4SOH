import pandas as pd
import os
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import scienceplots
plt.style.use(['science','nature'])
from matplotlib.backends.backend_pdf import PdfPages

root = '../data/XJTU data/'
# pdf = PdfPages('xjtu trajectory.pdf')
files = os.listdir(root)
fig = plt.figure(figsize=(4,2),dpi=200)
colors = [
'#80A6E2',
'#7BDFF2',
'#FBDD85',
'#F46F43',
'#403990',
'#CF3D3E'
]
markers = ['o','v','D','p','s','^']
legends = ['batch 1','batch 2','batch 3','batch 4','batch 5','batch 6']
batches = ['2C','3C','R2.5','R3','RW','satellite']
line_width = 1.0
for i in range(6):
    for f in files:
        if batches[i] in f:
            path = os.path.join(root,f)
            data = pd.read_csv(path)
            capacity = data['capacity'].values
            plt.plot(capacity[1:],color=colors[i],alpha=1,linewidth=line_width,
                     # linestyle=':',
                     marker=markers[i],markersize=2,markevery=50)
plt.xlabel('Cycle')
plt.ylabel('Capacity (Ah)')
custom_lines = [
    Line2D([0], [0], color=colors[0], linewidth=line_width,marker=markers[0],markersize=2.5),
    Line2D([0], [0], color=colors[1], linewidth=line_width,marker=markers[1],markersize=2.5),
    Line2D([0], [0], color=colors[2], linewidth=line_width,marker=markers[2],markersize=2.5),
    Line2D([0], [0], color=colors[3], linewidth=line_width,marker=markers[3],markersize=2.5),
    Line2D([0], [0], color=colors[4], linewidth=line_width,marker=markers[4],markersize=2.5),
    Line2D([0], [0], color=colors[5], linewidth=line_width,marker=markers[5],markersize=2.5)
]

custom_legend = plt.legend(custom_lines, legends, loc='upper right',
                           bbox_to_anchor=(1.0, 1), frameon=False,
                           ncol=3, fontsize=6)


plt.ylim([1.55,2.05])
plt.tight_layout()
plt.show()
# pdf.savefig(fig)
# plt.savefig('xjtu trajectory.svg',format='svg')
# pdf.close()

