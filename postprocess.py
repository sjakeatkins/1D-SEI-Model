datafolder = 'TMS_20202020-12-05_161638'

eps_min = 0.025
fs = 12

previous = 0

import os, csv
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

# If you've run this block before, re-set chdir back to its original value, first:
if previous:
    os.chdir(cwd)

# Get the current working directory:
cwd = os.getcwd()
previous = 1
os.chdir(cwd + '/output/' + datafolder)

data = pd.read_csv('solution.csv', header=0)

# Get variable names:
f = open('names.csv', 'r')
names = csv.reader(f)
data_names = data.index.to_list()

i = 0
for row in names:
    data_names[i] = (str('\t'.join(row)))
    i += 1

f.close()

data.index = data_names

from sei_1d_init import N_y, dy, nvars_node

# elif homogeneous:
#    from sei_1d_init import dy, nvars_node
# Transpose the data so that each row is for a time, and each column represents a variable:
data = data.T
data


###  PLOT SEI THICKNESS VS TIME

def thickness_calc(eps_min):
    thickness = []
    eps_SEI = data['eps SEI']

    for j, eps in eps_SEI.iterrows():
        grown = []
        growth = eps[eps['eps SEI'] > eps_min]
        if growth.empty:
            thickness.append(0.)
        else:
            # print(growth.iloc[-1])
            thickness.append((growth.count() - 1. + growth.iloc[-1]) * dy)
    return thickness


t = np.asarray(data.index[:], dtype=float)
y = np.asarray(thickness_calc(eps_min))

fig = plt.figure()
ax = fig.add_axes([0.15, 0.2, 0.75, 0.75])
fig.set_size_inches((3.5, 2.25))

plt.plot(t / 3600, y * 1e9, 'b.', markersize=1.50)
plt.xlim([0, 1.0])
plt.ylim([0, 20])
plt.xlabel('Time (s)')
plt.ylabel('SEI thickness (nm)')
# plt.xticks(np.array([0.,1000.,2000.,3000.,4000.]))


font = plt.matplotlib.font_manager.FontProperties(family='Times New Roman', size=fs)

for tick in ax.xaxis.get_major_ticks():
    tick.label1.set_fontsize(fs)
    tick.label1.set_fontname('Times New Roman')
for tick in ax.yaxis.get_major_ticks():
    tick.label1.set_fontsize(fs)
    tick.label1.set_fontname('Times New Roman')

ax.tick_params(axis="y", direction="in", length=5.5, right=True)
ax.tick_params(axis="x", direction="in", length=5.5, top=True)
ax.set_ylabel('SEI Thickness (nm)', fontsize=fs, family='Times New Roman', labelpad=1.)
ax.set_xlabel('Time (h)', fontsize=fs, family='Times New Roman')
plt.savefig('Thickness_vs_time.pdf', format='pdf', dpi=350)