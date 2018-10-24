import csv
import numpy as np
import matplotlib.pyplot as plt
from modules.utils import phase_length

# Visualize experiments result given the prediction labels

local_path = '/Users/seanxiang/data/cholec80/pred/'
remote_path = '/home/cxia8134/dev/baseline/pred/'

colours = {
    '0': 'red',  # preparation
    '1': 'green',  # calot triangle dissection
    '2': 'blue',  # clipping and cutting
    '3': 'cyan',  # gallbladder dissection
    '4': 'yellow',  # gallbladder packaging
    '5': 'pink',  # cleaning and coagulation
    '6': 'saddlebrown'  # gallbladder retraction
}
for i in range(20):
    # read prediction and label files
    y_pred_name = '{}{}.txt'.format(local_path, i)
    y_true_name = '{}{}_true.txt'.format(local_path, i)

    y_pred = None
    y_true = None
    with open(y_pred_name, 'r') as f:
        reader = csv.reader(f)
        y_pred = np.array(list(reader))

    with open(y_true_name, 'r') as f:
        reader = csv.reader(f)
        y_true = np.array(list(reader))

    l = len(y_pred)
    x = np.arange(l)

    # find the length of each phase
    pred_phases = phase_length(y_pred)
    true_phases = phase_length(y_true)

    # set up layout
    fig = plt.figure(figsize=(8, 6))
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    ax1.plot(color='black')

    # plotting ground truth
    start = 0
    for j in range(len(true_phases)):
        # for each phase, generate mask and plot a color band for it
        mask = np.zeros(l)
        phase = true_phases[j]
        length = phase[0]
        phase_label = phase[1]
        end = start + length
        for k in range(start, end):
            mask[k] = 1
        start = end

        co = colours[phase_label]
        band = ax1.fill_between(x, 0, 1, where=(mask == 1), facecolors=colours[phase_label], alpha=1, label=phase_label)
    plt.legend(loc='upper right', ncol=8)

    # plotting predicted phases
    start = 0
    for j in range(len(pred_phases)):
        mask = np.zeros(l)
        phase = pred_phases[j]
        length = phase[0]
        phase_label = phase[1]
        end = start + length
        for k in range(start, end):
            mask[k] = 1
        start = end

        co = colours[phase_label]
        band = ax2.fill_between(x, 0, 1, where=(mask == 1), facecolors=colours[phase_label], alpha=1, label=phase_label)

    plt.legend(loc='best', ncol=8)
    plt.savefig('../visualization/{}.jpg'.format(i), transparent=False)
    plt.show()
