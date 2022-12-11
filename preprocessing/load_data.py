import mne
import numpy as np


noisy = ['C1','POz','Oz','F2','P2','PO4','O2','CP2']

single_e = mne.read_epochs('pilot2/epochs_single.fif', preload=True)

artifacts_single = [0,1,2,3,5,12,4]

epochs.drop([8, 14,3, 43, 54, 61, 72, 87, 110,16, 20, 24, 48, 55, 69, 100, 121, 23, 27, 33, 35, 37, 39, 42, 46, 53, 64, 67, 71, 73, 78, 95, 101, 106, 115])
epochs.filter(1,120)

for e in epochs.events:
    if e[2] == 1 or e[2] == 2 or e[2] == 8 or e[2] == 9:
        e[2] = 20
    elif e[2] != 7:
        e[2] = 30

epochs.event_id['single'] = 20
epochs.event_id['repetition'] = 30
