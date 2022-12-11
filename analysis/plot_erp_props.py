import matplotlib.pyplot as plt

from mne import read_epochs

def get_subsets(path):
    epochs = read_epochs(path).copy()
#
    for e in epochs.events:
        if e[2] == 2 or e[2] == 4 or e[2] == 8 or e[2] == 16 :
            e[2] = 50
        elif e[2] != 255:
            e[2] = 60
#
    epochs.event_id['rep'] = 60
    epochs.event_id['single'] = 50
    epochs.info['bads'].append('Status')
#
    single = epochs['single','rest'].copy().crop(-2,2)
    rep = epochs['rep','rest'].copy()
    return single,rep
