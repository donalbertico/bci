import numpy as np

import mne
from mne import io
from mne.stats import permutation_t_test

epochs = mne.read_epochs('pilot2\single_epochs.fif')
epochs.filter(7,30)
epochs.apply_baseline(-1.5,-0.5)
rest_ep = epochs['rest']

data = rest_ep.get_data()
times = rest_ep.times

mask = np.logical_and(0.0 <= times, times <=1)
data = np.mean(data[:,:, mask], axis=2)
data = data

n_permutations = 50000
T0, p_values, H0 = permutation_t_test(data, n_permutations)

p_values <= 0.05
