import numpy as np
import matplotlib.pyplot as plt

from mne import Epochs, create_info, events_from_annotations, read_epochs, preprocessing
from mne.io import concatenate_raws, read_raw_edf
from mne.decoding import CSP
from mne.time_frequency import AverageTFR

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder


epochs_s = read_epochs('pilot2\epochs_single.fif')
# sgl_bad_epochs = [0, 2, 3, 5, 9, 16, 17, 19, 20, 21, 24, 26, 40, 43, 48, 52, 55, 56, 57, 61, 68, 80, 85, 87, 97, 98, 109, 110, 118]
bads = ['Pz', 'FT8', 'P8','TP8', 'P10', 'P9', 'PO8', 'Fp1', 'AF7', 'AF3', 'Fpz', 'AF4', 'AF8', 'T8', 'F6', 'CP6', 'FC4', 'C4', 'CP4', 'Fp2']
sgl_bad_epochs = [2, 11, 33, 36, 49, 51, 65, 70]
# bads = ['Pz', 'F7', 'POz', 'FT8', 'P8', 'P10', 'P9', 'PO8', 'AF3', 'AF7', 'Fp1', 'AF4', 'AF8', 'Fp2', 'Fpz', 'T8', 'FC4', 'F8', 'F6', 'Oz', 'F5', 'CP6', 'C6']
#

epochs_s.drop(sgl_bad_epochs)
# epochs_s.plot(scalings=10e-5, n_epochs=12, events=epochs_s.events)
ica_s =  preprocessing.read_ica('pilot2\ica_single.fif')

ica_s.apply(epochs_s, exclude=[0,1,3,9,16,15,14])
epochs_s.info['bads'] = bads

csp = CSP(n_components=3,reg=None,log=True, norm_trace=False, rank='info')
# csp = CSP(n_components=3,cov_est ='epoch',component_order='alternate',reg=None,log=True, norm_trace=False, rank='info')
clf = make_pipeline(csp,
                    LinearDiscriminantAnalysis())
n_splits = 10  # for cross-validation, 5 is better, here we use 3 for speed
cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# Classification & time-frequency parameters
tmin, tmax = -.200, 2.000
n_cycles = 10.  # how many complete cycles: used to define window size
min_freq = 7.
max_freq = 120.
n_freqs = 10  # how many frequency bins to use

freqs = np.linspace(min_freq, max_freq, n_freqs)  # assemble frequencies
freq_ranges = list(zip(freqs[:-1], freqs[1:]))

window_spacing = (n_cycles / np.max(freqs) / 2.)
centered_w_times = np.arange(tmin, tmax, window_spacing)[1:]
n_windows = len(centered_w_times)

# Instantiate label encoder
le = LabelEncoder()

freq_scores = np.zeros((n_freqs - 1,))


chance = len(epochs_s['single']) / len(epochs_s)
# Loop through each frequency range of interest
# epochs_s.apply_baseline(-1.0,0.0)

for freq, (fmin, fmax) in enumerate(freq_ranges):
    # Infer window size based on the frequency being used
    w_size = n_cycles / ((fmax + fmin) / 2.)  # in seconds
    # Apply band-pass filter to isolate the specified frequencies
    # Extract epochs from filtered data, padded by window size
    epochs = epochs_s.copy().crop(0,1.5).filter(fmin, fmax, fir_design='firwin',
                                   skip_by_annotation='edge')
    epochs.drop_channels(epochs_s.info['bads'])
    y = le.fit_transform(epochs.events[:, 2])
    X = epochs.get_data()
    # Save mean scores over folds for each frequency and time window
    csp.fit(X,y)
    freq_scores[freq] = np.mean(cross_val_score(
        estimator=clf, X=X, y=y, scoring='roc_auc', cv=cv, n_jobs=8), axis=0)
    if freq_scores[freq] > chance:
        csp.plot_patterns(epochs.info, title= 'patterns '+str(int(fmin))+ ' - ' +str(int(fmax)))


# fig, ax = plt.subplots()
plt.bar(freqs[:-1], freq_scores, width=np.diff(freqs)[0],align='edge', edgecolor='black')
plt.xticks(freqs)
plt.ylim([0, 1])
plt.axhline(chance, color='k', linestyle='--',label='chance level')
plt.yticks([round(t,2) for t in np.linspace(0,1,11)])
plt.xlabel('Frequency (Hz)')
plt.ylabel('Decoding Scores')
plt.title('Single vs Rest Scores')



epochs_r = read_epochs('pilot2\epochs_r.fif')

ica_r =  preprocessing.read_ica('pilot2\ica_repe.fif')
ica_r.apply(epochs_r, exclude=[0,1,4,14,17,11,8,7,15])
# ica_r.apply(epochs_r, exclude=[0,4,15,5])


epochs_r.drop_channels(['Pz','FC5','CP4','F8','P4','P6','PO4','FT8','TP8','AF7','Fpz','O1','Fp2','CP6','AF8', 'P8', 'POz','Oz','P10', 'P9', 'PO8', 'Fp1', 'AF3', 'AF8', 'Fp2', 'AF4', 'P7', 'T8', 'C6', 'C4', 'F6', 'FC4', 'FC6'])
# epochs_r.drop_channels(['Pz','FT8','TP8','AF7','Fpz','O1','Fp2','CP6','AF8', 'P8', 'POz','Oz','P10', 'P9', 'PO8', 'Fp1', 'AF3', 'AF8', 'Fp2', 'AF4', 'P7', 'T8', 'C6', 'C4', 'F6', 'FC4', 'FC6'])
epochs_r.drop([ 6, 9, 22, 8, 20, 24, 25, 28, 31, 32, 44, 57, 58, 41])
# epochs_r.apply_baseline((-1.5,-0.5))

chance = len(epochs_r['repetition']) / len(epochs_r)

for freq, (fmin, fmax) in enumerate(freq_ranges):
    rest = epochs_r['rest'].copy().filter(fmin, fmax, fir_design='firwin',
                                   skip_by_annotation='edge').crop(0,4)
    rep = epochs_r['repetition'].copy().filter(fmin, fmax, fir_design='firwin',
                                   skip_by_annotation='edge').crop(8,12)
    X = np.concatenate([rest.get_data(),rep.get_data()])
    labels = np.concatenate([rest.events[:,-1],rep.events[:,-1]])
    y = le.fit_transform(labels)
    # Save mean scores over folds for each frequency and time window
    csp.fit(X,y)
    freq_scores[freq] = np.mean(cross_val_score(
        estimator=clf, X=X, y=y, scoring='roc_auc', cv=cv, n_jobs=8), axis=0)
    if freq_scores[freq] > chance:
        csp.plot_patterns(epochs_r.info, title= 'patterns '+str(int(fmin))+ ' - ' +str(int(fmax)))


plt.bar(freqs[:-1],
    freq_scores, width=np.diff(freqs)[0],align='edge',
    edgecolor='black',color = 'grey')
plt.xticks(freqs)
plt.ylim([0, 1])
plt.axhline(chance, color='k', linestyle='--',label='chance level')
plt.yticks([round(t,2) for t in np.linspace(0,1,11)])
plt.xlabel('Frequency (Hz)')
plt.ylabel('Decoding Scores')
plt.title('Repetition (4s) vs Rest Scores - 9ica')
