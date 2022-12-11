import mne
from mne.preprocessing import annotate_muscle_zscore

epochs = mne.read_epochs_eeglab('pilot2/epochs.set')
filtered = epochs.copy().filter(1,120)

raw = epochs.get_data()
raw = raw.reshape(59,-1)
raw = mne.io.RawArray(raw, info = epochs.info)
annot_muscle, scores_muscle = annotate_muscle_zscore(raw,threshold=2, min_length_good=0.5)
raw.set_annotations(annot_muscle)
raw.plot(duration = 50, scalings = dict(eeg=20e-5))

ica = mne.preprocessing.ICA(n_components = 15, method = 'picard')
ica.fit(filtered)
exclude = [0,2,3,4,5,7,8,10,14]
ica.apply(epochs,exlcude exclude)

ecg_indices, ecg_scrores = ica.find_bads_ecg(epochs, ch_name='Fpz', method='correlation', threshold='auto')
