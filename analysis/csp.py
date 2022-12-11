import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mne.decoding import CSP
from sklearn.decomposition import PCA
from mne.decoding import UnsupervisedSpatialFilter
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import ShuffleSplit, cross_val_score

pca = UnsupervisedSpatialFilter(PCA(50), average=False)
lda = LinearDiscriminantAnalysis()
csp = CSP(n_components=3, log=True, ,rank='info')
cv = ShuffleSplit(50, test_size=0.2, random_state=42)

fig, axes = plt.subplots(4,1)

# raw = mne.io.read_raw_fif('pilot2/raw.fif', preload=True)
epochs = mne.read_epochs('pilot2\epochs.fif')
epochs_s = mne.read_epochs('pilot2\epochs_single.fif')
sgl_bad_epochs = [2, 11, 33, 36, 49, 51, 65, 70]
bads = ['Pz', 'F7', 'POz', 'FT8', 'P8', 'P10', 'P9', 'PO8', 'AF3', 'AF7', 'Fp1', 'AF4', 'AF8', 'Fp2', 'Fpz', 'T8', 'FC4', 'F8', 'F6', 'Oz', 'F5', 'CP6', 'C6']
sub_set =  epochs_s.copy().pick(['FT7','FC5','FC3','C3','C5','T7','CP3','CP5','CP1','C1','P1','P3','P5','P7','TP7'])
# epochs_s.plot(n_channels=20, n_epochs=20, scalings=dict(eeg=20e-5))
sgl_artifacts = [0,1,2,3,5,6,7,8,11,13,14,17,19,18,20,21,24]

ica_s =  mne.preprocessing.read_ica('pilot2\ica_single.fif')

ica_s.apply(epochs_s, exclude=[0,1,3,9,16])
epochs_s.info['bads'] = bads
epochs_s.drop(sgl_bad_epochs)
# epochs_s.apply_baseline((-1.0,0.))
cp = epochs_s.copy().crop(0,1.5).filter(7,20)
# data = pca.fit_transform(cp.get_data())
cp.drop_channels(cp.info['bads'])
labels = epochs_s.events[:,-1]

# csp.fit(cp.get_data(),labels)
# csp.plot_patterns(cp.info)
# csp.fit(data.get_data(),labels)
# csp.plot_patterns(epochs.info, ch_type='eeg', units='Patterns (AU)')

# clf = Pipeline([('PCA',pca),('CSP', csp), ('LDA', lda)])
clf = Pipeline([('CSP', csp), ('LDA', lda)])
scores = cross_val_score(clf, cp.get_data(), labels, cv=cv, n_jobs=8)
np.mean(scores)

scores_c = cross_val_score(clf, clean.get_data(), labels, cv=cv, n_jobs=8)

scores_df = pd.DataFrame(data = np.transpose([scores,scores_c]),columns=['-1ica','-6ica'])
scores_df.boxplot(ax=axes[3])
class_balance = np.mean(labels == labels[1])
class_balance = max(class_balance, 1. - class_balance)

axes[3].set_title('35-90Hz rep vs rest')
# axes[3].set_ylabel('accuracy')
axes[3].axhline(class_balance)


epochs_r = mne.read_epochs('pilot2\epochs_r.fif')

ica_r =  mne.preprocessing.read_ica('pilot2\ica_repe.fif')
ica_r.apply(epochs_r, exclude=[0,1,4,14,17])
# ica_r.apply(epochs_r, exclude=[0,4,15,5])


epochs_r.drop_channels(['Pz', 'FT8','TP8','AF7','Fpz','O1','Fp2','CP6','AF8', 'P8', 'POz','Oz','P10', 'P9', 'PO8', 'Fp1', 'AF3', 'AF8', 'Fp2', 'AF4', 'P7', 'T8', 'C6', 'C4', 'F6', 'FC4', 'FC6'])
epochs_r.drop([ 6, 9, 22, 8, 20, 24, 25, 28, 31, 32, 44, 57, 58, 41])
# epochs_r.apply_baseline((-1.5,-0.5))
rest = epochs_r['rest'].copy().filter(7,20).crop(0,4)
rep = epochs_r['repetition'].copy().filter(7,20).crop(8,12)
data = np.concatenate([rest.get_data(),rep.get_data()])
labels = np.concatenate([rest.events[:,-1],rep.events[:,-1]])

csp.fit(data,labels)
csp.plot_patterns(epochs_r.info)

scores = cross_val_score(clf, data, labels, cv=cv, n_jobs=8)
np.mean(scores)
class_balance = np.mean(labels == labels[1])
class_balance = max(class_balance, 1. - class_balance)
