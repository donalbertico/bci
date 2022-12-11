import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from mne import Epochs, create_info, events_from_annotations, read_epochs, preprocessing
from mne.decoding import CSP, UnsupervisedSpatialFilter
from mne.preprocessing import ICA
from mne.time_frequency import AverageTFR

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA


def compute_scores(all_epochs,tmin,tmax,pca_components,csp_components,base_line,):
    csp_fig, axes = plt.subplots(1)
    csp = CSP(n_components=csp_components, reg=None, log=True, norm_trace=False, rank='info')
    if pca_components > 0 :
        pca = UnsupervisedSpatialFilter(PCA(pca_components), average=False)
        clf = make_pipeline(pca,csp,LinearDiscriminantAnalysis())
    else:
        clf = make_pipeline(csp,LinearDiscriminantAnalysis())
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=38)
    # Classification & time-frequency parameters
    n_cycles = 10.  # how many complete cycles: used to define window size
    min_freq = 7.
    max_freq = 120.
    n_freqs = 10  # how many frequency bins to use
    classes = list(all_epochs.event_id.keys())
#
    freqs = np.linspace(min_freq, max_freq, n_freqs)  # assemble frequencies
    freq_ranges = list(zip(freqs[:-1], freqs[1:]))
    freq_labels = []
#
    window_spacing = (n_cycles / np.max(freqs) / 2.)
    centered_w_times = np.arange(tmin, tmax, window_spacing)[1:]
    n_windows = len(centered_w_times)
#
    le = LabelEncoder()
    freq_scores = []
#
    if base_line:
        all_epochs = all_epochs.copy().apply_baseline((-1.5,-0.5))
#
    for freq, (fmin, fmax) in enumerate(freq_ranges):
        # Infer window size based on the frequency being used
        w_size = n_cycles / ((fmax + fmin) / 2.)  # in seconds
        # Apply band-pass filter to isolate the specified frequencies
        # Extract epochs from filtered data, padded by window size
        epochs = all_epochs.copy().crop(tmin,tmax).filter(fmin, fmax, fir_design='firwin',
                                       skip_by_annotation='edge')
        epochs.drop_channels(epochs.info['bads'])
        y = le.fit_transform(epochs.events[:, 2])
        if tmin > 3 and (classes[0] == 'rest' or classes[1] == 'rest'):
            rest =  all_epochs['rest'].copy().crop(0,tmax-tmin).filter(fmin, fmax, fir_design='firwin',
                                           skip_by_annotation='edge')
            rest.drop_channels(rest.info['bads'])
            X = np.concatenate([epochs['rep'].get_data(),rest.get_data()])
        else :
            X = epochs.get_data()
        chance = np.mean(y == y[1])
        chance = max(chance, 1. - chance)
        # Save mean scores over folds for each frequency and time window
        freq_labels.append(str(int(fmin))+' - '+str(int(fmax)))
        freq_scores.append(cross_val_score(
            estimator=clf, X=X, y=y, scoring='roc_auc', cv=cv, n_jobs=8))
        if np.mean(freq_scores[freq]) > chance:
            if pca_components > 0:
                X = pca.fit_transform(X)
                csp.fit(X,y)
                csp.plot_patterns(rest.info, title= 'patterns '+str(int(fmin))+ ' - ' +str(int(fmax)))
            else:
                csp.fit(X,y)
                csp.plot_patterns(epochs.info, title= 'patterns '+str(int(fmin))+ ' - ' +str(int(fmax)))
    scores_df = pd.DataFrame(data=np.transpose(freq_scores), columns=[freq_labels])
    scores_df.boxplot(ax=axes)
    axes.axhline(chance)
    axes.set_xlabel('Frequency (Hz)')
    axes.set_ylabel('Decoding Scores')
    csp_fig.suptitle(classes[0]+' vs '+classes[1])
    csp_fig.show()
    return scores_df

scores = compute_scores(cp,4,8,52,4,True)

plt.bar(freqs[:-1], freq_scores, width=np.diff(freqs)[0],align='edge', edgecolor='black')
plt.xticks(freqs)
plt.ylim([0, 1])
plt.axhline(chance, color='k', linestyle='--',label='chance level')
plt.yticks([round(t,2) for t in np.linspace(0,1,11)])
plt.set_xlabel('Frequency (Hz)')
plt.set_ylabel('Decoding Scores')
plt.set_title('Single vs Rest Scores')
