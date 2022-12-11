import pyriemann
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder

def compute_psd_scores(all_epochs, tmin, tmax, base_line):
    csp_fig, axes = plt.subplots(1)
    cov = pyriemann.estimation.Covariances()
    mdm = pyriemann.classification.MDM()
    clf = make_pipeline(cov,mdm)
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=52)
#
    min_freq = 7.
    max_freq = 120.
    n_freqs = 10  # how many frequency bins to use
    classes = list(all_epochs.event_id.keys())
    freqs = np.linspace(min_freq, max_freq, n_freqs)
    freq_ranges = list(zip(freqs[:-1], freqs[1:]))
#
    le = LabelEncoder()
    freq_scores = []
    freq_labels = []
#
    if base_line:
        epochs = all_epochs.copy().apply_baseline((-1.5,-0.5))
#
    for freq, (fmin, fmax) in enumerate(freq_ranges):
        psd_fig, axesP = plt.subplots(1)
        epochs = all_epochs.copy().crop(tmin,tmax).filter(fmin, fmax, fir_design='firwin', skip_by_annotation='edge')
        epochs.drop_channels(all_epochs.info['bads'])
        y = le.fit_transform(epochs.events[:, 2])
        X = epochs.get_data() * 1e6
        chance = np.mean(y == y[1])
        chance = max(chance, 1. - chance)
        freq_scores.append(cross_val_score(estimator=clf, X=X, y=y, cv=cv, n_jobs=8))
        freq_labels.append(str(int(fmin))+' - '+str(int(fmax)))
    scores_df = pd.DataFrame(data=np.transpose(freq_scores), columns=[freq_labels])
    scores_df.boxplot(ax=axes)
    axes.axhline(chance)
    axes.set_xlabel('Frequency (Hz)')
    axes.set_ylabel('Decoding Scores')
    csp_fig.suptitle('MDM '+classes[0]+' vs '+classes[1])
    csp_fig.show()
    return scores_df

score = compute_psd_scores(single,0, 1.2, False)

            if np.mean(freq_scores[freq]) > chance:
                X = cov.fit_transform(X)
                # mdm.fit(X,y)
                # df = pd.DataFrame(data=mdm.covmeans_[0], index=epochs.ch_names, columns=epochs.ch_names)
                # g = sns.heatmap(
                #     df, ax=axesP[0], square=True, cbar=False, xticklabels=2, yticklabels=2)
                # g.set_title(str(int(fmin))+'-'+str(int(fmax))+'covariance - ')
                # df = pd.DataFrame(data=mdm.covmeans_[1], index=epochs.ch_names, columns=epochs.ch_names)
                # g = sns.heatmap(
                #     df, ax=axesP[1], square=True, cbar=False, xticklabels=2, yticklabels=2)
                # g.set_title(+'Hz covariance - '+classes[0])
                # psd_fig.show()
