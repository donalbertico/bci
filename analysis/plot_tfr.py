import numpy as np
import pandas as pd
import seaborn as sns
from mne.time_frequency import tfr_multitaper

def plot_tfr(epochs,tmin, tmax, channels):
    freqs = np.arange(7,120)
    freq_bounds = {'_': 0,
                   'delta': 3,
                   'theta': 7,
                   'alpha': 13,
                   'beta': 35,
                   'l-gamma': 70,
                   'm-gamma': 90,
                   'h-gamma': 120}
    tfr = tfr_multitaper(epochs.copy().pick(channels), freqs=freqs, n_cycles=freqs, n_jobs=8, use_fft=True,
            return_itc=False, average=False, decim=2)
    tfr.apply_baseline((-1.5,-0.3), mode='percent')
    tfr.crop(tmin,tmax)
#
    df = tfr.to_data_frame(time_format=None, long_format=True)
    df['band'] = pd.cut(df['freq'], list(freq_bounds.values()), labels=list(freq_bounds)[1:])
#
    df = df[df.band.isin(['beta','l-gamma','m-gamma'])]
    df['band'] = df['band'].cat.remove_unused_categories()
    g = sns.FacetGrid(df, row='band', col='channel', margin_titles=True)
    g.map(sns.lineplot, 'time', 'value', 'condition', n_boot=10)
    axline_kw = dict(color='black', linestyle='dashed', linewidth=0.5, alpha=0.5)
    g.map(plt.axhline, y=0, **axline_kw)
    g.map(plt.axvline, x=0, **axline_kw)
    # g.set(ylim=(None, 1.5))
    g.set(xlim=(tmin, tmax))
    g.set_axis_labels("Time (s)", "ERDS (%)")
    g.set_titles(col_template="{col_name}", row_template="{row_name}")
    g.add_legend(ncol=2, loc='lower center')
    g.fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.08)
    plt.show()
