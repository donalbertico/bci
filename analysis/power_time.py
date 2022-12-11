import mne
import matplotlib.pyplot as plt
import scipy.stats as stats

epochs = mne.read_epochs('pilot2\cleaned.fif')

bands = [dict(fmin=4, fmax=8, title='4-8Hz'),
        dict(fmin=8, fmax=12, title='8-12Hz'),
        dict(fmin=12, fmax=30, title='12-30Hz'),
        dict(fmin=30, fmax=60, title='30-60Hz')]
rept_cls = ['r_pinch','r_stop','r_right','r_left']
single_cls = ['right','left','pause','work']
# broca = ['FC5','FC3']
# wernicke = ['CP3','P3','CP5','P5']
# auditory = ['C5','T7','TP7']
overlap_epochs = [16, 34, 55, 62, 71, 77,101]
# overlap_epochs = [19, 40, 60, 74, 80, 106]
broca = ['FC5']
wernicke = ['CP5']
auditory = ['C5']
epochs.drop(overlap_epochs)

broca_id = mne.pick_channels(epochs.info['ch_names'], include=broca)
wernicke_id = mne.pick_channels(epochs.info['ch_names'], include=wernicke)
auditory_id = mne.pick_channels(epochs.info['ch_names'], include=auditory)

# sgl_mean.plot(titles=dict(eeg='single imagery - brocka/wirneck'))
# rest_mean.plot(titles=dict(eeg='rest - brocka/wirneck'))
rio_dict = dict(broca_id = broca_id, wernicke_id = wernicke_id, auditory = auditory_id)

fig, axes = plt.subplots(4,3)

epochs = epochs.pick(broca+wernicke+auditory)
row = 0
for band in bands :
    rep_epochs = epochs[rept_cls].crop(tmin=0, tmax=1.5).filter(band['fmin'],band['fmax']).average()
    single_epochs = epochs[single_cls].crop(tmin=0, tmax=1.5).filter(band['fmin'],band['fmax']).average()
    rest_epochs = epochs['rest'].crop(tmin=0, tmax=1.5).filter(band['fmin'],band['fmax']).average()
    re_df = rep_epochs.to_data_frame()
    sg_df = single_epochs.to_data_frame()
    rest_df = rest_epochs.to_data_frame()
    for ch in broca+wernicke+auditory:
        re_df[ch] = re_df[ch]**2
        sg_df[ch] = sg_df[ch]**2
        rest_df[ch] = rest_df[ch]**2
    # rio_dict = dict(broca_id = broca_id, wernicke_id = wernicke_id, auditory = auditory_id)
    # rep_mean = mne.channels.combine_channels(rep_epochs,rio_dict, method='mean')
    # rest_mean = mne.channels.combine_channels(rest_epochs,rio_dict, method='mean')
    # sgl_mean = mne.channels.combine_channels(single_epochs,rio_dict, method='mean')
    st, p = stats.ttest_rel(a=rest_df['FC5'], b=sg_df['FC5'])
    if row == 0:
        title = str(broca) +' t:' + str(format(st, '.2f'))
    else:
        title =  ' t:' + str(format(st, '.2f'))
    rest_df.plot(x='time',y=broca,ax=axes[row,0], title= title)
    sg_df.plot(x='time',y=broca,ax=axes[row,0], title= title)
    axes[row,0].set_ylabel(band['title'])
    axes[row,0].set_xlabel('')
    if row == 0 :
        axes[row,0].legend(['rest', 'single'])
    else:
        axes[row,0].legend('', frameon=False)
    # mne.viz.plot_compare_evokeds(conditions, picks = broca, title=title, combine = 'mean', show=False, axes=axes[row,0])
    st, p = stats.ttest_rel(a=rest_df['CP5'], b=sg_df['CP5'])
    if row == 0:
        title =  str(wernicke) +' t:' + str(format(st, '.2f'))
    else:
        title = ' t:' + str(format(st, '.2f'))
    rest_df.plot(x='time',y=wernicke,ax=axes[row,1], title= title)
    sg_df.plot(x='time',y=wernicke,ax=axes[row,1], title= title)
    axes[row,1].legend('', frameon=False)
    axes[row,1].set_xlabel('')
    st, p = stats.ttest_rel(a=rest_df['C5'], b=sg_df['C5'])
    if row == 0:
        title = str(auditory) +' t:' + str(format(st, '.2f'))
    else:
        title = ' t:' + str(format(st, '.2f'))
    rest_df.plot(x='time',y=auditory,ax=axes[row,2], title= title)
    sg_df.plot(x='time',y=auditory,ax=axes[row,2], title= title)
    axes[row,2].legend('', frameon=False)
    axes[row,2].set_xlabel('')
    row += 1

fig_r, axes = plt.subplots(4,3)
row = 0
for band in bands :
    rep_epochs = epochs[rept_cls].crop(tmin=0, tmax=4).filter(band['fmin'],band['fmax']).average()
    single_epochs = epochs[single_cls].crop(tmin=0, tmax=4).filter(band['fmin'],band['fmax']).average()
    rest_epochs = epochs['rest'].crop(tmin=0, tmax=4).filter(band['fmin'],band['fmax']).average()
    rep_df = rep_epochs.to_data_frame()
    sg_df = single_epochs.to_data_frame()
    rest_df = rest_epochs.to_data_frame()
    for ch in broca+wernicke+auditory:
        rep_df[ch] = rep_df[ch]**2
        sg_df[ch] = sg_df[ch]**2
        rest_df[ch] = rest_df[ch]**2
    # rio_dict = dict(broca_id = broca_id, wernicke_id = wernicke_id, auditory = auditory_id)
    # rep_mean = mne.channels.combine_channels(rep_epochs,rio_dict, method='mean')
    # rest_mean = mne.channels.combine_channels(rest_epochs,rio_dict, method='mean')
    # sgl_mean = mne.channels.combine_channels(single_epochs,rio_dict, method='mean')
    st, p = stats.ttest_rel(a=rest_df['FC5'], b=rep_df['FC5'])
    if row == 0:
        title = str(broca) +' t:' + str(format(st, '.2f'))
    else:
        title = ' t:' + str(format(st, '.2f'))
    rest_df.plot(x='time',y=broca,ax=axes[row,0], title= title)
    rep_df.plot(x='time',y=broca,ax=axes[row,0], title= title)
    if row == 0 :
        axes[row,0].legend(['rest', 'repetition'])
    else:
        axes[row,0].legend('', frameon = False)
    axes[row,0].set_ylabel(band['title'])
    axes[row,0].set_xlabel('')
    # mne.viz.plot_compare_evokeds(conditions, picks = broca, title=title, combine = 'mean', show=False, axes=axes[row,0])
    st, p = stats.ttest_rel(a=rest_df['CP5'], b=rep_df['CP5'])
    if row == 0:
        title =  str(wernicke) +' t:' + str(format(st, '.2f'))
    else:
        title = ' t:' + str(format(st, '.2f'))
    rest_df.plot(x='time',y=wernicke,ax=axes[row,1], title= title)
    rep_df.plot(x='time',y=wernicke,ax=axes[row,1], title= title)
    axes[row,1].legend('', frameon = False)
    axes[row,1].set_xlabel('')
    st, p = stats.ttest_rel(a=rest_df['C5'], b=rep_df['C5'])
    if row == 0:
        title = str(auditory) +' t:' + str(format(st, '.2f'))
    else:
        title = ' t:' + str(format(st, '.2f'))
    rest_df.plot(x='time',y=auditory,ax=axes[row,2], title= title)
    rep_df.plot(x='time',y=auditory,ax=axes[row,2], title= title)
    axes[row,2].legend('', frameon = False)
    axes[row,2].set_xlabel('')
    row += 1

fig.tight_layout(pad=0.01)
fig_r.tight_layout(pad=0.01)
#qch
