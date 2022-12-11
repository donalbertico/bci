import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import pandas as pd
import seaborn as sns
import mne
from mne.datasets import eegbci
from mne.io import concatenate_raws, read_raw_edf
from mne.time_frequency import tfr_multitaper
from mne.stats import permutation_cluster_1samp_test as pcluster_test

epochs = mne.read_epochs('pilot2\epochs_single.fif')
# sgl_bad_epochs = [0, 2, 3, 5, 9, 16, 17, 19, 20, 21, 24, 26, 40, 43, 48, 52, 55, 56, 57, 61, 68, 80, 85, 87, 97, 98, 109, 110, 118]
bads = ['Pz', 'FT8', 'P8','TP8', 'P10', 'P9', 'PO8', 'Fp1', 'AF7', 'AF3', 'Fpz', 'AF4', 'AF8', 'T8', 'F6', 'CP6', 'FC4', 'C4', 'CP4', 'Fp2']
sgl_bad_epochs = [2, 11, 33, 36, 49, 51, 65, 70]
# bads = ['Pz', 'F7', 'POz', 'FT8', 'P8', 'P10', 'P9', 'PO8', 'AF3', 'AF7', 'Fp1', 'AF4', 'AF8', 'Fp2', 'Fpz', 'T8', 'FC4', 'F8', 'F6', 'Oz', 'F5', 'CP6', 'C6']
#

epochs.drop(sgl_bad_epochs)
# epochs.plot(scalings=10e-5, n_epochs=12, events=epochs.events)
ica_s =  mne.preprocessing.read_ica('pilot2\ica_single.fif')

ica_s.apply(epochs, exclude=[0,1,3,9,16,15,14])
epochs.info['bads'] = bads


rept_cls = ['r_pinch','r_stop','r_right','r_left']
single_cls = ['right','left','pause','work']
broca = ['FC5','FC3']
auditory = ['C3','C5']
wernicke = ['CP3','CP5','P3']


bands = [dict(fmin=7, fmax=20, title='7-20Hz'),
        dict(fmin=70, fmax=80, title='70-80Hz'),
        dict(fmin=90, fmax=120, title='90-120Hz')]


freqs = np.arange(7,100)

subset = epochs.pick(['FC5','FC3','C3','C5','CP3','CP5','P3']).crop(-2,2)
kwargs = dict(n_permutations=100, step_down_p=0.05, seed=1,
              buffer_size=None, out_type='mask', n_jobs=8)

tfr = tfr_multitaper(subset, freqs=freqs, n_cycles=freqs, n_jobs=8,
        use_fft=True, return_itc=False, average = False, decim=2)
tfr.apply_baseline((-1.5,0),mode="percent")
tfr.crop(0,1.5)

aoi = [broca,auditory,wernicke]
figs = []
fg=0
# for i in aoi:
#     figs.append(plt.subplots(len(i),2))
# #
#     for ch in range(0,len(i)):

fig ,axes = plt.subplots(3,2)

chs=['FC5','FC3','C3','C5','CP3','CP5','P3']

ch=6
_, c1, p1, _ = pcluster_test(tfr['single'].data[:, ch], tail=1, **kwargs)
_, c2, p2, _ = pcluster_test(tfr['single'].data[:, ch], tail=-1, **kwargs)
_, c3, p3, _ = pcluster_test(tfr['rest'].data[:, ch], tail=1, **kwargs)
_, c4, p4, _ = pcluster_test(tfr['rest'].data[:, ch], tail=-1, **kwargs)
#
#
cs = np.stack(c1 + c2, axis=2)  # combined clusters
ps = np.concatenate((p1, p2))  # combined p-values
masks = cs[..., ps <= 0.05].any(axis=-1)

tfr['single'].average().plot([ch], cmap="RdBu",
    colorbar=False, mask=masks, show=False,
    axes=axes[2,0],
    mask_style="mask")
#
#
c = np.stack(c3 + c4, axis=2)  # combined clusters
p = np.concatenate((p3, p4))  # combined p-values
maskr = c[..., p <= 0.05].any(axis=-1)
tfr['rest'].average().plot([ch], cmap="RdBu",
    colorbar=False, mask=maskr,show=False, axes=axes[2,1],
    mask_style="mask")


axes[2,0].set_title(chs[ch])
# axes[1,1].set_title('rest')











single_tfr = tfr[single_cls].average()
single_tfr.apply_baseline((-1.3,-0.3), mode='percent')
single_tfr = single_tfr.crop(-1,1.5)
rest_tfr = tfr['rest'].average()
rest_tfr.apply_baseline((-1.2, -0.2), mode='percent')
rest_tfr = rest_tfr.crop(-1,1.5)
rep_tfr = tfr[rept_cls].average()
rep_tfr.apply_baseline((-1.2, -0.2), mode='percent')
rep_tfr = rep_tfr.crop(-1,1.5)

rest_tfr = rest_tfr.to_data_frame(time_format=None)
single_tfr = single_tfr.to_data_frame(time_format=None)

psd_fig, axes_psd = plt.subplots(7,6)

i = 4
rest_bands = list()
single_bands = list()
rep_bands = list()
p = 0
while i < len(freqs):
    single_sub_bands = list()
    rest_sub_bands = list()
    rep_sub_bands = list()
    rng = range(5)
    if i>6 and i < 13:
        rng = range(6)
    elif i>13 and i < 33:
        rng = range(10)
    elif i>33 and i <40:
        rng = range(20)
    elif i>40:
        rng = range(30)
    lg_freqs = np.logspace(*np.log10([i,i+len(rng)]), num=8)
    n_cycles = lg_freqs / 2
    s_pwr= mne.time_frequency.tfr_morlet(subset[single_cls].crop(-1,1.5), freqs=lg_freqs, n_cycles=n_cycles, use_fft=True, return_itc=False)
    r_pwr= mne.time_frequency.tfr_morlet(subset['rest'].crop(-1,1.5), freqs=lg_freqs, n_cycles=n_cycles, use_fft=True, return_itc=False)
    s_pwr.plot(baseline=(-0.9,-0.3), picks=['FC5','C5','CP5'], axes=[axes_psd[p,0],axes_psd[p,2],axes_psd[p,4]], show=False, colorbar=False)
    r_pwr.plot(baseline=(-0.9,-0.3), picks=['FC5','C5','CP5'], axes=[axes_psd[p,1],axes_psd[p,3],axes_psd[p,5]], show=False, colorbar=False)
    p += 1
    for j in rng :
        if i+j < 100 :
            single_sub_bands.append(single_tfr[single_tfr['freq'] == i+j])
            single_sub_bands[j] = single_sub_bands[j].copy().reset_index()
            rest_sub_bands.append(rest_tfr[rest_tfr['freq'] == i+j])
            rest_sub_bands[j] = rest_sub_bands[j].copy().reset_index()
    for ch in broca+wernicke+auditory:
        for j in range(1,len(single_sub_bands)) :
            single_sub_bands[0][ch] += single_sub_bands[j][ch]
            rest_sub_bands[0][ch] += rest_sub_bands[j][ch]
        single_sub_bands[0][ch] /= len(single_sub_bands)
        rest_sub_bands[0][ch] /= len(rest_sub_bands)
    single_bands.append(single_sub_bands[0].copy())
    rest_bands.append(rest_sub_bands[0].copy())
    i += len(rng)

j=0
for sx in axes_psd:
    i = 0
    for ax in sx :
        ax.set_ylabel('')
        ax.set_xlabel('')
        if i != 0 :
            ax.set_yticks([])
        if j==0 and i ==0:
            ax.set_title('       FC5')
        elif j==0 and i ==2:
            ax.set_title('       C5')
        elif j==0 and i ==4:
            ax.set_title('       CP5')
        if j != len(axes_psd) -1:
            ax.set_xticks([])
        i += 1
    j += 1


fig, axes = plt.subplots(7,3)

row = 0
for band in range(len(single_bands)):
    rest_bands[band].plot(x='time',y='FC5', ax = axes[band,0])
    single_bands[band].plot(x='time',y='FC5', ax = axes[band,0])
    st, p = stats.ttest_rel(a=rest_bands[band]['FC5'], b=single_bands[band]['FC5'])
    axes[band,0].legend('',frameon=False)
    if row == 0:
        axes[band,0].legend(['rest','single'])
        title= 'FC5 t: ' + str(format(st,'.2f'))
    else:
        title= 't: ' + str(format(st,'.2f'))
    if band == len(single_bands) -1:
        axes[band,0].set_ylabel(str(int(single_bands[band]['freq'][0]))+' - '+str(int(single_bands[band-1]['freq'][0]))+'Hz')
    else:
        axes[band,0].set_ylabel(str(int(single_bands[band]['freq'][0]))+' - '+str(int(single_bands[band+1]['freq'][0]))+'Hz')
    axes[band,0].set_title(title)
    rest_bands[band].plot(x='time',y='C5', ax = axes[band,1])
    single_bands[band].plot(x='time',y='C5', ax = axes[band,1])
    st, p = stats.ttest_rel(a=rest_bands[band]['C5'], b=single_bands[band]['C5'])
    if row == 0:
        title= 'C5 t: ' + str(format(st,'.2f'))
    else:
        title= 't: ' + str(format(st,'.2f'))
    axes[band,1].set_title(title)
    rest_bands[band].plot(x='time',y='CP5', ax = axes[band,2])
    single_bands[band].plot(x='time',y='CP5', ax = axes[band,2])
    st, p = stats.ttest_rel(a=rest_bands[band]['CP5'], b=single_bands[band]['CP5'])
    if row == 0:
        title= 'CP5 t: ' + str(format(st,'.2f'))
    else:
        title= 't: ' + str(format(st,'.2f'))
    axes[band,2].set_title(title)
    axes[band,1].legend('',frameon=False)
    axes[band,2].legend('',frameon=False)
    row += 1

fig.tight_layout(pad=0.01)
#sdf


##
#
# REP
#
##


subset = epochs.pick(broca+wernicke+auditory)



tfr = tfr_multitaper(subset, freqs=freqs, n_cycles=freqs,
        use_fft=True, return_itc=False, average = False, decim=2)
rest_tfr = tfr['rest'].average()
rest_tfr.apply_baseline((-1.2, -0.2), mode='percent')
rest_tfr = rest_tfr.crop(-1,4)
rep_tfr = tfr[rept_cls].average()
rep_tfr.apply_baseline((-1.4, -0.4), mode='percent')
rep_tfr = rep_tfr.crop(-1,4)

rest_tfr = rest_tfr.to_data_frame(time_format=None)
single_tfr = single_tfr.to_data_frame(time_format=None)

psd_fig, axes_psd = plt.subplots(7,6)

freqs = np.arange(4,100)

i = 4
rest_bands = list()
rep_bands = list()
p = 0
while i < len(freqs):
    rest_sub_bands = list()
    rep_sub_bands = list()
    rng = range(5)
    if i>6 and i < 13:
        rng = range(6)
    elif i>13 and i < 33:
        rng = range(10)
    elif i>33 and i <40:
        rng = range(20)
    elif i>40:
        rng = range(30)
    lg_freqs = np.logspace(*np.log10([i,i+len(rng)]), num=8)
    n_cycles = lg_freqs / 2
    rep_pwr= mne.time_frequency.tfr_morlet(subset[rept_cls].crop(-2,4), freqs=lg_freqs, n_cycles=n_cycles, use_fft=True, return_itc=False)
    r_pwr= mne.time_frequency.tfr_morlet(subset['rest'].crop(-2,4), freqs=lg_freqs, n_cycles=n_cycles, use_fft=True, return_itc=False)
    rep_pwr.plot(baseline=(-0.9,-0.3), picks=['FC5','C5','CP5'], axes=[axes_psd[p,0],axes_psd[p,2],axes_psd[p,4]], show=False, colorbar=False)
    r_pwr.plot(baseline=(-0.9,-0.3), picks=['FC5','C5','CP5'], axes=[axes_psd[p,1],axes_psd[p,3],axes_psd[p,5]], show=False, colorbar=False)
    p += 1
    for j in rng :
        if i+j < 100 :
            rep_sub_bands.append(rep_tfr[rep_tfr['freq'] == i+j])
            rep_sub_bands[j] = rep_sub_bands[j].copy().reset_index()
            rest_sub_bands.append(rest_tfr[rest_tfr['freq'] == i+j])
            rest_sub_bands[j] = rest_sub_bands[j].copy().reset_index()
    for ch in broca+wernicke+auditory:
        for j in range(1,len(rep_sub_bands)) :
            rep_sub_bands[0][ch] += rep_sub_bands[j][ch]
            rest_sub_bands[0][ch] += rest_sub_bands[j][ch]
        rep_sub_bands[0][ch] /= len(rep_sub_bands)
        rest_sub_bands[0][ch] /= len(rest_sub_bands)
    single_bands.append(rep_sub_bands[0].copy())
    rest_bands.append(rest_sub_bands[0].copy())
    i += len(rng)

j=0
for sx in axes_psd:
    i = 0
    for ax in sx :
        ax.set_ylabel('')
        ax.set_xlabel('')
        if i != 0 :
            ax.set_yticks([])
        if j==0 and i ==0:
            ax.set_title('       FC5')
        elif j==0 and i ==2:
            ax.set_title('       C5')
        elif j==0 and i ==4:
            ax.set_title('       CP5')
        if j != len(axes_psd) -1:
            ax.set_xticks([])
        i += 1
    j += 1


fig, axes = plt.subplots(7,3)

row = 0
for band in range(len(single_bands)):
    rest_bands[band].plot(x='time',y='FC5', ax = axes[band,0])
    single_bands[band].plot(x='time',y='FC5', ax = axes[band,0])
    st, p = stats.ttest_rel(a=rest_bands[band]['FC5'], b=single_bands[band]['FC5'])
    axes[band,0].legend('',frameon=False)
    if row == 0:
        axes[band,0].legend(['rest','repetition'])
        title= 'FC5 t: ' + str(format(st,'.2f'))
    else:
        title= 't: ' + str(format(st,'.2f'))
    if band == len(single_bands) -1:
        axes[band,0].set_ylabel(str(int(single_bands[band]['freq'][0]))+' - '+str(int(single_bands[band-1]['freq'][0]))+'Hz')
    else:
        axes[band,0].set_ylabel(str(int(single_bands[band]['freq'][0]))+' - '+str(int(single_bands[band+1]['freq'][0]))+'Hz')
    axes[band,0].set_title(title)
    rest_bands[band].plot(x='time',y='C5', ax = axes[band,1])
    single_bands[band].plot(x='time',y='C5', ax = axes[band,1])
    st, p = stats.ttest_rel(a=rest_bands[band]['C5'], b=single_bands[band]['C5'])
    if row == 0:
        title= 'C5 t: ' + str(format(st,'.2f'))
    else:
        title= 't: ' + str(format(st,'.2f'))
    axes[band,1].set_title(title)
    rest_bands[band].plot(x='time',y='CP5', ax = axes[band,2])
    single_bands[band].plot(x='time',y='CP5', ax = axes[band,2])
    st, p = stats.ttest_rel(a=rest_bands[band]['CP5'], b=single_bands[band]['CP5'])
    if row == 0:
        title= 'CP5 t: ' + str(format(st,'.2f'))
    else:
        title= 't: ' + str(format(st,'.2f'))
    axes[band,2].set_title(title)
    axes[band,1].legend('',frameon=False)
    axes[band,2].legend('',frameon=False)
    row += 1

fig.tight_layout(pad=0.01)
