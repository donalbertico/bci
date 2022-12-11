import mne

epochs = mne.read_epochs('pilot2/cleaned2.fif')
epochs.event_id['single'] = 20
for e in epochs.events:
    if e[2] == 1 or e[2] == 4 or e[2] == 6 or e[2] == 8:
        e[2] = 20
