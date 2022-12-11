dict = readtable('ch_dic.csv')
fields = fieldnames(dict)
biosemi = {EEG.chanlocs.labels}

for ch = 1:64
    EEG.chanlocs(ch).labels = dict.(fields{ch}){1}
end