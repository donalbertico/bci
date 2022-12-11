
for e = 1:
    current = EEG.event(e).type;
    if strcmp(current,"condition 128")
        EEG.urevent(e).type = 'pause';
        if e < 129
            EEG.event(e).type = 'pause';
        end
    end
    if strcmp(current,"condition 131")
        EEG.event(e).type = 'work';
        if e < 129
          EEG.urevent(e).type = 'work';
        end
    end
    if strcmp(current,"condition 130")
        EEG.event(e).type = 'left';
        if e < 129
          EEG.urevent(e).type = 'left';
        end
    end
    if strcmp(current,"condition 133")
        EEG.event(e).type = 'right';
        if e < 129
          EEG.urevent(e).type = 'right';
        end
    end
    if strcmp(current,"condition 2")
        EEG.event(e).type = 'r_pinch';
        if e < 129
          EEG.urevent(e).type = 'r_pinch';
        end
    end
    if strcmp(current,"condition 4")
        EEG.event(e).type = 'r_stop';
        if e < 129
          EEG.urevent(e).type = 'r_stop';
        end
    end
    if strcmp(current,"condition 8")
        EEG.event(e).type = 'r_left';
        if e < 129
          EEG.urevent(e).type = 'r_left';
        end
    end
    if strcmp(current,"condition 16")
        EEG.event(e).type = 'r_right';
        if e < 129
          EEG.urevent(e).type = 'r_right';
        end
    end
    if strcmp(current,'255')
        EEG.event(e).type = 'rest';
        if e < 129
          EEG.urevent(e).type = 'rest';
        end
    end
end
