from mne.io import read_raw_nirx

def load_raw(path):
    raw = read_raw_nirx(path)
    raw.annotations.rename({2.0 : 'r_pinch', 4.0 : 'r_stop', 8.0 : 'r_left', 16.0 : 'r_right'
                    128.0 : 'pause', 130.0 : 'left', 131.0 : 'work', 134.0 : 'right', 255.0 : 'rest'})
