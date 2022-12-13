from mne import Epochs, pick_types, events_from_annotations
from mne.io import read_raw_nirx
from itertools import compress
from mne.preprocessing.nirs import beer_lambert_law, scalp_coupling_index, source_detector_distances, optical_density

event_dict = {'left': 1, 'right': 2, 'r_pinch': 3, 'rest': 4, 'r_stop': 5}

def load_raw(path):
    raw = read_raw_nirx(path)
    raw_od = optical_density(raw)
    sci = scalp_coupling_index(raw_od)
    raw_od.info['bads'] = list(compress(raw_od.ch_names, sci < 0.5))
    raw_hmo = beer_lambert_law(raw_od, ppf=0.1)
    raw_hmo.info['bads'] = raw_hmo.info['bads'] + ['S7_D2 hbo', 'S6_D4 hbo', 'S4_D4 hbo', 'S7_D4 hbo', 'S7_D3 hbo']
    raw_hmo = raw_hmo.filter(0.05, 0.7, h_trans_bandwidth=0.2, l_trans_bandwidth=0.02)
    event, event_dict = events_from_annotations(raw_hmo)
    reject_critaria = dict(hbo=80e-6)
    epochs = Epochs(raw_hmo, event, event_id=event_dict,
            tmin=-5, tmax=12,
            reject=reject_critaria, reject_by_annotation=True,
            proj=True, baseline=(None,0), preload=True,
            detrend=None, verbose=True)
