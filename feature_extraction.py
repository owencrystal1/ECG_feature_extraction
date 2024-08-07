import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
import re
import scipy.signal as signal
import os
import neurokit2 as nk
import pickle
import biosppy.signals.ecg as ecg
from tqdm import tqdm
from datetime import datetime
from datetime import timedelta



def extract_features(data, fs, leads):

    def is_nan(value):
        try:
            return np.isnan(float(value))
        except (ValueError, TypeError):
            return False
    
    if leads == 'all':
        leads = ['I','II','III','AvR','AvL','AvF','V1','V2','V3','V4','V5','V6']
    else:
        leads = leads 
        
    all_feat = []
    for lead in leads:
        sample_data = data[lead]
        clean_ecg = nk.ecg_clean(sample_data, sampling_rate=fs, method="pantompkins1985")
        _, rpeaks = nk.ecg_peaks(clean_ecg, sampling_rate=fs)

        # determine P, Q, S, T peaks and bounds 
        signals, waves = nk.ecg_delineate(clean_ecg, rpeaks, method="dwt",sampling_rate=500)

        def fill_nan_with_zero(d):
            for key, value in d.items():
                if isinstance(value, dict):  # If value is a nested dictionary, recurse into it
                    fill_nan_with_zero(value)
                elif isinstance(value, float) and math.isnan(value):  # Check if value is NaN
                    d[key] = 0
            return d

        waves = fill_nan_with_zero(waves)
        rpeaks = fill_nan_with_zero(rpeaks)

        #  amplitudes:
        P_amp = np.mean(clean_ecg[[x for x in waves["ECG_P_Peaks"] if not is_nan(x)]])
        Q_amp = np.mean(clean_ecg[[x for x in waves["ECG_Q_Peaks"] if not is_nan(x)]])
        R_amp = np.mean(clean_ecg[[x for x in rpeaks["ECG_R_Peaks"] if not is_nan(x)]])
        S_amp = np.mean(clean_ecg[[x for x in waves["ECG_S_Peaks"] if not is_nan(x)]])
        T_amp = np.mean(clean_ecg[[x for x in waves["ECG_T_Peaks"] if not is_nan(x)]])

        PR_ints = np.array(waves["ECG_R_Onsets"]) - np.array(waves["ECG_P_Onsets"]) # P-R Interval = R_onset - P_onset
        PR_ints = [x for x in PR_ints if not np.isnan(x)]
        PR_int = np.mean(PR_ints)/fs

        PR_segs = np.array(waves["ECG_R_Onsets"]) - np.array(waves["ECG_P_Offsets"]) # P-R Segment = R_onset - P_offset
        PR_segs = [x for x in PR_segs if not np.isnan(x)]
        PR_seg = np.mean(PR_segs)/fs


        ST_ints = np.array(waves["ECG_T_Offsets"]) - np.array(waves["ECG_R_Offsets"]) # 
        ST_ints = [x for x in ST_ints if not np.isnan(x)]
        ST_int = np.mean(ST_ints)/fs

        ST_segs = np.array(waves["ECG_T_Onsets"]) - np.array(waves["ECG_R_Offsets"]) # 
        ST_segs = [x for x in ST_segs if not np.isnan(x)]
        ST_seg = np.mean(ST_segs)/fs

        QT_ints = np.array(waves["ECG_T_Offsets"]) - np.array(waves["ECG_R_Onsets"]) # 
        QT_ints = [x for x in QT_ints if not np.isnan(x)]
        QT_int = np.mean(QT_ints)/fs

        # calculate difference between current T offset to next Q onset
        TQ_ints = [(np.array(waves["ECG_R_Onsets"]))[i+1] - (np.array(waves["ECG_T_Offsets"]))[i] for i in range(len(waves["ECG_T_Offsets"]) - 1)]
        TQ_int = np.mean([x for x in TQ_ints if not np.isnan(x)])/fs

        QRS_durs = np.array(waves["ECG_R_Offsets"]) - np.array(waves["ECG_R_Onsets"])
        QRS_durs = [x for x in QRS_durs if not np.isnan(x)]
        QRS_dur = np.mean(QRS_durs)/fs

        R_peaks = np.array(rpeaks["ECG_R_Peaks"])
        R_peaks = [x for x in R_peaks if not np.isnan(x)]
        RR_int = np.mean(np.diff(R_peaks))/fs

        feat12 = [P_amp, Q_amp, R_amp, S_amp, T_amp, PR_int, PR_seg, ST_int, ST_seg, QT_int, TQ_int, QRS_dur, RR_int]
        all_feat = all_feat + feat12

        #unique_features = ['P_amp', 'Q_amp', 'R_amp', 'S_amp', 'T_amp','PR_int','PR_seg','ST_int','ST_seg','QT_int','TQ_int','QRS','RR_int']


    return all_feat