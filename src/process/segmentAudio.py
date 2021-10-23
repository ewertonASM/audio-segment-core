#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
from pydub import AudioSegment
import scipy.io.wavfile as wavfile
import process.featureExtraction as FE
import process.trainAudio as TA


def read_audio_file(input_file):
    """This function returns a numpy array that stores the audio samples of a
    specified WAV file
    """

    sampling_rate = -1
    signal = np.array([])
    try:
        audiofile = AudioSegment.from_file(input_file)
        data = np.array([])
        if audiofile.sample_width == 2:
            data = np.fromstring(audiofile._data, np.int16)
        elif audiofile.sample_width == 4:
            data = np.fromstring(audiofile._data, np.int32)

        if data.size > 0:
            sampling_rate = audiofile.frame_rate
            temp_signal = []
            for chn in list(range(audiofile.channels)):
                temp_signal.append(data[chn::audiofile.channels])
            signal = np.array(temp_signal).T
    except:
        print("Error: file not found or other I/O error. (DECODING FAILED)")

    if signal.ndim == 2 and signal.shape[1] == 1:
        signal = signal.flatten()

    return sampling_rate, signal


def smooth_moving_avg(signal, window=11):
    window = int(window)
    if signal.ndim != 1:
        raise ValueError("")
    if signal.size < window:
        raise ValueError("Input vector needs to be bigger than window size.")
    if window < 3:
        return signal
    s = np.r_[2 * signal[0] - signal[window - 1::-1],
              signal, 2 * signal[-1] - signal[-1:-window:-1]]
    w = np.ones(window, 'd')
    y = np.convolve(w / w.sum(), s, mode='same')

    return y[window:-window + 1]


def stereo_to_mono(signal):
    """
    This function converts the input signal to MONO (if it is STEREO)
    """

    if signal.ndim == 2:
        if signal.shape[1] == 1:
            signal = signal.flatten()
        else:
            if signal.shape[1] == 2:
                signal = (signal[:, 1] / 2) + (signal[:, 0] / 2)

    return signal


def silence_removal(signal, sampling_rate, st_win, st_step, smooth_window=0.5,
                    weight=0.5):

    if weight >= 1:
        weight = 0.99
    if weight <= 0:
        weight = 0.01


    signal = stereo_to_mono(signal)
    st_feats, _ = FE.feature_extraction(signal, sampling_rate,
                                        st_win * sampling_rate,
                                        st_step * sampling_rate)


    st_energy = st_feats[1, :]
    en = np.sort(st_energy)

    st_windows_fraction = int(len(en) / 10)

    low_threshold = np.mean(en[0:st_windows_fraction]) + 1e-15

    high_threshold = np.mean(en[-st_windows_fraction:-1]) + 1e-15

    low_energy = st_feats[:, np.where(st_energy <= low_threshold)[0]]

    high_energy = st_feats[:, np.where(st_energy >= high_threshold)[0]]

    features = [low_energy.T, high_energy.T]

    features_norm, mean, std = TA.normalize_features(features)
    svm = TA.train_svm(features_norm, 1.0)

    prob_on_set = []
    for index in range(st_feats.shape[1]):
        cur_fv = (st_feats[:, index] - mean) / std
        prob_on_set.append(svm.predict_proba(cur_fv.reshape(1, -1))[0][1])
    prob_on_set = np.array(prob_on_set)

    prob_on_set = smooth_moving_avg(prob_on_set, smooth_window / st_step)

    prog_on_set_sort = np.sort(prob_on_set)

    nt = int(prog_on_set_sort.shape[0] / 10)
    threshold = (np.mean((1 - weight) * prog_on_set_sort[0:nt]) +
                 weight * np.mean(prog_on_set_sort[-nt::]))

    max_indices = np.where(prob_on_set > threshold)[0]

    index = 0
    seg_limits = []
    time_clusters = []

    while index < len(max_indices):

        cur_cluster = [max_indices[index]]
        if index == len(max_indices) - 1:
            break
        while max_indices[index + 1] - cur_cluster[-1] <= 2:
            cur_cluster.append(max_indices[index + 1])
            index += 1
            if index == len(max_indices) - 1:
                break
        index += 1
        time_clusters.append(cur_cluster)
        seg_limits.append([cur_cluster[0] * st_step,
                           cur_cluster[-1] * st_step])

    min_duration = 0.2
    seg_limits_2 = []
    for s_lim in seg_limits:
        if s_lim[1] - s_lim[0] > min_duration:
            seg_limits_2.append(s_lim)
    seg_limits = seg_limits_2

    return seg_limits


def silenceRemoval(input_file, smoothing_window=1.0, weight=0.2):

    indice = 1
    snippets_audio = []

    """
    Remove silence segments from an audio file and split on those segments
    """

    if not os.path.isfile(input_file):
        raise Exception("Input audio file not found!")

    [fs, x] = read_audio_file(input_file)
    segmentLimits = silence_removal(x, fs, 0.05, 0.05, smoothing_window, weight)

    for i, s in enumerate(segmentLimits):
        strOut = "{0:s}_{1:.3f}-{2:.3f}_{3}.wav".format(input_file[0:-4], s[0], s[1], indice)
        # print(strOut
        wavfile.write(strOut, fs, x[int(fs * s[0]):int(fs * s[1])])
        snippets_audio.append(strOut)
        indice+=1
    
    return snippets_audio


