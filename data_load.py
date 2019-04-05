# -*- coding: utf-8 -*-
# /usr/bin/python2

import glob
import random

import librosa
import numpy as np
from tensorpack.dataflow.base import RNGDataFlow
from tensorpack.dataflow.common import BatchData
from tensorpack.dataflow import PrefetchData
from audio import read_wav, preemphasis, amp2db
from hparam import hparam as hp
from utils import normalize_0_1

class DataFlowForConvert(RNGDataFlow):

    def __init__(self, data_path):
        self.wav_file = data_path
        self.batch_size = 1

    def __call__(self, n_prefetch=1, n_thread=1):
        df = self
        df = BatchData(df, 1)
        df = PrefetchData(df, n_prefetch, n_thread)
        return df

    def get_data2(self):
        while True:
            yield get_mfccs_and_spectrogram(self.wav_file, isConverting=True, trim=False)
        

class DataFlow(RNGDataFlow):

    def __init__(self, data_path, batch_size):
        self.batch_size = batch_size
        self.wav_files = glob.glob(data_path)

    def __call__(self, n_prefetch=1000, n_thread=1):
        df = self
        df = BatchData(df, self.batch_size)
        df = PrefetchData(df, n_prefetch, n_thread)
        return df


class Net1DataFlow(DataFlow):

    def get_data(self):

        while True:
            wav_file = random.choice(self.wav_files)
            npz_file = wav_file.replace("WAV","npz")
            #yield get_mfccs_and_phones(wav_file)
            yield read_mfccs_and_phones(npz_file)
    def size(self):
        return hp.train1.batch_size


class Net2DataFlow(DataFlow):

    def __init__(self, data_path, batch_size):
        self.batch_size = batch_size
        npz_path = data_path + '/npz/*.npz'
        self.npz_files = glob.glob(npz_path)

    def get_data(self):
        while True:
            npz_file = random.choice(self.npz_files)
            #print(npz_file)
            yield read_mfccs_and_spectrogram(npz_file)

"""
def load_data(mode):
    wav_files = glob.glob(getattr(hp, mode).data_path)

    return wav_files
"""

def wav_random_crop(wav, sr, duration):
    assert (wav.ndim <= 2)

    target_len = sr * duration
    wav_len = wav.shape[-1]
    start = np.random.choice(range(np.maximum(1, wav_len - target_len)), 1)[0]
    end = start + target_len
    if wav.ndim == 1:
        wav = wav[start:end]
    else:
        wav = wav[:, start:end]
    return wav


def get_mfccs_and_phones(wav_file, trim=False, random_crop=True):

    '''This is applied in `train1` or `test1` phase.
    '''

    # Load
    wav = read_wav(wav_file, sr=hp.default.sr)

    mfccs, _, _ = _get_mfcc_and_spec(wav, hp.default.preemphasis, hp.default.n_fft,
                                     hp.default.win_length,
                                     hp.default.hop_length)

    # timesteps
    num_timesteps = mfccs.shape[0]

    # phones (targets)
    phn_file = wav_file.replace("WAV", "PHN")
    phn2idx, idx2phn = load_vocab()
    phns = np.zeros(shape=(num_timesteps,))
    bnd_list = []
    for line in open(phn_file, 'r').read().splitlines():
        start_point, _, phn = line.split()
        bnd = int(start_point) // hp.default.hop_length
        phns[bnd:] = phn2idx[phn]
        bnd_list.append(bnd)

    # Trim
    if trim:
        start, end = bnd_list[1], bnd_list[-1]
        mfccs = mfccs[start:end]
        phns = phns[start:end]
        assert (len(mfccs) == len(phns))

    # Random crop
    n_timesteps = (hp.default.duration * hp.default.sr) // hp.default.hop_length + 1
    if random_crop:
        start = np.random.choice(range(np.maximum(1, len(mfccs) - n_timesteps)), 1)[0]
        end = start + n_timesteps
        mfccs = mfccs[start:end]
        phns = phns[start:end]
        assert (len(mfccs) == len(phns))

    # Padding or crop
    mfccs = librosa.util.fix_length(mfccs, n_timesteps, axis=0)
    phns = librosa.util.fix_length(phns, n_timesteps, axis=0)

    # Padding with first and second derivative of mfcc
    mfccs = get_first_and_second_derivative(mfccs)    
    return mfccs, phns

def get_first_and_second_derivative(mfccs):
    if mfccs.shape[-1] < 5:
        return
    f_mfccs = np.zeros(mfccs.shape)
    s_mfccs = np.zeros(mfccs.shape)

    f_mfccs[:,0] = (-mfccs[:,2]+4*mfccs[:,1]-3*mfccs[:,0])/2
    f_mfccs[:,1] = (-mfccs[:,3]+4*mfccs[:,2]-3*mfccs[:,1])/2
    f_mfccs[:,-1] = (3*mfccs[:,-1]-4*mfccs[:,-2]+mfccs[:,-3])/2
    f_mfccs[:,-2] = (3*mfccs[:,-2]-4*mfccs[:,-3]+mfccs[:,-4])/2

    s_mfccs[:,0] = -mfccs[:,3]+4*mfccs[:,2]-5*mfccs[:,1]+2*mfccs[:,0]
    s_mfccs[:,1] = -mfccs[:,4]+4*mfccs[:,3]-5*mfccs[:,2]+2*mfccs[:,1]
    s_mfccs[:,-1] = -mfccs[:,-4]+4*mfccs[:,-3]-5*mfccs[:,-2]+2*mfccs[:,-1]
    s_mfccs[:,-2] = -mfccs[:,-5]+4*mfccs[:,-4]-5*mfccs[:,-3]+2*mfccs[:,-2]
    
    
    for i in range(2,mfccs.shape[-1]-2):
        f_mfccs[:,i] = (-mfccs[:,i+2]+8*mfccs[:,i+1]-8*mfccs[:,i-1]+mfccs[:,i-2])/12
        s_mfccs[:,i] = (-mfccs[:,i+2]+16*mfccs[:,i+1]-30*mfccs[:,i]+16*mfccs[:,i-1]-mfccs[:,i-2])/12
    
    mfccs = np.concatenate((mfccs,f_mfccs,s_mfccs),axis=1)
    return mfccs
    
if __name__ =='__main__':
    mfccs, phns = get_mfccs_and_phones("/home/cocoonmola/datasets/TIMIT2/DR3/FCMG0/SA1.WAV")
    print(mfccs.shape)
    print(phns.shape)
    
def read_mfccs_and_phones(npz_file):
    np_arrays = np.load(npz_file)

    mfccs = np_arrays['mfccs']
    phns = np_arrays['phns']

    np_arrays.close()

    return mfccs, phns


def get_mfccs_and_spectrogram(wav_file, trim=True, random_crop=False, isConverting=False):
    '''This is applied in `train2`, `test2` or `convert` phase.
    '''


    # Load
    wav, _ = librosa.load(wav_file, sr=hp.default.sr)

    # Trim
    if trim:
        wav, _ = librosa.effects.trim(wav, frame_length=hp.default.win_length, hop_length=hp.default.hop_length)

    if random_crop:
        wav = wav_random_crop(wav, hp.default.sr, hp.default.duration)

    
    # Padding or crop if not Converting
    if isConverting is False:
        length = int(hp.default.sr * hp.default.duration)
        wav = librosa.util.fix_length(wav, length)

    return _get_mfcc_and_spec(wav, hp.default.preemphasis, hp.default.n_fft, hp.default.win_length, hp.default.hop_length)


# TODO refactoring
def _get_mfcc_and_spec(wav, preemphasis_coeff, n_fft, win_length, hop_length):

    # Pre-emphasis
    y_preem = preemphasis(wav, coeff=preemphasis_coeff)

    # Get spectrogram
    D = librosa.stft(y=y_preem, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    mag = np.abs(D)

    # Get mel-spectrogram
    mel_basis = librosa.filters.mel(hp.default.sr, hp.default.n_fft, hp.default.n_mels)  # (n_mels, 1+n_fft//2)
    mel = np.dot(mel_basis, mag)  # (n_mels, t) # mel spectrogram

    # Get mfccs, amp to db
    mag_db = amp2db(mag)
    mel_db = amp2db(mel)
    mfccs = np.dot(librosa.filters.dct(hp.default.n_mfcc, mel_db.shape[0]), mel_db)

    # Normalization (0 ~ 1)
    mag_db = normalize_0_1(mag_db, hp.default.max_db, hp.default.min_db)
    mel_db = normalize_0_1(mel_db, hp.default.max_db, hp.default.min_db)

    return mfccs.T, mag_db.T, mel_db.T  # (t, n_mfccs), (t, 1+n_fft/2), (t, n_mels)


def read_mfccs_and_spectrogram(npz_file):
    np_arrays = np.load(npz_file)

    mfccs = np_arrays['mfccs']
    mag_db = np_arrays['mag_db']
    mel_db = np_arrays['mel_db']

    np_arrays.close()

    return mfccs, mag_db, mel_db
    


phns = ['h#', 'aa', 'ae', 'ah', 'ao', 'aw', 'ax', 'ax-h', 'axr', 'ay', 'b', 'bcl',
        'ch', 'd', 'dcl', 'dh', 'dx', 'eh', 'el', 'em', 'en', 'eng', 'epi',
        'er', 'ey', 'f', 'g', 'gcl', 'hh', 'hv', 'ih', 'ix', 'iy', 'jh',
        'k', 'kcl', 'l', 'm', 'n', 'ng', 'nx', 'ow', 'oy', 'p', 'pau', 'pcl',
        'q', 'r', 's', 'sh', 't', 'tcl', 'th', 'uh', 'uw', 'ux', 'v', 'w', 'y', 'z', 'zh']


def load_vocab():
    phn2idx = {phn: idx for idx, phn in enumerate(phns)}
    idx2phn = {idx: phn for idx, phn in enumerate(phns)}

    return phn2idx, idx2phn


