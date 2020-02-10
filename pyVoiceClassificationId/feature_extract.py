import glob
import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import specgram
import soundfile as sf
from tqdm import tqdm

def extract_feature(file_name):
    """Generates feature input from audio fole (mfccs, chroma, mel, contrast, tonnetz).
       original from: https://github.com/mtobeiyf/audio-classification
       - filename : path from single audio file
    """
    X, sample_rate = sf.read(file_name, dtype='float32')
    if X.ndim > 1:
        X = X[:,0]
    X = X.T
    X = np.asfortranarray(X)
    stft = np.abs(librosa.stft(X))
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
    mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T,axis=0)
    
    return mfccs, chroma, mel, contrast, tonnetz

def get_exttracted_features(fn):
    """Returns features for individual audio file.
    +> from micah5 https://github.com/micah5/pyAudioClassification.git <+
    - fn: file name, this is return feature extraction from single audio file
    """
    try:
        mfccs, chroma, mel, contrast, tonnetz = extract_feature(fn)
        ext_features = np.hstack([mfccs, chroma, mel, contrast, tonnetz])
        return ext_features
    except Exception as e:
        print("[Error] extract feature error. %s" % (e))
        return None
    
def parse_audio_file(fn):
    """Returns features of single audio file
    +> from micah5 https://github.com/micah5/pyAudioClassification.git <+
    - fn: file name, this is return feature extraction from single audio file
    """
    features = np.empty((0,193))
    ext_features = get_ext_features(fn)
    features = np.vstack([features,ext_features])
    return np.array(features)

def parse_audio_files(parent_dir, sub_dirs, file_ext=None, verbose=True):
    """Parses directory in search of specified file types, then compiles feature data from them.
    +> from micah5 https://github.com/micah5/pyAudioClassification.git <+
    - parent_dir: parent directory from labeled audio folder
    - sub_dir: directory contain all single labeled audio file
    - file_ext: file extension type
    
    """
    # by default test for only these types
    if file_ext == None:
        file_types = ['*.ogg', '*.wav']
    else:
        file_types = []
        file_types.push(file_ext)
    features, labels = np.empty((0,193)), np.empty(0)
    for label, sub_dir in enumerate(sub_dirs):
        for file_ext in file_types:
            # file names
            iter = glob.glob(os.path.join(parent_dir, sub_dir, file_ext))
            if len(iter) > 0:
                if verbose: print('Reading', os.path.join(parent_dir, sub_dir, file_ext), '...')
                for fn in tqdm(iter):
                    ext_features = get_ext_features(fn)
                    if type(ext_features) is np.ndarray:
                        features = np.vstack([features, ext_features])
                        labels = np.append(labels, label)
    return np.array(features), np.array(labels, dtype = np.int)