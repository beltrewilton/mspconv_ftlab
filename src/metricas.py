def ccc(gold, pred):
    
    import numpy as np

    gold = np.squeeze(gold)
    pred = np.squeeze(pred)
    gold_mean = np.mean(gold, axis=-1, keepdims=True)
    pred_mean = np.mean(pred, axis=-1, keepdims=True)
    covariance = np.mean((gold - gold_mean) * (pred - pred_mean), axis=-1, keepdims=True)
    gold_var = np.mean(np.square(gold - gold_mean), axis=-1, keepdims=True)
    pred_var = np.mean(np.square(pred - pred_mean), axis=-1, keepdims=True)
    ccc = 2. * covariance / (gold_var + pred_var + np.square(gold_mean - pred_mean) + np.finfo(float).eps)
    
    return ccc

def classical_features(data, sample_rate):
    import numpy as np
    import librosa
    
    # ZCR
    result = np.array([])
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
    result=np.hstack((result, zcr)) # stacking horizontally

    # Chroma_stft
    stft = np.abs(librosa.stft(data))
    chroma_stft = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
    result = np.hstack((result, chroma_stft)) # stacking horizontally

    # MFCC
    mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mfcc)) # stacking horizontally

    # Root Mean Square Value
    rms = np.mean(librosa.feature.rms(y=data).T, axis=0)
    result = np.hstack((result, rms)) # stacking horizontally

    # MelSpectogram
    mel = np.mean(librosa.feature.melspectrogram(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mel)) # stacking horizontally
    
    return result