import numpy as np
import pandas as pd

def ccc(gold, pred):

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


def get_training_features_simple(feature_function, feature_function_parameters : dict, frame_duration: float, start: float, step: float, df: pd.DataFrame) -> pd.DataFrame:
    """
        Inputs:
            -frame_duration: Función usada para la extracción de features
            -start: tiempo inicial de la ventana movil
            -step: paso al que avanaza la función movil
            -df:
            -feature_function

        Output:
            Dataframe consisting of 3 columns: Time, Arousal, Dominance, Valence.
            The Arousal, Dominance and Valence columns represent the mean vote calculated for corresponding Time in the audio.
    """

    end = df.iloc[-1]['Time']
    
    X, Y = [], [] 
    
    while start + 2.5 < end:
        
        df_frame = df[(df['Time'] >= start) & (df['Time'] <= start + frame_duration)]
        
        feature = feature_function(df['Data'].values, feature_function_parameters['sr'])
        emotion = df_frame.groupby('Emotion').count().sort_values(by = 'Time', ascending = False).reset_index().loc[0,'Emotion']
        a, d, v = df_frame[['Arousal','Dominance','Valence']].mean()

        X.append(feature)
        Y.append([a,v,d,emotion])
         
        start += step
        print(start, '/', end)
        
    df_features = pd.DataFrame(X)
    df_features[['Arousal','Dominance','Valence','Emotion']] = Y
    
    return df_features