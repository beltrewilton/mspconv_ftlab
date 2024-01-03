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