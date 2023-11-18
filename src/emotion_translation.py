from src.utils import ekmans_thresholds

def ekman_emotion(valence : float, arousal : float, dominance : float) -> str:

    best_emotion = None
    min_distance = float('inf')

    # Calculate the Euclidean distance from the VAD values to each emotion's threshold
    for emotion, (v_thresh, a_thresh, d_thresh) in ekmans_thresholds.items():
        distance = (((valence / 100) - v_thresh) ** 2 + ((arousal / 100) - a_thresh) ** 2 + ((dominance / 100) - d_thresh) ** 2) ** 0.5 
        
        # Update the best emotion if the current emotion is closer
        if distance < min_distance:
            best_emotion = emotion
            min_distance = distance 
    
    return best_emotion