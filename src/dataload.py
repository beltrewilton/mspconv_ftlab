import pandas as pd
import numpy as np
import librosa

def load_audio_data(df_annotations: pd.DataFrame, pc_num: int, part_num: int) -> (np.ndarray, int):

    """
        Inputs:
            -df_annotations: Dataset annotations directory. For every file contains contains a row with the name, emotion, annotator, podcast part and number.
            -pc_num: PodCast Number
            -part_num: Part 

        Output:
            1- A numpy array with the audio time series
            2- Integer sampling rate
    """
    
    df_part_data = df_annotations[(df_annotations['PC_Num'] == pc_num) & (df_annotations['Part_Num'] == part_num)].reset_index()

    audio_name = df_part_data['Audio_Name'][0]
    start_time = df_part_data['start_time'][0]
    end_time = df_part_data['end_time'][0]

    audio_path = "data/MSPCORPUS/Audio/" + audio_name

    return librosa.load(audio_path, offset = start_time, duration = end_time - start_time)

def audio_select_mean_vote(df_annotations: pd.DataFrame, pc_num: int, part_num: int) -> pd.DataFrame:

    """
        Inputs:
            -df_annotations: Dataset annotations directory. For every file contains contains a row with the name, emotion, annotator, podcast part and number.
            -pc_num: PodCast Number
            -part_num: Part 

        Output:
            Dataframe consisting of 3 columns: Time, Arousal, Dominance, Valence.
            The Arousal, Dominance and Valence columns represent the mean vote calculated for corresponding Time in the audio.
    """

    votation_means = pd.DataFrame(columns = ['Time','Vote','Emotion'])
    emotions = ['Valence','Arousal','Dominance']

    for emotion in emotions:
        
        time = pd.DataFrame(columns = ['Time','Annotation','Annotator'])    
        df_copy = df_annotations[(df_annotations['PC_Num'] == pc_num) & (df_annotations['Part_Num'] == part_num) & (df_annotations.Emotion == emotion)]
        
        for name, annotator, emot in zip(df_copy['Name'], df_copy['Annotator'], df_copy['Emotion']):
        
            temp_df = pd.read_csv(f'data/MSPCORPUS/Annotations/{emot}/{name}', skiprows=9, header=None, names=['Time', 'Annotation'])
            temp_df['Annotator'] = annotator
            time = pd.concat([time, temp_df], ignore_index = True)
            
        df_pivot = pd.DataFrame(time.pivot_table(columns = 'Annotator', index = 'Time', values = 'Annotation').to_records()).set_index('Time')
        df_pivot = df_pivot.fillna(method='ffill')
        df_pivot['Vote'] = df_pivot.mean(axis = 1)
        df_pivot['Emotion'] = emotion
        
        votation_means = pd.concat([votation_means, df_pivot.reset_index()[['Time','Vote','Emotion']]])
        
    df_emotions_vote = pd.DataFrame(votation_means.pivot_table(columns = 'Emotion', index = 'Time', values = 'Vote').to_records()).set_index('Time')
    df_emotions_vote = df_emotions_vote.fillna(method='ffill')
    
    return df_emotions_vote.reset_index()