import torch
import numpy as np
from scipy.io.wavfile import write
from pydub import AudioSegment
from pprint import pprint
torch.set_num_threads(1)

USE_ONNX = False
model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                              model='silero_vad',
                              force_reload=False,  # TODO ajustar
                              onnx=USE_ONNX)


def chunk(msp_wave: str, CUT_NEAR: int = 3, SAMPLING_RATE: int = 16_000, silence_threshold: float = 0.100) -> dict:
    (get_speech_timestamps,
     save_audio,
     read_audio,
     VADIterator,
     collect_chunks) = utils

    wav = read_audio(msp_wave, sampling_rate=SAMPLING_RATE)
    # audio "contaminacion" da tiempo malos, ver parametros de get_speech_timestamps
    # get_speech_timestamps(
    #     wav,
    #     vad_models[pid],
    #     0.46,  # speech prob threshold
    #     16000,  # sample rate
    #     300,  # min speech duration in ms
    #     20,  # max speech duration in seconds
    #     600,  # min silence duration
    #     512,  # window size
    #     200,  # spech pad ms
    # )
    # def get_speech_timestamps(audio: torch.Tensor,
    #                           model,
    #                           threshold: float = 0.5,
    #                           sampling_rate: int = 16000,
    #                           min_speech_duration_ms: int = 250,
    #                           max_speech_duration_s: float = float('inf'),
    #                           min_silence_duration_ms: int = 100,
    #                           window_size_samples: int = 512,
    #                           speech_pad_ms: int = 30,
    #                           return_seconds: bool = False,
    #                           visualize_probs: bool = False,
    #                           progress_tracking_callback: Callable[[float], None] = None):
    # ('audio', 'model', 'threshold', 'sampling_rate', 'min_speech_duration_ms', 'max_speech_duration_s',
    #  'min_silence_duration_ms', 'window_size_samples', 'speech_pad_ms', 'return_seconds', 'visualize_probs',
    #  'progress_tracking_callback', 'i', 'step', 'min_speech_samples', 'speech_pad_samples', 'max_speech_samples',
    #  'min_silence_samples', 'min_silence_samples_at_max_speech', 'audio_length_samples', 'speech_probs',
    #  'current_start_sample', 'chunk', 'speech_prob', 'progress', 'progress_percent', 'triggered', 'speeches',
    #  'current_speech', 'neg_threshold', 'temp_end', 'prev_end', 'next_start', 'speech', 'silence_duration',
    #  'speech_dict')
    speech_timestamps = get_speech_timestamps(wav, model,
                                              sampling_rate=SAMPLING_RATE,
                                              min_silence_duration_ms=100
                                              # max_speech_duration_s=3, # fuerza bruta.! no resulta :(
                                              # return_seconds=True, # hace redondeo no adecuado
                                              )
    start, end, z = 0, 0, 0
    jumps = {}
    for i, t in enumerate(speech_timestamps):
        start = t['start']/SAMPLING_RATE
        silence = start - end
        end = t['end']/SAMPLING_RATE
        jumps[i] = {'silence':silence, 'start': start, 'end': end, 'cut': False}
        tolerance = end - z
        if tolerance >= CUT_NEAR and silence >= silence_threshold:
                                               # (silence >= silence_threshold
                                               #   or get the next end_of_sentence whit stable_ts
                                               #   in a given range of time, ex. 15 seconds)
            z = end
            jumps[i]['cut'] = True

    cuts = {}
    cuts[0] = {'start': jumps[0]['start'], 'end': jumps[0]['end'], 'length': -1, 'silence': jumps[0]['silence'], 'cut': jumps[0]['cut']}
    idx = 0
    for i in range(1, len(jumps)):
        if jumps[0]['cut'] and i == 1:
            cuts[idx]['length'] = cuts[idx]['end'] - cuts[idx]['start']
            idx += 1

        if jumps[i]['cut'] and jumps[i - 1]['cut']:
            # q[idx - 1]['end'] = a[i - 1]['end']
            cuts[idx] = {'start': jumps[i]['start'], 'end': -1, 'length': -1, 'silence': jumps[i]['silence']}
            # idx += 1

        if jumps[i]['cut']:
            cuts[idx]['end'] = jumps[i]['end']
            cuts[idx]['length'] = cuts[idx]['end'] - cuts[idx]['start']
            idx += 1 # define un grupo 1, 2, 3, 4, .... N

        if jumps[i-1]['cut'] and not jumps[i]['cut']:
            cuts[idx] = {'start': jumps[i]['start'], 'end': -1, 'length': -1, 'silence': jumps[i]['silence']}
            idx = len(cuts) - 1

    if cuts[len(cuts) - 1]['end'] == -1:
        cuts[len(cuts) - 1]['end'] = jumps[i]['end']
        cuts[len(cuts) - 1]['length'] = cuts[len(cuts) - 1]['end'] - cuts[len(cuts) - 1]['start']

    return cuts


def create_silence(duration, sr=24_000):
    # Generate an array of zeros representing silence
    silence = np.zeros(int(duration * sr))
    return silence


def split_audio(cuts: dict, input_file: str):
    audio = AudioSegment.from_wav(input_file)

    for cut in cuts.items():
        i, cut = cut[0], cut[1]
        start_time_ms = cut['start'] * 1000
        end_time_ms = cut['end'] * 1000
        segmented_audio = audio[start_time_ms:end_time_ms]
        # Export the segmented audio to a new file
        output_file = input_file.replace('.wav', f'_{i}.wav')
        segmented_audio.export(output_file, format="wav")
        silence_file = input_file.replace('.wav', f'-silence--{i}.wav')
        silence = create_silence(cut['silence'])
        write(silence_file, 24_000, silence)


if __name__ == "__main__":
    msp_wave = "/Users/beltre.wilton/Downloads/SER-Datasets/MSP-Conversation-1.1/Audio/MSP-Conversation_0830.wav"
    cuts = chunk(msp_wave)
    pprint(cuts)
    # split_audio(cuts, volcals)