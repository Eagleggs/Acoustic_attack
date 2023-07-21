import wave
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
import torch


def wav_to_spec(input_wav_file):
    with wave.open(input_wav_file, 'rb') as wav_file:
        # Check if the WAV file is in PCM format
        if wav_file.getsampwidth() != 2:
            raise ValueError("WAV file should be in 16-bit PCM format.")

        # Get audio parameters
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        frame_rate = wav_file.getframerate()
        frames = wav_file.getnframes()
        # Read audio data from the WAV file
        pcm_data = wav_file.readframes(frames)
        pcm_data = np.frombuffer(pcm_data, dtype=np.int16)
        slice_length = channels * frame_rate  # 1s per segment
        remainder = len(pcm_data) % slice_length # pad the output
        if remainder != 0:
            pad_length = slice_length - remainder
            pcm_data = np.pad(pcm_data, (0, pad_length), mode='constant', constant_values=0)
        pcm_slices = np.array([pcm_data[i:i + slice_length] for i in range(0, len(pcm_data), slice_length)])

        freq, t, stft = signal.spectrogram(pcm_slices, fs=44100, mode='magnitude', nperseg=800, noverlap=100, nfft=1000)
        np.save(input_wav_file[:-4], stft)
        # for i in range(0,10):
        #     plt.pcolormesh(t, freq, abs(stft[i]), shading='gouraud')
        #     plt.title('Spectrogramm using STFT amplitude')
        #     plt.ylabel('Frequency [Hz]')
        #     plt.xlabel('Time [seconds]')
        #     plt.show()
        stft = torch.from_numpy(stft).float()
        return stft
