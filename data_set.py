import os
import torch
from torch.utils.data import Dataset, DataLoader

import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from scipy import signal
import torch.nn.functional as F


class PCMDataSet(Dataset):
    def __init__(self, folder_path):
        self.file_paths = [os.path.join(folder_path, file) for file in os.listdir(folder_path)]
        self.labels = [self.get_label(file) for file in os.listdir(folder_path)]

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, index):
        npz_path = self.file_paths[index]
        label = self.labels[index]
        # with open(npz_path, 'rb') as f:
        #     stft_data = np.load(npz_path)
        #
        # hi = highpassfilter();
        # pcm_data = hi.butter_highpass_filter(pcm_data, 2000, 63333) #filter the human speaking noise
        # index_f = 0
        # amplitude = 10000
        # while index_f > 16000 or index_f < 7000:
        #     index_f = np.argmax(pcm_data > amplitude)
        #     amplitude += 100
        #     if amplitude > 30000:
        #         break
        # max = np.max(pcm_data[index_f:index_f + 3000])
        # if np.max(pcm_data) != max:
        #     print(np.max(pcm_data) )
        # indices = np.where(pcm_data > max - 5000)[0]
        # indices = indices[indices < index_f + 3000]
        # id = np.max(indices)
        # waveform = torch.from_numpy(pcm_data.copy()[id:id + 3000]).float()
        # freq, t, stft = signal.spectrogram(waveform, fs=63333, mode='magnitude',nperseg=10,noverlap=1,nfft = 400)
        # t = stft_data['arr_0']
        # stft = torch.from_numpy(t).float()
        # print(stft.shape)

        # waveform = F.normalize(waveform, p=2.0, dim=0, eps=1e-12, out=None)
        # waveform = waveform.unsqueeze(1)
        # time_index = torch.arange(waveform.shape[0]).unsqueeze(1)
        # waveform = torch.cat((waveform, time_index), dim=1)
        # print(waveform)
        return torch.rand(10,501,125), label

    def get_label(self, file):
        o = torch.zeros(521)
        o[0] = 1
        o[1] = 1
        return o
