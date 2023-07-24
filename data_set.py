import csv
import os
import torch
from torch.utils.data import Dataset, DataLoader

import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from scipy import signal
import torch.nn.functional as F

from wav_2_spec import wav_to_spec

hashtable = {}
class PCMDataSet(Dataset):
    def __init__(self, folder_path):
        global hashtable  # Access the global hashtable
        with open("class_labels_indices.csv", 'r') as file:
            csv_reader = csv.reader(file)
            for row_index, row in enumerate(csv_reader):
                if row:  # Check if the row is not empty
                    key = row[1]  # Use the first value of the row as the key
                    hashtable[key] = row[0]  # Store the row number as the value

        self.file_paths = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if (file.endswith(".npy") and ("07qrkrw" in file or "07rpkh9" in file))]
        self.labels = [self.get_label(file) for file in os.listdir(folder_path) if (file.endswith(".npy") and ("07qrkrw" in file or "07rpkh9" in file)) ]

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, index):
        npy_ptah = self.file_paths[index]
        label = self.labels[index]
        # label = torch.ones(527)
        spec = torch.from_numpy(np.load(npy_ptah)).float()
        return spec, label

    def get_label(self, file):
        # Find the substring between "|m|" and ".wav"
        # file= file.replace("_m_", "/m/")
        # file= file.replace("_g_", "/g/")
        # file= file.replace("_t_", "/t/")
        start_index = file.find("_ _") # Add len("|m|") to skip the "|m|" part
        if start_index == -1:
            start_index = file.find("_")
        else:
            start_index = start_index+2
        end_index = file.find(".npy")
        labels = file[start_index:end_index]
        labels= labels.replace("_m_", "/m/")
        labels= labels.replace("_g_", "/g/")
        labels= labels.replace("_t_", "/t/")
        # Split the substring into three parts using ","
        split_parts = labels.split(",")
        label = torch.zeros(527)
        # a = np.zeros(527)
        for l in split_parts:
            index = int(hashtable[l])
            label[index] = 1
            # for i in range(527):
            #     a[i] = np.exp(-0.5 * ((i - index) / 5) ** 2)
            # label += torch.from_numpy(a)
        return label