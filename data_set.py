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

        self.file_paths = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith(".wav")]
        self.labels = [self.get_label(file) for file in os.listdir(folder_path) if file.endswith(".wav") ]

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, index):
        wav_path = self.file_paths[index]
        label = self.labels[index]
        spec = wav_to_spec(wav_path)

        return spec, label

    def get_label(self, file):
        # Find the substring between "|m|" and ".wav"
        start_index = file.find("|")  # Add len("|m|") to skip the "|m|" part
        end_index = file.find(".wav")

        labels = file[start_index:end_index]
        labels= labels.replace("|", "/")
        # Split the substring into three parts using ","
        split_parts = labels.split(",")
        label = torch.zeros(527)
        for l in split_parts:
            index = int(hashtable[l])
            label[index]= 1
        return label