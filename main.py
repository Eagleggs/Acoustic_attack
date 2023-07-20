import torch
from model import Attention_CNN
from wav_2_spec import wav_to_spec


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    model = Attention_CNN(maximum_t=30, k=854, heads=16)
    spec = wav_to_spec("test.wav")
    model.forward(spec)
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
