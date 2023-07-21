import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch

# Use multi-head attention to encode the frequency knowledge in the spectrogram
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, k, heads=8):
        super(MultiHeadSelfAttention, self).__init__()
        self.heads = heads
        self.to_keys = nn.Linear(k, k * heads, bias=False)
        self.to_queries = nn.Linear(k, k * heads, bias=False)
        self.to_values = nn.Linear(k, k * heads, bias=False)

        self.unifyheads = nn.Linear(k * heads, k)

    def forward(self, x):
        b, t, k = x.size()
        h = self.heads
        queries = self.to_queries(x).view(b, t, h, k)
        keys = self.to_keys(x).view(b, t, h, k)
        values = self.to_values(x).view(b, t, h, k)

        keys = keys.transpose(1, 2).reshape(b * h, t, k)
        queries = queries.transpose(1, 2).reshape(b * h, t, k)
        values = values.transpose(1, 2).reshape(b * h, t, k)

        w_prime = torch.bmm(queries, keys.transpose(1, 2))
        w_prime = w_prime / (k ** (1 / 2))
        w = F.softmax(w_prime, dim=2)

        y = torch.bmm(w, values).view(b, h, t, k)
        y = y.transpose(1, 2).reshape(b, t, h * k)

        y = self.unifyheads(y)

        return y  # output [b,t,k]


class Attention_CNN(nn.Module):
    def __init__(self, maximum_t, k, heads):
        self.maximum_t = maximum_t# max 30s
        super(Attention_CNN, self).__init__()
        self.position = nn.Embedding(self.maximum_t,k)
        self.attentionlayer = MultiHeadSelfAttention(k, heads=heads)
        self.layernorm = nn.LayerNorm(k)
        self.do = nn.Dropout(0.2)
        self.feature_extraction = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=13,stride=1, padding_mode='zeros'),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=9,stride=1, padding_mode='zeros'),
            nn.ReLU(),
            nn.Conv2d(in_channels=8, out_channels=4, kernel_size=3,stride=1, padding_mode='zeros'),
            nn.ReLU(),
            nn.Conv2d(in_channels=4, out_channels=2, kernel_size=3,stride=2, padding_mode='zeros'),
            nn.ReLU(),
            nn.Conv2d(in_channels=2, out_channels=1, kernel_size=3,stride=2, padding_mode='zeros')
        )
        self.classification =nn.Sequential(
            nn.Linear(k, 521, bias=True),
            nn.Sigmoid(),
        )

    def forward(self,x):
        # CNN feature extraction
        b, s, k, t = x.shape
        slices = torch.chunk(x, chunks=s, dim=1)
        extracted_slices = [self.feature_extraction(chunk) for chunk in slices]
        extracted_slices = torch.cat(extracted_slices,dim=1)
        extracted_features = torch.flatten(extracted_slices,start_dim=2)
        batch,slice,feature = extracted_features.shape

        # Attention mechanism
        p = torch.arange(self.maximum_t, device=x.device).view(1, self.maximum_t).expand(b, self.maximum_t) # positional embedding
        p = self.position(p)
        x = extracted_features + p[:,:slice,:]
        y = self.attentionlayer(x)
        x = x + y
        x = self.do(x)
        x = self.layernorm(x)

        # Classification network
        y = self.classification(x)
        y = torch.sum(y,dim=1)
        y = y / 10
        return y
