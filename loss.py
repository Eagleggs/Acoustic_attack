import torch
import torch.nn.functional as F

def cosine_similarity_loss(output,label):
    dot_product = torch.dot(output.flatten(), label.flatten())
    norm1 = torch.linalg.norm(output)
    norm2 = torch.linalg.norm(label)

    acc = dot_product / (norm1 * norm2)
    acc = 1 - acc
    return acc