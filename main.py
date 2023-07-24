import numpy as np
import torch
from model import Attention_CNN
from wav_2_spec import wav_to_spec
from torch import optim
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split
from data_set import  PCMDataSet
from loss import cosine_similarity_loss

def train(train_iter, model, optimizer, lr_scheduler, criterion,device,GRADIENT_CLIPPING=1.0):
    avg_loss = 0
    correct = 0
    total = 0

    # Iterate through batches
    for batch in tqdm(train_iter):
        inp, label = batch
        inp = inp.to(device)
        label = label.to(device)
        output = model(inp)
        loss = criterion(output, label)
        loss += cosine_similarity_loss(output,label)
        loss.backward()
        if GRADIENT_CLIPPING > 0.0:
            nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIPPING)

        optimizer.step()
        lr_scheduler.step()
        dot_product = torch.dot(output.flatten(), label.flatten())
        norm1 = torch.linalg.norm(output)
        norm2 = torch.linalg.norm(label)

        acc = dot_product / (norm1 * norm2) * 100
        # acc =torch.cosine_similarity(torch.zeros(30,30),torch.ones(30,30))
        # predict = torch.where(output < 0.1, torch.tensor(0), torch.tensor(1))
        # matching_ones = (predict == 1) & (label == 1)
        # matching_ones = matching_ones.sum().item()
        # acc = matching_ones / torch.sum(label) * 100
        # flat_tensor1 = predict.flatten().int()
        # flat_tensor2 = label.flatten().int()
        # # Calculate the intersection (common elements where both tensors have 1s)
        # intersection = sum(flat_tensor1 & flat_tensor2)
        #
        # # Calculate the union (total unique positions where at least one tensor has 1)
        # union = sum(flat_tensor1 | flat_tensor2)
        #
        # # Calculate the Jaccard similarity coefficient
        # similarity = intersection / union * 100

        # Keep track of loss and accuracy
        avg_loss += loss.item()
        print(f"loss:{loss.item()},similarity:{acc.item()}")
        # print(f"\n {output}")
    return avg_loss / len(train_iter), acc.item()


def test(test_iter, model,device):
    correct = 0
    total = 0

    for batch in tqdm(test_iter):
        inp, label = batch
        inp = inp.to(device)
        label = label.to(device)
        output = model(inp)
        output = torch.where(output>0.6,torch.tensor(1).float(),torch.tensor(0).float())
        print(output)
        dot_product = torch.dot(output.flatten(), label.flatten())
        norm1 = torch.linalg.norm(output)
        norm2 = torch.linalg.norm(label)

        acc = dot_product / (norm1 * norm2) * 100
        # predict = torch.where(output > 0.1, torch.tensor(1), torch.tensor(0))
        # matching_ones = (predict == 1) & (label == 1)
        # matching_ones = matching_ones.sum().item()
        # acc = matching_ones / torch.sum(label) * 100
        # flat_tensor1 = predict.flatten().int()
        # flat_tensor2 = label.flatten().int()
        # # Calculate the intersection (common elements where both tensors have 1s)
        # intersection = sum(flat_tensor1 & flat_tensor2)
        #
        # # Calculate the union (total unique positions where at least one tensor has 1)
        # union = sum(flat_tensor1 | flat_tensor2)
        #
        # # Calculate the Jaccard similarity coefficient
        # similarity = intersection / union

        # Keep track of loss and accuracy
        print(f"acc:{acc}")
        # output = output.squeeze()

        # predicted = torch.where(output > 0.7)
        # total += inp.size(0)


        # correct += (predicted == torch.where(label > 0.7)).sum().item()
    return acc.item()


def run(device,epochs=600, BATCH_SIZE=10):
    model = Attention_CNN(maximum_t=30, k=1024, heads=4)
    model = model.to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(params=model.parameters(), lr=4e-4, weight_decay=1e-3)
    lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda i: min(30/(i+1), 1.0))
    # dataset = PCMDataSet("../drive/MyDrive/audioset_train")
    dataset = PCMDataSet("/media/lu/B6AEFCF5AEFCAECD/dataset/audioset_train")
    train_size = int(0.99 * len(dataset))  # 90% for training
    test_size = len(dataset) - train_size  # Remaining 10% for testing
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])


    train_iter = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_iter = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)
    best_test_acc = 0
    best_train_acc = 0
    patience = 0  # early stopping
    # Training loop
    for epoch in range(epochs):
        print(f'\n Epoch {epoch}')

        # Train on data
        train_loss, train_acc = train(train_iter,
                                      model,
                                      optimizer,
                                      lr_scheduler,
                                      criterion,device)

        test_acc = test(test_iter, model,device)
        print(f'\n training loss:{train_loss},training acc:{train_acc}')
        print(f'\nFinished.test acc:{test_acc}')
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            patience = 0
            torch.save(model.state_dict(), 'model_best_1.pt')
        else:
            if train_acc > best_train_acc:
                best_train_acc = train_acc
                patience += 1
                if patience > 15:
                    break
    torch.save(model.state_dict(), 'model_final_1.pt')

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    device = 'cuda'
    torch.cuda.empty_cache()
    run(device=device)

