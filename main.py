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


def train(train_iter, model, optimizer, lr_scheduler, criterion,GRADIENT_CLIPPING=1.0):
    avg_loss = 0
    correct = 0
    total = 0

    # Iterate through batches
    for batch in tqdm(train_iter):
        inp, label = batch
        inp = inp.to('cuda')
        label = label.to('cuda')
        output = model(inp)
        # output = output.squeeze()
        loss = criterion(output, label)
        print(f"output:{output} ")
        loss.backward()
        if GRADIENT_CLIPPING > 0.0:
            nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIPPING)

        optimizer.step()
        lr_scheduler.step()
        predict = torch.where(output > 0.7, torch.tensor(1), torch.tensor(0))
        matching_ones = (predict == 1) & (label == 1)
        matching_ones = matching_ones.sum().item()
        acc = matching_ones / torch.sum(label) * 100
        # Keep track of loss and accuracy
        avg_loss += loss.item()
    return avg_loss / len(train_iter), acc


def test(test_iter, model):
    correct = 0
    total = 0

    for batch in tqdm(test_iter):
        inp, label = batch
        inp = inp.to('cuda')
        label = label.to('cuda')
        output = model(inp)
        predict = torch.where(output > 0.7, torch.tensor(1), torch.tensor(0))
        matching_ones = (predict == 1) & (label == 1)
        matching_ones = matching_ones.sum().item()
        acc = matching_ones / torch.sum(label) * 100
        # output = output.squeeze()

        # predicted = torch.where(output > 0.7)
        # total += inp.size(0)
        # correct += (predicted == torch.where(label > 0.7)).sum().item()
    return acc


def run(epochs=600, BATCH_SIZE=1):
    model = Attention_CNN(maximum_t=30, k=2975, heads=16)
    model = model.to('cuda')
    criterion = nn.BCELoss()
    optimizer = optim.Adam(params=model.parameters(), lr=1e-5, weight_decay=1e-3)
    lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda i: min(20/(i+1), 1.0))
    dataset = PCMDataSet("../drive/MyDrive/audioset_train")
    train_size = int(0.8 * len(dataset))  # 90% for training
    test_size = len(dataset) - train_size  # Remaining 10% for testing
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])


    # test_dataset_list = list(test_dataset)
    # train_dataset_list = list(train_dataset)
    # num_test_data = len(test_dataset_list)
    # num_train_data = len(train_dataset_list)
    # # # Print the number of data points in the test set
    # print("Number of data points in the test set:", num_test_data)
    # print("Number of data points in the train set:", num_train_data)
    # train_dataset = PCMDataSet('all_data')
    # test_dataset = PCMDataSet('all_eval')
    # test_dataset_list = list(test_dataset)
    # train_dataset_list = list(train_dataset)
    # num_test_data = len(test_dataset_list)
    # num_train_data = len(train_dataset_list)
    # # Print the number of data points in the test set
    # print("Number of data points in the test set:", num_test_data)
    # print("Number of data points in the train set:", num_train_data)

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
                                      criterion)
        test_acc = test(test_iter, model)
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
    torch.cuda.empty_cache()
    run()

