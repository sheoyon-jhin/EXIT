import os 
import numpy as np
import random
import torch
import torch.nn as nn
from torch.optim import optimizer
import torch.utils.data as data_utils

import models
from parse import parse_args


def inf_generator(iterable):
    """Allows training with DataLoaders in a single infinite loop:
    for i, (x, y) in enumerate(inf_generator(train_loader)):
    """
    iterator = iterable.__iter__()
    while True:
        try:
            yield iterator.__next__()
        except StopIteration:
            iterator = iterable.__iter__()


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

args = parse_args()

np.random.seed(args.seed)
random.seed(args.seed)
torch.manual_seed(args.seed)

# if you are using GPU
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.random.manual_seed(args.seed)

batch_size = 1024

# dataset_name = args.dataset_name
PATH0 = os.path.dirname(os.path.abspath(__file__))
# import pdb ; pdb.set_trace()
PATH = PATH0+'/datasets/google_baseline/0/'

train_X = torch.load(PATH + 'train_X.pt')
val_X = torch.load(PATH + 'val_X.pt')
test_X = torch.load(PATH + 'test_X.pt')
train_y = torch.load(PATH + 'train_y.pt')
val_y = torch.load(PATH + 'val_y.pt')
test_y = torch.load(PATH + 'test_y.pt')

train = data_utils.TensorDataset(train_X, train_y)
valid = data_utils.TensorDataset(val_X, val_y)
test = data_utils.TensorDataset(test_X, test_y)

train_loader = data_utils.DataLoader(
train, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False
)


if __name__ == "__main__":
    
    device = torch.device(
        "cuda:" + str(0) if torch.cuda.is_available() else "cpu"
    )
    baseline = torch.cuda.max_memory_allocated(device)
    if args.model == 'rnn': # RNN
        model = models.RNN_forecasting(input_dim=7, hidden_dim=args.h_channels, num_layers=args.layers, output_dim=7)
    elif args.model == 'lstm': # LSTM
        model = models.LSTM_forecasting(input_dim=7, hidden_dim=args.h_channels, num_layers=args.layers, output_dim=7)
    elif args.model == 'gru': # GRU
        model = models.GRU_forecasting(input_dim=7, hidden_dim=args.h_channels, num_layers=args.layers, output_dim=7)
    else:
        print("Error: wrong model name")

    model = model.to(device)
    print(model)

    criterion = nn.MSELoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    data_gen = inf_generator(train_loader)
    batches_per_epoch = len(train_loader)
    loss_sum = 0
    min_train_mse = float("inf")
    min_val_mse = float("inf")
    min_test_mse = float("inf")
    best_epoch = 0
    plateau_terminate =50
    for itr in range(args.epoch * batches_per_epoch):
        x, y = data_gen.__next__()
        x = x.to(device)
        y = y.to(device)
        # import pdb ;pdb.set_trace()
        output = model(x)
        
        loss = criterion(output[:,:,:-1], y[:,:,:-1])
        loss_sum += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if itr % batches_per_epoch == 0:
            train_loss = loss_sum / batches_per_epoch
            loss_sum = 0

            with torch.no_grad():
                val_X = val_X.to(device)
                val_y = val_y.to(device)
                preds = model(val_X)
                # import pdb; pdb.set_trace()
                val_loss = criterion(preds[:,:,:-1], val_y[:,:,:-1])
                val_loss = val_loss.item()
                
                if val_loss*1.001 < min_val_mse:
                    min_val_mse = val_loss

                    test_X = test_X.to(device)
                    test_y = test_y.to(device)
                    preds = model(test_X)
                    test_loss = criterion(preds[:,:,:-1], test_y[:,:,:-1])
                    test_loss = test_loss.item()
                    if test_loss*1.001 < min_test_mse:
                        min_test_mse = test_loss
                    print('Epoch {}, Train Loss {:.5f}, Val Loss {:.5f} Test Loss {:.5f}'.format(itr // batches_per_epoch, train_loss, val_loss,test_loss))
            if train_loss * 1.001 < min_train_mse:
                    best_epoch = itr // batches_per_epoch
                
            print('Epoch {}, Train Loss {:.5f}, Val Loss {:.5f}'.format(itr // batches_per_epoch, train_loss, val_loss))
            if itr // batches_per_epoch > best_epoch + plateau_terminate:
                print(f"breaking! at Epoch {itr // batches_per_epoch}")
                break
    usage = torch.cuda.max_memory_allocated(device)
    memory_usage = usage - baseline 
    print("Total Memory usage is {}".format(memory_usage))
    print('Min test mse {:.5f} at Epoch {}'.format(min_test_mse, best_epoch))