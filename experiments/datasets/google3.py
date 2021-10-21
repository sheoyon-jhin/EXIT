import collections as co
import numpy as np
import os
import pathlib
import sktime
import sktime.utils.load_data
import torch
import urllib.request
import zipfile
import time_dataset
from . import common

here = pathlib.Path(__file__).resolve().parent



def _pad(channel, maxlen):
    # X중 하나의 데이터 들어옴 (Series) - (116,)
    channel = torch.tensor(channel) # Series를 tensor로 바꿈
    out = torch.full((maxlen,), channel[-1]) # tensor의 마지막 원소를 (maxlen,) 크기로 새로운 텐서 out을 만듦
    out[:channel.size(0)] = channel # 텐서 out의 원래 범위 만큼을 원래 값으로 채움
    return out # 리턴



def _process_data(data_path,sequence,y_seq, missing_rate, intensity):
    
    
    
    data = np.loadtxt(data_path, delimiter=",", skiprows=1)
    total_length = len(data)
    data = data[::-1]
    
    min_val = np.min(data, 0)
    max_val = np.max(data, 0) - np.min(data, 0)
    
    norm_data = time_dataset.normalize(data)
    total_length = len(norm_data)
    idx = np.array(range(total_length)).reshape(-1,1)
    norm_data = np.concatenate((norm_data,idx),axis=1)#맨 뒤에 관측시간에 대한 정보 저장

    seq_data = []
    
    for i in range(len(norm_data) - sequence -y_seq + 1): # 1은 한칸씩 밀면서 보는 것 ! 즉 시간 0 ~ 24 값 보고 그다음 텀에서 시간 1 ~ 25 이렇게 봄 
        # 총 3662개의 sequence가 들어간다. 
        x = norm_data[i : i + sequence+y_seq]
        seq_data.append(x)
    
    samples = []
    idx = torch.randperm(len(seq_data))
    for i in range(len(seq_data)):
        samples.append(seq_data[idx[i]])
    
    for j in range(len(samples)):
        if j == 0 : 
            this = torch.tensor(samples[j])
            this = torch.reshape(this,[1,this.shape[0],this.shape[1]])
        else : 
            
            tt = torch.from_numpy(np.flip(samples[j],axis=0).copy())
            this0 = torch.reshape(tt,[1,tt.shape[0],tt.shape[1]])
            this = torch.cat([this,this0]) # (3772,50,7)
            

    

    # data = time_dataset.TimeDataset(data_path,sequence)

    # train_X = train_X.to_numpy()
    # test_X = test_X.to_numpy()
    # TODO
    # X = np.concatenate((train_X, test_X), axis=0) # 2858,3  -> 
    # y = np.concatenate((train_y, test_y), axis=0)
    X = this[:,:sequence,:].float()
    y = this[:,sequence:,:].float()
    final_index = (torch.ones(X.shape[0]) * (sequence-1)).cuda() 
    final_index = final_index.long()
            
    
    times = torch.linspace(0, X.size(1) - 1, X.size(1)) # 0 ~ 181

    # missing_rate != 0 이면 missing_rate 만큼 nan 값으로 대체
    generator = torch.Generator().manual_seed(56789)
    for Xi in X:
        removed_points = torch.randperm(X.size(1), generator=generator)[:int(X.size(1) * missing_rate)].sort().values
        Xi[removed_points] = float('nan')
    print(f"missing_rate : {missing_rate}")
    X=X.cuda()
    y=y.cuda()
    times = times.cuda()
    (times, train_coeffs, val_coeffs, test_coeffs, train_y, val_y, test_y, train_final_index, val_final_index,
     test_final_index, input_channels) = common.preprocess_data_forecasting3(times, X, y, final_index, append_times=True)

    num_classes = 1

    return (times, train_coeffs, val_coeffs, test_coeffs, train_y, val_y, test_y, train_final_index, val_final_index,
            test_final_index, num_classes, input_channels)


def get_data(data_path, time_seq, y_seq, missing_rate, device, time_augment, batch_size):
    
    base_base_loc = here / 'processed_data'
    base_loc = base_base_loc / 'Stock_google_1015_plz'
    dataset_name = "google"
    # import pdb ; pdb.set_trace()
    # if time_augment:
    #     loc = base_loc / ('GOOG' + str(time_seq)+'_'+ str(y_seq) + '_' +str(missing_rate)+'_time_aug')
    # else:
    #     loc = base_loc / ('GOOG' + str(time_seq)+'_'+ str(y_seq) + '_' +str(missing_rate))
    # import pdb ; pdb.set_trace()

    loc = base_loc / (dataset_name +'_'+ str(int(missing_rate * 100)) +str(y_seq)+ ('_intensity' if time_augment else ''))
    if os.path.exists(loc):
        # import pdb ; pdb.set_trace()
        tensors = common.load_data(loc)
        times = tensors['times']
        train_coeffs = tensors['train_a'], tensors['train_b'], tensors['train_c'], tensors['train_d']
        val_coeffs = tensors['val_a'], tensors['val_b'], tensors['val_c'], tensors['val_d']
        test_coeffs = tensors['test_a'], tensors['test_b'], tensors['test_c'], tensors['test_d']
        train_y = tensors['train_y']
        val_y = tensors['val_y']
        test_y = tensors['test_y']
        train_final_index = tensors['train_final_index']
        val_final_index = tensors['val_final_index']
        test_final_index = tensors['test_final_index']
        num_classes = int(tensors['num_classes'])
        input_channels = int(tensors['input_channels'])
    else:
        
        if not os.path.exists(base_base_loc):
            os.mkdir(base_base_loc)
        if not os.path.exists(base_loc):
            os.mkdir(base_loc)
        if not os.path.exists(loc):
            os.mkdir(loc)
        (times, train_coeffs, val_coeffs, test_coeffs, train_y, val_y, test_y, train_final_index, val_final_index,
        test_final_index, num_classes, input_channels) = _process_data(data_path, time_seq, y_seq, missing_rate, time_augment)
        
        common.save_data(loc,
                         times=times,
                         train_a=train_coeffs[0], train_b=train_coeffs[1], train_c=train_coeffs[2],
                         train_d=train_coeffs[3],
                         val_a=val_coeffs[0], val_b=val_coeffs[1], val_c=val_coeffs[2], val_d=val_coeffs[3],
                         test_a=test_coeffs[0], test_b=test_coeffs[1], test_c=test_coeffs[2], test_d=test_coeffs[3],
                         train_y=train_y, val_y=val_y, test_y=test_y, train_final_index=train_final_index,
                         val_final_index=val_final_index, test_final_index=test_final_index,
                         num_classes=torch.as_tensor(num_classes), input_channels=torch.as_tensor(input_channels))
    
    

    times, train_dataloader, val_dataloader, test_dataloader = common.wrap_data(times, train_coeffs, val_coeffs,
                                                                                test_coeffs, train_y, val_y, test_y,
                                                                                train_final_index, val_final_index,
                                                                                test_final_index, device,
                                                                                num_workers=0, batch_size=batch_size)

    return times, train_dataloader, val_dataloader, test_dataloader, num_classes, input_channels

