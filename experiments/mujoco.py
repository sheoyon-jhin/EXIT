import common
import torch
from random import SystemRandom
import datasets
import numpy as np
import os 
import random
from parse import parse_args

from tensorboardX import SummaryWriter

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"

args = parse_args()

def main(
    weight_decay = args.weight_decay,
    manual_seed=args.seed,
    missing_rate=args.missing_rate,
    device="cuda",
    intensity=args.intensity,  # Whether to include intensity or not
    max_epochs=args.epoch,
    *,  
    model_name=args.model,
    hidden_channels=args.h_channels,
    hi_hidden_channels=args.hi_h_channels,
    hidden_hidden_channels=args.hh_channels,
    num_hidden_layers=args.layers,
    lr=args.lr,
    time_lr = args.time_lr,
    result_folder = args.result_folder,
    learn_t = args.learn_t,
    time_l = args.time_l,
    dry_run=False,
    method = args.method,
    step_mode = args.step_mode,
    time_seq=args.time_seq,
    y_seq=args.y_seq,
    **kwargs
):                                                                
    # kwargs passed on to cdeint

    batch_size = 1024
    lr = lr 
    PATH = os.path.dirname(os.path.abspath(__file__))
    
    np.random.seed(manual_seed)
    random.seed(manual_seed)
    torch.manual_seed(manual_seed)

    torch.cuda.manual_seed(manual_seed)
    torch.cuda.manual_seed_all(manual_seed)
    torch.random.manual_seed(manual_seed)
    
    # these models use the intensity for their evolution. They won't explicitly use it as an input unless we include it
    # via the use_intensity parameter, though.
    time_augment = intensity or (model_name in ('odernn_forecasting', 'dt_forecasting', 'decay_forecasting'))

    times, train_dataloader, val_dataloader, test_dataloader = datasets.mujoco.get_data(batch_size, missing_rate,time_augment, time_seq,y_seq)
    
    output_time = y_seq
    # file = '/home/bigdyl/minju_Learnable_Path/experiments/h_0/'+f'sepsis_{args.method}_{hidden_channels}_{num_hidden_layers}.npy'
    
    input_channels =  time_augment + 14 #15?
    folder_name = 'MuJoCo'
    # test_name = f'{model_name}_{hidden_channels}_{ode_hidden_hidden_channels}_{hidden_hidden_channels}_{lr}_{c1}_{c2}_{experiment_id}'
    test_name ="_".join([ str(j) for i,j in dict(vars(args)).items()])
    
    
    writer = SummaryWriter(f"{result_folder}/runs/{folder_name}/{test_name}")
    make_model = common.make_model(model_name, input_channels, 1, hidden_channels,hi_hidden_channels,hidden_hidden_channels,
                                 num_hidden_layers,use_intensity=intensity, method = args.method,kinetic_energy_coef=args.kinetic,jacobian_norm2_coef= args.jacobian,div_samples=1, initial=True, output_time=output_time)
  
    if dry_run:
        name = None
    else:
        name = 'MuJoCo'
    num_classes = 2
    return common.main(test_name, times,weight_decay, train_dataloader, val_dataloader, test_dataloader, device,
                       make_model, num_classes, max_epochs, lr,time_lr, writer, kwargs, pos_weight=torch.tensor(10),
                       step_mode=args.step_mode,learn_t=learn_t,time_l = time_l)


def run_all(intensity, device, model_names=('ncde', 'odernn', 'dt', 'decay', 'gruode')):
    model_kwargs = dict(ncde=dict(hidden_channels=49, hidden_hidden_channels=49, num_hidden_layers=4),
                        odernn=dict(hidden_channels=128, hidden_hidden_channels=128, num_hidden_layers=4),
                        dt=dict(hidden_channels=187, hidden_hidden_channels=None, num_hidden_layers=None),
                        decay=dict(hidden_channels=187, hidden_hidden_channels=None, num_hidden_layers=None),
                        gruode=dict(hidden_channels=187, hidden_hidden_channels=None, num_hidden_layers=None))
    for model_name in model_names:
        # Hyperparameters selected as what ODE-RNN did best with.
        for _ in range(5):
            main(intensity, device, model_name=model_name, **model_kwargs[model_name])
if __name__ == "__main__":
    main(method = args.method)
    
