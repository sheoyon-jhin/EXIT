import common
import torch
import datasets
import random
from tensorboardX import SummaryWriter 


from random import SystemRandom

import numpy as np
import os
from parse import parse_args
args = parse_args()
class InitialValueNetwork(torch.nn.Module):
    def __init__(self, intensity, hidden_channels, model):
        super(InitialValueNetwork, self).__init__()
        self.linear1 = torch.nn.Linear(7 if intensity else 5, 256)
        self.linear2 = torch.nn.Linear(256, hidden_channels)

        self.model = model

    def forward(self, times,terminal_time, coeffs, final_index, **kwargs):
        *coeffs, static = coeffs
        z0 = self.linear1(static)
        z0 = z0.relu()
        z0 = self.linear2(z0)
        return self.model(times, terminal_time,coeffs, final_index, z0=z0, **kwargs)


def main(intensity=args.intensity, weight_decay = args.weight_decay,manual_seed = args.seed,                                                              # Whether to include intensity or not
         device='cuda', max_epochs=args.epoch, pos_weight=10, *,                         # training parameters
         model_name=args.model, hidden_channels=args.h_channels,hi_hidden_channels=args.hi_h_channels, hidden_hidden_channels=args.hh_channels, num_hidden_layers=args.layers,  # model parameters
         lr = args.lr,time_lr = args.time_lr,time_l=args.time_l,result_folder = args.result_folder,learn_t = args.learn_t, dry_run=False,
         **kwargs):                                                               # kwargs passed on to cdeint
    # import pdb ; pdb.set_trace()
    batch_size = 1024
    lr = lr 
    np.random.seed(manual_seed)
    random.seed(manual_seed)
    torch.manual_seed(manual_seed)

    torch.cuda.manual_seed(manual_seed)
    torch.cuda.manual_seed_all(manual_seed)
    torch.random.manual_seed(manual_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    
    static_intensity = intensity
    # these models use the intensity for their evolution. They won't explicitly use it as an input unless we include it
    # via the use_intensity parameter, though.
    time_intensity = intensity or (model_name in ('odernn', 'dt', 'decay'))

    times, train_dataloader, val_dataloader, test_dataloader = datasets.sepsis.get_data(static_intensity,
                                                                                        time_intensity,device,
                                                                                        batch_size)

    input_channels = 1 + (1 + time_intensity) * 34
    make_model = common.make_model(model_name, input_channels, 1, hidden_channels,hi_hidden_channels,
                                   hidden_hidden_channels, num_hidden_layers, use_intensity=intensity,method = args.method,kinetic_energy_coef=args.kinetic, jacobian_norm2_coef=args.jacobian, div_samples=1,  initial=False)
    
    # import pdb ; pdb.set_trace()
    def new_make_model():
        model, regularise = make_model()
        model.linear.weight.register_hook(lambda grad: 100 * grad)
        model.linear.bias.register_hook(lambda grad: 100 * grad)
        return InitialValueNetwork(intensity, hidden_channels, model), regularise
    folder_name = 'SEPSIS'
    # import pdb ; pdb.set_trace()
    test_name = "_".join([ str(j) for i,j in dict(vars(args)).items()])
    PATH = os.path.dirname(os.path.abspath(__file__))
    writer = SummaryWriter(f"{result_folder}/runs/{folder_name}/{str(test_name)}")
    if dry_run:
        name = None
    else:
        intensity_str = '_intensity' if intensity else '_nointensity'
        name = 'sepsis' + intensity_str
    num_classes = 2
    # import pdb ; pdb.set_trace()
    return common.main(test_name, times,weight_decay,train_dataloader, val_dataloader, test_dataloader, device,
                       new_make_model, num_classes, max_epochs, lr,time_lr,writer, kwargs, pos_weight=torch.tensor(pos_weight),
                       step_mode=args.step_mode,learn_t = learn_t,time_l = time_l)


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
    main()
