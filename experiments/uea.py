import common
import torch
import datasets
from tensorboardX import SummaryWriter
from parse import parse_args
import numpy as np 
import random

args=parse_args()
def main(dataset_name='CharacterTrajectories',weight_decay=args.weight_decay, manual_seed=args.seed,
         missing_rate=args.missing_rate, device='cuda', max_epochs=args.epoch, *, model_name=args.model,                                      # training parameters
         hidden_channels=args.h_channels,hi_hidden_channels=args.hi_h_channels, hidden_hidden_channels=args.hh_channels,
         num_hidden_layers=args.layers,lr = args.lr,time_lr=args.time_lr,time_l= args.time_l,result_folder=args.result_folder,learn_t=args.learn_t,  # model parameters
         dry_run=False,
         **kwargs):                                                               # kwargs passed on to cdeint
    # import pdb ; pdb.set_trace()
    batch_size = 32
    lr = lr
    np.random.seed(manual_seed)
    random.seed(manual_seed)
    torch.manual_seed(manual_seed)

    torch.cuda.manual_seed(manual_seed)
    torch.cuda.manual_seed_all(manual_seed)
    torch.random.manual_seed(manual_seed)
    # Need the intensity data to know how long to evolve for in between observations, but the model doesn't otherwise
    # use it because of use_intensity=False below.
    intensity_data = True if model_name in ('odernn', 'dt', 'decay') else False

    (times, train_dataloader, val_dataloader,
     test_dataloader, num_classes, input_channels) = datasets.uea.get_data(dataset_name, missing_rate, device,
                                                                           intensity=intensity_data,
                                                                           batch_size=batch_size)

    if num_classes == 2:
        output_channels = 1
    else:
        output_channels = num_classes
    folder_name = 'CharacterTrajectories'
    test_name ="_".join([ str(j) for i,j in dict(vars(args)).items()])
    
    writer = SummaryWriter(f"{result_folder}/runs/{folder_name}/{test_name}")
    make_model = common.make_model(model_name, input_channels, output_channels, hidden_channels,hi_hidden_channels, hidden_hidden_channels,
                                   num_hidden_layers, use_intensity=False, method=args.method,kinetic_energy_coef=args.kinetic,jacobian_norm2_coef=args.jacobian,div_samples=1,initial=True)
    
    
    if dry_run:
        name = None
    else:
        name = dataset_name + str(int(missing_rate * 100))
    return common.main(name, times,weight_decay, train_dataloader, val_dataloader, test_dataloader, device, make_model,
                       num_classes, max_epochs, lr,time_lr,writer, kwargs, step_mode=args.step_mode,learn_t=learn_t,time_l = time_l)


def run_all(group, device, dataset_name, model_names=('ncde', 'odernn', 'dt', 'decay', 'gruode')):
    if group == 1:
        missing_rate = 0.3
    elif group == 2:
        missing_rate = 0.5
    elif group == 3:
        missing_rate = 0.7
    else:
        raise ValueError
    model_kwargs = dict(ncde=dict(hidden_channels=32, hidden_hidden_channels=32, num_hidden_layers=3),
                        odernn=dict(hidden_channels=32, hidden_hidden_channels=32, num_hidden_layers=3),
                        dt=dict(hidden_channels=47, hidden_hidden_channels=None, num_hidden_layers=None),
                        decay=dict(hidden_channels=47, hidden_hidden_channels=None, num_hidden_layers=None),
                        gruode=dict(hidden_channels=47, hidden_hidden_channels=None, num_hidden_layers=None))
    for model_name in model_names:
        # Hyperparameters selected as what ODE-RNN did best with.
        for _ in range(5):
            main(dataset_name, missing_rate, device, model_name=model_name, **model_kwargs[model_name])
if __name__ == "__main__":
    main()