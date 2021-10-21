import common
import datasets

from tensorboardX import SummaryWriter 
from parse import parse_args
args = parse_args()

def main(device='cuda', weight_decay = args.weight_decay,manual_seed = args.seed,
        max_epochs=args.epoch, *, model_name=args.model, 
        hidden_channels=args.h_channels, hi_hidden_channels=args.hi_h_channels,hidden_hidden_channels=args.hh_channels, 
        num_hidden_layers=args.layers,lr=args.lr,time_lr=args.time_lr,time_l = args.time_l,result_folder= args.result_folder,learn_t = args.learn_t,
        dry_run=False,
        **kwargs):                                                               # kwargs passed on to cdeint

    batch_size = 1024
    lr = lr #0.00005 * (batch_size / 32)

    intensity_data = True if model_name in ('odernn', 'dt', 'decay') else False
    times, train_dataloader, val_dataloader, test_dataloader = datasets.speech_commands.get_data(intensity_data,
                                                                                                 batch_size)
    input_channels = 1 + (1 + intensity_data) * 20
    folder_name = 'Speech_Commands'
    test_name = "_".join([ str(j) for i,j in dict(vars(args)).items()])
    writer = SummaryWriter(f"{result_folder}/runs/{folder_name}/{test_name}")
    make_model = common.make_model(model_name, input_channels, 10, hidden_channels,hi_hidden_channels, hidden_hidden_channels,
                                   num_hidden_layers, use_intensity=False,method = args.method,kinetic_energy_coef=args.kinetic,jacobian_norm2_coef=args.jacobian, div_samples=1,initial=True)

    def new_make_model():
        model, regularise = make_model()
        model.linear.weight.register_hook(lambda grad: 100 * grad)
        model.linear.bias.register_hook(lambda grad: 100 * grad)
        return model, regularise

    name = None if dry_run else 'speech_commands'
    num_classes = 10
    return common.main(test_name, times,weight_decay, train_dataloader, val_dataloader, test_dataloader, device, new_make_model,
                       num_classes, max_epochs, lr,time_lr,writer, kwargs, step_mode=args.step_mode,learn_t=learn_t,time_l = time_l)


def run_all(device, model_names=('ncde', 'odernn', 'dt', 'decay', 'gruode')):
    model_kwargs = dict(ncde=dict(hidden_channels=90, hidden_hidden_channels=40, num_hidden_layers=4),
                        odernn=dict(hidden_channels=128, hidden_hidden_channels=64, num_hidden_layers=4),
                        dt=dict(hidden_channels=160, hidden_hidden_channels=None, num_hidden_layers=None),
                        decay=dict(hidden_channels=160, hidden_hidden_channels=None, num_hidden_layers=None),
                        gruode=dict(hidden_channels=160, hidden_hidden_channels=None, num_hidden_layers=None))
    for model_name in model_names:
        # Hyperparameters selected as what ODE-RNN did best with.
        for _ in range(5):
            main(device, model_name=model_name, **model_kwargs[model_name])

if __name__ =="__main__":
    main()