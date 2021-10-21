import argparse

# argparser -> model name, hidden_channels, etc
# main(intensity = True, model_name='ncde',hidden_channels=49, hidden_hidden_channels=49, num_hidden_layers=4)


def parse_args():
    parser = argparse.ArgumentParser(description='IDEA4')
    parser.add_argument('--seed', type=int, default=2021,
                        help='Seed - Test your luck!')
    parser.add_argument('--intensity', type=bool,
                        default=True, help='Intensity')
    parser.add_argument('--model', type=str, default='ncde', help='Model Name')
    parser.add_argument('--h_channels', type=int,
                        default=49, help='Hidden Channels')
    parser.add_argument('--hi_h_channels', type=int,
                        default=49, help='Hidden Channels')
    parser.add_argument('--batch',type=int,default=32)
    parser.add_argument('--hh_channels', type=int,
                        default=49, help='Hidden Hidden Channels')
    parser.add_argument('--layers', type=int, default=60,
                        help='Num of Hidden Layers')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Learning Rate')
    parser.add_argument('--kinetic', type=float,
                        help='kinetic')
    parser.add_argument('--jacobian', type=float,
                        help='jacobian')
    
    parser.add_argument('--time_lr', type=float, default=1.0,
                        help='Time_Learning Rate')
    parser.add_argument('--epoch', type=int, default=200, help='Epoch')
    
    parser.add_argument('--step_mode', type=str,
                        default='valloss', help='Model Name')
    parser.add_argument('--learn_t', type=bool,
                        default=True, help='Learning Time or not')
    parser.add_argument('--time_l', type=str,
                        default='end', help='Learning Which Time')
    # parser.add_argument('--dataset_name', type=str,
    #                     default='CharacterTrajectories', help='Dataset Name')
    parser.add_argument('--missing_rate', type=float,
                        default=0.3, help='Missing Rate')
    parser.add_argument('--weight_decay', type=float,
                        default=1e-5, help='weight_decay')
    
    # parser.add_argument('--rtol', type=float,
    #                     default=1e-11, help='ODEINT  Rtol ')
    # parser.add_argument('--atol', type=float, default=1e-11,
    #                     help='Hutchinson coefficient')
    parser.add_argument('--method', type=str, default='rk4', help='ode solver')
    parser.add_argument('--time_seq', type=int, default=50, help='time_seq')
    parser.add_argument('--y_seq', type=int, default=10, help='y_seq')
    parser.add_argument('--result_folder', type=str, default='/home/bigdyl/IDEA4/experiments/tensorboard', help='tensorboard log folder')
    # parser.add_argyment('--')
    return parser.parse_args()
