import os
import argparse
import numpy as np
import math
parser = argparse.ArgumentParser()
parser.add_argument('--folder', type=str)
parser.add_argument('--model', type = str )
# parser.add_argument('--metric',type =str)

# parser.add_argument('--delta_t', type = float,default = 0.25 )
args = parser.parse_args()

# PATH = "/home/bigdyl/minju_Learnable_Path/experiments/mujoco_0812/"
PATH_0 = os.path.dirname(os.path.abspath(__file__))

PATH = PATH_0 +"/"+ str(args.folder) + "/"
model = str(args.model)
files = os.listdir(PATH)
best_scores = []
from pandas import Series,DataFrame
results = DataFrame({'epoch':[],
'train_loss':[]
})
for file in files:
    if model not in file:
        continue
    else:
        f = open(PATH+file, 'r')
        lines = f.readlines()
        
        best_test_loss = math.inf
        best_epoch = 0
        # import pdb ; pdb.set_trace()
        for line in lines:

            if 'Train loss' not in line:
                continue
            else:
                
                scores = line.split()
                # import pdb ; pdb.set_trace()
                try : 
                    import pdb ; pdb.set_trace()
                    train_loss = float(scores[4])#, int(scores[1].split(']')[0])
                    epoch = int(scores[1])
                    # epochs = epoch
                    results=results.append({'epoch':epoch,
                                    'train_loss':train_loss
                                    },ignore_index=True)

                except IndexError : continue
results.to_csv(f"train_loss_google/{model}_google_result2.csv",mode='w')
    

