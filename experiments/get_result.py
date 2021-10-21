import os
import argparse
import numpy as np
import math
parser = argparse.ArgumentParser()
parser.add_argument('--folder', type=str)
parser.add_argument('--model', type = str )
parser.add_argument('--metric',type =str)

# parser.add_argument('--delta_t', type = float,default = 0.25 )
args = parser.parse_args()

# PATH = "/home/bigdyl/minju_Learnable_Path/experiments/mujoco_0812/"
PATH_0 = os.path.dirname(os.path.abspath(__file__))
# import pdb ; pdb.set_trace()
PATH = PATH_0 +"/"+ str(args.folder) + "/"
model = str(args.model)
files = os.listdir(PATH)
best_scores = []

for file in files:
    if model not in file:
        continue
    else:
        f = open(PATH+file, 'r')
        lines = f.readlines()
        if args.metric == 'loss':
            best_test_loss = math.inf
            best_epoch = 0
            # import pdb ; pdb.set_trace()
            for line in lines:

                if 'Test loss' not in line:
                    continue
                else:
                    # import pdb ; pdb.set_trace()
                    if 'Train' in line:
                        continue
                    # import pdb ; pdb.set_trace()
                    else:
                        scores = line.split()
                        try : 
                            # import pdb ; pdb.set_trace()
                            test_loss = float(scores[-1])#, int(scores[1].split(']')[0])
                            if test_loss < best_test_loss: #sepsis는 best val_auc 의 test auc
                                # best_val_auc = val_auc
                                best_test_loss = test_loss
                                # best_epoch = epoch

                        except IndexError : continue
            best_scores.append((best_test_loss, file, best_epoch))
            best = sorted(best_scores, key = lambda x : x[0], reverse = False)
        if args.metric == 'auc':
            best_val_auc = 1e-10
            best_test_auc = 1e-10
            best_epoch = 0
            # import pdb ; pdb.set_trace()
            for line in lines:
                if 'Macro' in line:
                    continue
                
                else:
                    if 'Test' not in line:
                        continue
                    else:
                        # import pdb ; pdb.set_trace()
                        scores = line.split()
                        try : 
                            # import pdb ; pdb.set_trace()
                            val_auc, test_auc, epoch = float(scores[9]), float(scores[-1]), int(scores[1])
                            if test_auc > best_test_auc: #sepsis는 best val_auc 의 test auc
                                # best_val_auc = val_auc
                                best_test_auc = test_auc
                                best_epoch = epoch

                        except IndexError : continue
            best_scores.append((best_test_auc, file, best_epoch))
            best = sorted(best_scores, key = lambda x : x[0], reverse = True)
        if args.metric == 'accuracy':
            best_test_acc = 1e-10
            best_epoch = 0
            # import pdb ; pdb.set_trace()
            for line in lines:
                if 'Macro' in line:
                    continue
                
                else:
                    if 'BEST' not in line:
                        continue
                    else:
                        # import pdb ; pdb.set_trace()
                        scores = line.split()
                        try : 
                            # import pdb ; pdb.set_trace()
                            val_acc, test_acc, epoch = float(scores[10]), float(scores[-1]), int(scores[2])
                            if test_acc > best_test_acc: #sepsis는 best val_auc 의 test auc
                                # best_val_auc = val_auc
                                best_test_acc = test_acc
                                best_epoch = epoch

                        except IndexError : continue
            best_scores.append((best_test_acc, file, best_epoch))
            best = sorted(best_scores, key = lambda x : x[0], reverse = True)



print("[All Scores]")
for b in best:
    print(f"- Score : {b[0]} (Epoch : {b[2]}) \n- Param : {b[1]}\n")

if len(best) == 0:
    print(f"No File! (Model = {model})")
else:
    print("\n###############################################################################################################")
    print(f"- Best Score : {best[0][0]} (Epoch : {best[0][2]}) \n- Best Param : {best[0][1]}")
    print("###############################################################################################################\n")

top_5 = np.array([best[i][0] for i in range(5)])
print(f"Top 5 Mean : {np.mean(top_5)} Top 5 STD : {np.std(top_5)}")