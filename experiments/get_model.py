import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--folder', type=str)
parser.add_argument('--date', type = str )
parser.add_argument('--data', type = str )
parser.add_argument('--time', type = str )
# parser.add_argument('--delta_t', type = float,default = 0.25 )
args = parser.parse_args()
date = args.date
data = args.data

time =args.time
# PATH = "/home/bigdyl/minju_Learnable_Path/experiments/mujoco_0812/"
PATH_0 = os.path.dirname(os.path.abspath(__file__))
# import pdb ; pdb.set_trace()
PATH = PATH_0 +"/"+ str(args.folder) + "/"

files = os.listdir(PATH)
best_scores = []

from pandas import Series,DataFrame
results = DataFrame({'seed':[],
'learn_time':[],
'model_name':[],
'hidden_size':[],
'hhidden_size':[],
'layers':[],
'lr':[],
'kinetic':[],
'time_lr':[],
'weight_decay':[],
'METRICS':[],
'epoch':[]
})

for file in files:
    
    if data not in file:
        continue
    else:
        ff_ = os.listdir(PATH+file)
        for ff in ff_:
            
            if len(ff)>0:
                # import pdb ;pdb.set_trace()
                f_file = file.split('_')
                results=results.append({'seed':f_file[0],
                                    'learn_time':f_file[1],
                                    'model_name':f_file[2],
                                    'hidden_size':f_file[3],
                                    'hhidden_size':f_file[4],
                                    'layers':f_file[6],
                                    'lr':f_file[7],
                                    'kinetic':f_file[8],
                                    'time_lr':f_file[10],
                                    'weight_decay':f_file[15],
                                    'METRICS':ff.split('_')[-1].split('.pt')[0],
                                    'epoch':ff.split('_')[1]
                                    },ignore_index=True)
results.to_csv(f"{data}_result_{date}_{time}.csv",mode='w')
        
#     best_val_auc = 1e-10
#     best_test_auc = 0
#     best_epoch = 0
#     # import pdb ; pdb.set_trace()
#     for line in lines:
#         if 'Test' not in line:
#             continue
#         else:
#             # import pdb ; pdb.set_trace()
#             scores = line.split()
#             try : 
#                 # import pdb ; pdb.set_trace()
#                 val_auc, test_auc, epoch = float(scores[5].split(',')[0]), float(scores[-1]), scores[1]
#                 if val_auc > best_val_auc: #sepsis는 best val_auc 의 test auc
#                     best_val_auc = val_auc
#                     best_test_auc = test_auc
#                     best_epoch = epoch

#             except IndexError : continue
#     best_scores.append((best_test_auc, file, best_epoch))

# best = sorted(best_scores, key = lambda x : x[0], reverse = True)

# print("[All Scores]")
# for b in best:
#     print(f"- Score : {b[0]} (Epoch : {b[2]}) \n- Param : {b[1]}\n")

# if len(best) == 0:
#     print(f"No File! (Model = {model})")
# else:
#     print("\n###############################################################################################################")
#     print(f"- Best Score : {best[0][0]} (Epoch : {best[0][2]}) \n- Best Param : {best[0][1]}")
#     print("###############################################################################################################\n")