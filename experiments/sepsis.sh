
for seed in  "9999" "112"
do
    for weight_decay in 1e-3 
    do 
        for lr in 0.005 
        do
            for time_lr in 1.0 
            do
                for h_channel in 49 
                do 
                    for hi_h_channel in 69 
                    do
                        for step_mode in 'valloss'
                        do
                            for layer in 3
                            do
                                for method in 'rk4' 
                                do
                                    CUDA_VISIBLE_DEVICES=0 python3 -u sepsis.py --epoch 500 --learn_t 'True' --kinetic 1.0 --jacobian 1.0 --step_mode $step_mode --method $method --time_lr $time_lr --seed $seed --weight_decay $weight_decay --model='idea4' --h_channel $h_channel --hi_h_channel $hi_h_channel --hh_channel $hi_h_channel --layer $layer --lr $lr --result_folder 'tensorboard_sepsis_both' > ../experiments/0612_{$seed}_{learn_time}_kinetic_{1.0}_{$h_channel}_{$hi_h_channel}_{$layer}_{$lr}_{$weight_decay}_{$method}_{$step_mode}_{$time_lr}.csv
                                done
                            done
                        done
                    done
                done        
            done
        done    
    done
done


#python3 -u sepsis.py --seed="112" --model='idea4' --result_folder '/home/bigdyl/IDEA4/experiments/tensorboard' --weight_decay 1e-4 --h_channel 49 --hi_h_channel 69 --hh_channel 49 --layer 4 --lr 0.001



# python3 -u sepsis.py --learn_t '' --kinetic 0 --jacobian 0 --learn_t '' --step_mode 'valloss' --method 'rk4' --time_lr 0.1 --seed="112" --weight_decay 1e-5 --model='idea4' --h_channel 49 --hi_h_channel 69 --hh_channel 49 --layer 4 --lr 0.001 --result_folder '/home/bigdyl/IDEA4/experiments/tensorboard_trash' 
# python3 -u sepsis.py --learn_t 'True' --epoch 500 --kinetic 1.0 --jacobian 1.0  --step_mode 'valloss' --method 'rk4' --time_lr 1.0 --seed="112" --weight_decay 1e-3 --model='idea4' --h_channel 49 --hi_h_channel 69 --hh_channel 69 --layer 4 --lr 0.001 --result_folder '/home/bigdyl/IDEA4/experiments/tensorboard_trash' 


CUDA_VISIBLE_DEVICES=1 python3 -u sepsis.py --epoch 500 --learn_t 'True' --kinetic 1.0 --jacobian 1.0 --step_mode 'valloss' --method rk4 --time_lr 1.0 --seed 112 --weight_decay 1e-3 --model='idea4' --h_channel 49 --hi_h_channel 69 --hh_channel 69 --layer 4 --lr 0.001 --result_folder 'tensorboard_sepsis_both'
