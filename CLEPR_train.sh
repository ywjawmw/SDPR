#!/bin/bash
mess_dropouts=('[0.0,0.0]')
adj_types=('norm')
embed_sizes=(32)
lrs=(2e-5)
regs=('[7e-3]')
co_lamdas=(1e-6)
max_step_lens=(26)
steps=(6)
ts=(0.3)

for mess_dropout in ${mess_dropouts[@]};
do
    for reg in ${regs[@]};
    do
        for adj_type in ${adj_types[@]};
        do
            for embed_size in ${embed_sizes[@]};
            do
                for lr in ${lrs[@]};
                do
                  for co_lamda in ${co_lamdas[@]};
                  do
                    for step in ${steps[@]};
                    do
                      for max_step_len in ${max_step_lens[@]};
                      do
                        for t in ${ts[@]};
                        do
                          python clepr_main.py --fusion 'add' --test_file 'valid_id.txt' --gpu_id 0 --result_index 1  --layer_size [64,128] --mlp_layer_size [128] --dataset Herb --regs $reg  --embed_size $embed_size --alg_type 'CLEPR' --adj_type  $adj_type --lr $lr --save_flag 1  --pretrain 0 --t $t --batch_size 512 --epoch 2000 --verbose 1  --mess_dropout  $mess_dropout --model_wpath "date_2022-09-18_60_ori_emb_seed1234" --two_save_tail 'loss_512_col-neg-sample-train' --seed 1234 --attention 1 --co_lamda $co_lamda --step $step --max_step_len $max_step_len --hard_neg 1 --use_S1 1;
                        done
                      done
                    done
                  done
                done
            done
        done
    done
done
