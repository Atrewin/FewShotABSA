#!/usr/bin/env bash
echo usage: pass gpu id list as param
echo log file path ../result/


gpu_list=$1

# Comment one of follow 2 to switch debugging status
#do_debug=--do_debug
do_debug=

# ======= dataset setting ======
support_shots_lst=(1)  # 1-shot
word_piece_data=True


# Cross evaluation's data
cross_data_id_lst=(5)  # for debug

# ====== train & test setting ======
seed_lst=(0)
#seed_lst=(10150 10151 10152 10153 10154 10155 10156 10157 10158 10159)

#lr_lst=(0.000001 0.000005 0.00005)
lr_lst=(0.00001)

clip_grad=5

decay_lr_lst=(0.5)
#decay_lr_lst=(-1)

#upper_lr_lst=(0.005 0.0005 0.0001)
upper_lr_lst=(0.001)

fix_embd_epoch_lst=(-1)
#fix_embd_epoch_lst=(1 2)

#warmup_epoch=-1
warmup_epoch=0



train_batch_size_lst=(1)
test_batch_size=4
grad_acc=1
#grad_acc=4  # if the GPU-memory is not enough, use bigger gradient accumulate
epoch=1

# ==== model setting =========
# ---- encoder setting -----

#embedder=bert
embedder=sep_bert

#emission_lst=(mnet)
#emission_lst=(proto_with_label)
emission_lst=(proto)
#emission_lst=(mnet proto)

similarity=dot

emission_normalizer=none
#emission_normalizer=softmax
#emission_normalizer=norm

#emission_scaler=none
#emission_scaler=fix
emission_scaler=learn
#emission_scaler=relu
#emission_scaler=exp

do_div_emission=-dbt
#do_div_emission=

ems_scale_rate_lst=(0.01)
#ems_scale_rate_lst=(0.01 0.02 0.05 0.005)

label_reps=sep
#label_reps=cat

ple_normalizer=none
ple_scaler=fix
ple_scale_r=0.5
#ple_scale_r=1
#ple_scale_r=0.01

emb_log=

# ------ decoder setting -------
#decoder_lst=(rule)
decoder_lst=(sms)


# ======= default path (for quick distribution) ==========
bert_base_uncased=/home/cike/hui/Pre-BERTs/bert-base-uncased/uncased_L-12_H-768_A-12
bert_base_uncased_vocab=/home/cike/hui/Pre-BERTs/bert-base-uncased/uncased_L-12_H-768_A-12/vocab.txt
base_data_dir=./processed_data/v3


echo [START] set jobs  on gpu [ ${gpu_list} ]
# === Loop for all case and run === 2
for seed in ${seed_lst[@]}
do
     for support_shots in ${support_shots_lst[@]}
     do
          for train_batch_size in ${train_batch_size_lst[@]}
          do
              for decay_lr in ${decay_lr_lst[@]}
              do
                  for fix_embd_epoch in ${fix_embd_epoch_lst[@]}
                  do
                      for lr in ${lr_lst[@]}
                      do
                          for upper_lr in ${upper_lr_lst[@]}
                          do
                              for ems_scale_r in ${ems_scale_rate_lst[@]}
                              do
                                  for emission in ${emission_lst[@]}
                                  do
                                      for decoder in ${decoder_lst[@]}
                                      do
                                          for cross_data_id in ${cross_data_id_lst[@]}
                                          do
                                              # model names
 #                                            model_name=FewShotSA.dec_${decoder}.enc_${embedder}.ems_${emission}.lb_${label_reps}_scl_${ple_scaler}${ple_scale_r}.sim_${similarity}.lr_${lr}.up_lr_${upper_lr}.bs_${train_batch_size}_${test_batch_size}.sp_b_${grad_acc}.w_ep_${warmup_epoch}.ep_${epoch}${do_debug}
                                              model_name=FewShotSA
                                              data_dir=${base_data_dir}/
                                              file_mark=cross_id_${cross_data_id}.seed_${seed}
                                              train_file_name=train_${cross_data_id}.json
                                              dev_file_name=test_${cross_data_id}.json
                                              test_file_name=test_${cross_data_id}.json
                                              trained_model_path=${data_dir}${model_name}.DATA.${file_mark}/model.path

                                              echo [CLI]
                                              echo Model: ${model_name}
                                              echo Task:  ${file_mark}
                                              echo [CLI]
                                              export OMP_NUM_THREADS=2  # threads num for each task

                                              CUDA_VISIBLE_DEVICES=${gpu_list} PYTHONIOENCODING=utf-8 python3 -W ignore main.py ${do_debug} \
                                                  --seed ${seed} \
                                                  --do_train \
                                                  --do_predict \
                                                  --train_path ${data_dir}${train_file_name} \
                                                  --dev_path ${data_dir}${dev_file_name} \
                                                  --test_path ${data_dir}${test_file_name} \
                                                  --output_dir ./outputs_models/${model_name}.DATA.${file_mark} \
                                                  --bert_path ${bert_base_uncased} \
                                                  --bert_vocab ${bert_base_uncased_vocab} \
                                                  --train_batch_size ${train_batch_size} \
                                                  --cpt_per_epoch 4 \
                                                  --delete_checkpoint \
                                                  --gradient_accumulation_steps ${grad_acc} \
                                                  --num_train_epochs ${epoch} \
                                                  --learning_rate ${lr} \
                                                  --decay_lr ${decay_lr} \
                                                  --upper_lr ${upper_lr} \
                                                  --clip_grad ${clip_grad} \
                                                  --fix_embed_epoch ${fix_embd_epoch} \
                                                  --warmup_epoch ${warmup_epoch} \
                                                  --test_batch_size ${test_batch_size} \
                                                  --context_emb ${embedder} \
                                                  --label_reps ${label_reps} \
                                                  --emission ${emission} \
                                                  --similarity ${similarity} \
                                                  --e_nm ${emission_normalizer} \
                                                  --e_scl ${emission_scaler} \
                                                  --ems_scale_r ${ems_scale_r} \
                                                  --ple_nm ${ple_normalizer} \
                                                  --ple_scl ${ple_scaler} \
                                                  --ple_scale_r ${ple_scale_r} \
                                                  --allow_override \
                                                  ${emb_log} \
                                                  ${do_div_emission} \
                                                  --decoder ${decoder} \
                                                  --load_feature \
#                                                  | tee ./result/${model_name}.DATA.${file_mark}.log
                                              echo [CLI]
                                              echo Model: ${model_name}
                                              echo Task:  ${file_mark}
                                              echo [CLI]
                                          done
                                      done
                                  done
                              done
                          done
                      done
                  done
              done
          done
     done
done


echo [FINISH] set jobs on gpu [ ${gpu_list} ]
