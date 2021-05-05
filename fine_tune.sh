#!/usr/bin/env bash
set -eux
function assign_value(){
  var_name=$1
  value=$2
  script_path=$3
  if [[ ! -f ${script_path} ]];then
	echo "${script_path} does not exist"
	exit 1
  fi
  if [[ $(grep -c "${var_name}" "${script_path}") -eq 0 ]];then
	echo "pattern:${var_name} not found in ${script_path}"
	exit 1
  fi
  if [ ${var_name} == "lr" ];then
     sed -i "s/lr\", type=float, default=.*/lr\", type=float, default=${value})/g" ${script_path}
  elif [ ${var_name} == "batch_size" ];then
     sed -i "s/batch_size\", type=int, default=.*/batch_size\", type=int, default=${value})/g" ${script_path}
  elif [ ${var_name} == "num_epochs" ];then
     sed -i "s/num_epochs\", type=int, default=.*/num_epochs\", type=int, default=${value})/g" ${script_path}
  else
     exit 1 
  fi
 # sed -i 's/ ${var_name}", type=int, default=.* / ${var_name}", typr=int, default=${value}/g ' ${script_path}
 # sed -i 's/ ${var_name}", type=float, default=.* / ${var_name}",type=float, default=${value}/g' ${script_path}
}
function run(){
  lr=$1
  num_epochs=$2
  batch_size=$3
  RUN="./mtb_baseline_main_task1_1.py" 
  assign_value "lr" ${lr} ${RUN}
  assign_value "num_epochs" ${num_epochs} ${RUN}
  assign_value "batch_size" ${batch_size} ${RUN}
  echo "start running lr: ${lr} batch_size: ${batch_size} num_epochs: ${num_epochs}">> ./result.txt
  CUDA_VISIBLE_DEVICES=5 python mtb_baseline_main_task1_1.py
  echo -e "\n">> ./result.txt
  sleep 15
}

for batch_size in 1 4 8;do
  for lr in 1e-5 3e-5 ;do
    for num_epochs in 15 20;do
      rm -rf ./my_model/task_train_accuracy_per_epoch_3.pkl
      rm -rf ./my_model/task_test_model_best_3.pth.tar
      rm -rf ./my_model/task_test_losses_per_epoch_3.pkl
      rm -rf ./my_model/task_test_f1_per_epoch_3.pkl
      rm -rf ./my_model/task_test_checkpoint_3.pth.tar   
      run ${lr} ${num_epochs} ${batch_size}
    done
  done
done

