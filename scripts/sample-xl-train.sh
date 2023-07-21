# xl version using 2 V100 (32G) GPU
qsub -I -qgpuvolta  -Pik70 -lwalltime=00:30:00,ncpus=24,ngpus=2,mem=256GB,jobfs=10GB,storage=scratch/ik70,wd

export PYTHONPATH=/scratch/ik70/virtual/dainlp-2306/lib/python3.9/site-packages

module load intel-mkl/2020.3.304
module load python3/3.9.2
module load cuda/11.7.0
module load cudnn/8.1.1-cuda11
module load nccl/2.17.1

cd /home/599/xd2744/2311AC/code

rm -r /scratch/ik70/TEMP/2311AC
mkdir -p /scratch/ik70/TEMP/2311AC

lr=1e-5
port=50000


running_id=lr_${lr}
output_dir=/scratch/ik70/TEMP/2311AC/saved_models
logging_dir=/scratch/ik70/TEMP/2311AC/logging_dir

# xl version
model=flan-t5-xl
batch_size=1
num_gpus=2

/scratch/ik70/virtual/dainlp-2306/bin/deepspeed --num_gpus=${num_gpus} --master_port $port ./train.py \
--model_dir /scratch/ik70/Corpora/flan-t5/${model} \
--train_dir /scratch/ik70/cache_dir/2311AC/0/sample/train \
--per_device_train_batch_size ${batch_size} \
--output_dir $output_dir \
--logging_dir $logging_dir \
--output_metrics_filepath ./train.metrics \
--num_train_epochs 2 \
--seed 52 \
--deepspeed ./0.json

python3 $output_dir/zero_to_fp32.py $output_dir $output_dir/pytorch_model.bin

# xl version
model=flan-t5-xl
batch_size=1
num_gpus=1

/scratch/ik70/virtual/dainlp-2306/bin/deepspeed --num_gpus=${num_gpus} --master_port $port ./test.py \
--model_dir /scratch/ik70/Corpora/flan-t5/${model} \
--test_dir /scratch/ik70/cache_dir/2311AC/0/sample/test \
--per_device_eval_batch_size ${batch_size} \
--output_dir $output_dir \
--logging_dir $logging_dir \
--output_metrics_filepath ./test.metrics \
--output_predictions_filepath ./test.pred \
--deepspeed ./0.json

rm -r $output_dir