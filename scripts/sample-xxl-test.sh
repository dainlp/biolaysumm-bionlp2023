# flan-t5-xxl using 1 V100 (32G) GPU
qsub -I -qgpuvolta  -Pik70 -lwalltime=01:00:00,ncpus=12,ngpus=1,mem=384GB,jobfs=10GB,storage=scratch/ik70,wd

export PYTHONPATH=/scratch/ik70/virtual/dainlp-2306/lib/python3.9/site-packages

module load intel-mkl/2020.3.304
module load python3/3.9.2
module load cuda/11.7.0
module load cudnn/8.1.1-cuda11
module load nccl/2.17.1

cd /home/599/xd2744/2311AC/code

port=50000

output_dir=/scratch/ik70/TEMP/2311AC/saved_models
logging_dir=/scratch/ik70/TEMP/2311AC/logging_dir

model=flan-t5-xxl
batch_size=1
num_gpus=1

/scratch/ik70/virtual/dainlp-2306/bin/deepspeed --num_gpus=${num_gpus} --master_port $port ./test.py \
--model_dir /scratch/ik70/Corpora/flan-t5/${model} \
--test_dir /scratch/ik70/cache_dir/2311AC/0/sample/test \
--per_device_eval_batch_size ${batch_size} \
--output_dir $output_dir \
--fp16 \
--logging_dir $logging_dir \
--output_metrics_filepath ./test.metrics \
--output_predictions_filepath ./test.pred \
--deepspeed ./0.json

python3 inference.py