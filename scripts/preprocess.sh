qsub -I -qgpuvolta  -Pik70 -lwalltime=00:30:00,ncpus=12,ngpus=1,mem=8GB,jobfs=10GB,storage=scratch/ik70,wd

export PYTHONPATH=/scratch/ik70/virtual/dainlp-2306/lib/python3.9/site-packages

module load intel-mkl/2020.3.304
module load python3/3.9.2

cd /home/599/xd2744/2311AC/code

python3 preprocess.py \
--train_filepath /home/599/xd2744/2311AC/data/sample.json \
--dev_filepath /home/599/xd2744/2311AC/data/sample.json \
--test_filepath /home/599/xd2744/2311AC/data/sample.json \
--model_dir /scratch/ik70/Corpora/flan-t5/flan-t5-small \
--cache_dir /scratch/ik70/cache_dir/2311AC/0/sample
