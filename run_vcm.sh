CUDA_VISIBLE_DEVICES=1 python main.py --mode train --dataset vcm  --pid_num 500 --output_path logs/vcm
CUDA_VISIBLE_DEVICES=3 python main.py --mode train --dataset bupt  --pid_num 1074 --output_path logs/bupt

CUDA_VISIBLE_DEVICES=1 python main.py --mode test --dataset vcm   --pid_num 500   --resume_test_path logs/vcm/models  --output_path logs/vcm_test
CUDA_VISIBLE_DEVICES=3 python main.py --mode test --dataset bupt  --pid_num 1074  --resume_test_path logs/bupt/models --output_path logs/bupt_test

CUDA_VISIBLE_DEVICES=1 python main.py --mode test --dataset vcm   --pid_num 1074  --resume_test_path logs/bupt/models --output_path logs/bupt2vcm_test
CUDA_VISIBLE_DEVICES=1 python main.py --mode test --dataset bupt  --pid_num 500   --resume_test_path  logs/vcm/models --output_path logs/vcm2bupt_test