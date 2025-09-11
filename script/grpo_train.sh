
#!/bin/bash


# nohup python -m src.trainer.MyGRPOTrainer --task train --model Qwen3-4B-Base --step 5000 > grpo_train.out 2>&1 &

python -m src.trainer.MyGRPOTrainer --task train --model Qwen3-4B-Base --step 5000
