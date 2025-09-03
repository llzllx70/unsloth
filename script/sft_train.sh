
#!/bin/bash


# nohup python -m src.trainer.MySFTTrainer --task train --model Qwen3-4B-Base --step 50 > sft_train.out 2>&1 &

python -m src.trainer.MySFTTrainer --task train --model Qwen3-4B-Base --step 10
