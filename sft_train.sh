
#!/bin/bash


nohup python ./src/MySFTTrainer.py --task train --model Qwen3-4B-Base --step 50 > train.out 2>&1 &
