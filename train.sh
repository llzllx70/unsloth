
#!/bin/bash


nohup python ./src/MyGRPOTrainer.py --task train --model Qwen3-4B-Base > train.out 2>&1 &
