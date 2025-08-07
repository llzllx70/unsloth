
#!/bin/bash


nohup python -m src.trainer.MyGRPOTrainer.py --task train --model Qwen3-4B-Base --step 50 > train.out 2>&1 &
