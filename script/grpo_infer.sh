
#!/bin/bash


nohup python ./src/trainer/MyGRPOTrainer.py --task infer --model Qwen3-4B-Base > infer.out 2>&1 &
