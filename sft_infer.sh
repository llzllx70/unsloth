
#!/bin/bash


nohup python ./src/trainer/MySFTTrainer.py --task infer --model Qwen3-4B-Base > infer.out 2>&1 &

# python ./src/MySFTTrainer.py --task infer --model Qwen3-4B-Base
