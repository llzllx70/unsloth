
#!/bin/bash


# nohup python -m src.trainer.MySFTTrainer --task infer --model Qwen3-4B-Base > infer.out 2>&1 &

python -m src.trainer.MySFTTrainer --task infer

