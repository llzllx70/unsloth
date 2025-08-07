
#!/bin/bash


nohup python -m src.trainer.MyGRPOTrainer --task infer --model Qwen3-4B-Base > infer.out 2>&1 &
