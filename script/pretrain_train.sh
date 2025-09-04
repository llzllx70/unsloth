
#!/bin/bash


nohup python -m src.trainer.MyPretrainTrainer --task train --model Qwen3-4B-Base --step 10 > pretrain_train.out 2>&1 &
