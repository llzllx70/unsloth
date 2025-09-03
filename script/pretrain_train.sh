
#!/bin/bash


nohup python -m src.trainer.MyPretrainTrainer --task train --model Qwen3-4B-Base --step 20 > pretrain_train.out 2>&1 &
