## 所需环境
- python = 3.10
- numpy
- pygame

## 开始训练

- python train.py
- 恢复训练  python train.py --resume best_ai_gen_130.pkl
- 给敌人加载模型 python3 train.py --opponent best_ai_final.pkl

## 观察训练结果

- python train.py watch --model best_ai_gen_130.pkl
- python train.py watch --model best_ai_gen_170.pkl --opponent best_ai_gen_170.pkl