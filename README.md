# TANKTROUBLE_python-edition

语言:python 3.10.6

需要pygame和numpy

### 演示图
![img1](https://github.com/mglyn/TANKTROUBLE-pythonedition/blob/main/pics/pic1.png)
![img2](https://github.com/mglyn/TANKTROUBLE-pythonedition/blob/main/pics/pic2.png)

### 演示视频
https://www.bilibili.com/video/BV13d4y1Y7Tx/?vd_source=6d48c8dce1e2a6f3b5318760f3511c93

使用递归分割算法生成迷宫
基于pygame的AABB碰撞盒实现的精细碰撞逻辑

## 虚拟环境运行
```powershell
# 未安装虚拟环境时必要步骤
python -m venv .venv
# 未安装虚拟环境时必要步骤
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process

.\.venv\Scripts\Activate.ps1

# 未安装虚拟环境时必要步骤
python -m pip install --upgrade pip setuptools wheel
# 未安装虚拟环境时必要步骤
pip install -r requirements.txt

python main.py

# 调试模式运行，会暴露相应数据
python debug_run.py
```

# 强化学习训练
```powershell
# 图形化
python source/rl/train.py

# 默认优先加载 Interrupted ，如无则加载 latest
python source/rl/train.py --load auto
# 或者直接 (默认就是 auto)
python source/rl/train.py

# 指定加载 latest
python source/rl/train.py --load latest

# 指定加载 interrupted

python source/rl/train.py --load interrupted

# 从头开始
python source/rl/train.py --load none

# 命令行
python source/rl/train.py --mode cli

# 默认优先加载 Interrupted ，如无则加载 latest
python source/rl/train.py --load auto --mode cli
# 或者直接 (默认就是 auto)
python source/rl/train.py --mode cli

# 指定加载 latest
python source/rl/train.py --load latest --mode cli

# 指定加载 interrupted
python source/rl/train.py --load interrupted --mode cli

# 从头开始
python source/rl/train.py --load none --mode cli
```
    
