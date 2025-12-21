## 所需环境

- python3.10
- matplotlib
- torch

## 生成迷宫

```CLI
python maze_generator.py
```
可以选择DFS算法或者Prim算法生成迷宫

## 解迷宫

```CLI
python maze_solver.py
```
使用了A*算法来解迷宫

## 决策AI对战

```CLI
python maze_game.py
```

## DQN训练、测试与对战

- 训练：`python train.py`
- 测试：`python test.py`
- 对战：`python battle_dqn.py`

## DQN训练效果示例

请先进入到`example`文件夹中
```CLI
cd example
```
- 训练：`python train_torch.py`
- 测试：`python test_torch.py`
- 对战：`python battle_torch.py`
