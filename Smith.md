这是一个关于 SmithAI (AgentSmith) 决策逻辑的完整技术总结。这份文档旨在帮助另一个 AI 模型理解其核心架构、决策优先级和具体算法，以便将其移植到新的坦克大战环境中。

SmithAI (AgentSmith) 决策系统架构总结
1. 总体架构概述
SmithAI 采用了一个 分层决策系统 (Hierarchical Decision Making)，由 AgentSmith 类作为中央大脑（Controller），根据当前战场局势动态切换三种具体的策略（Strategy）：

DodgeStrategy (躲避策略)：优先级最高。当检测到即将到来的炮弹威胁时触发。
AttackStrategy (攻击策略)：优先级中等。当处于射程内且计算出有效射击路径（直射或反弹）时触发。
ContactStrategy (接触/寻路策略)：优先级最低。当没有威胁且无法立即攻击时，使用 A* 算法向敌人移动。
整个系统的核心在于 弹道预测 (Ballistic Prediction) 和 碰撞模拟。

2. 核心感知与预测模块 (Perception & Prediction)
在做出任何决策之前，AgentSmith 首先构建战场的未来状态模型。

2.1. 炮弹威胁检测 (getIncomingShells)
筛选：遍历场上所有炮弹，排除自己刚发射的（TTL满的）炮弹。
范围判定：只关注距离自身 THREATENED_RANGE (150单位) 内的炮弹。
2.2. 弹道预测 (ballisticsPredict)
这是 AI 的核心能力。它不只看当前位置，而是预测炮弹未来的轨迹。

分段预测 (BallisticSegment)：将炮弹轨迹分解为多个直线段。
反弹模拟：模拟炮弹在墙壁上的反弹。如果炮弹撞墙，计算反射向量，生成新的轨迹段。
时间戳对齐：每个轨迹段都包含 start 和 end 的时间步（globalStep），确保躲避计算时的时间同步。
数据结构：BallisticSegment 包含起点、终点、角度、长度、所属炮弹ID等。
3. 决策逻辑详解 (Decision Logic)
3.1. 最高优先级：躲避策略 (DodgeStrategy)
当预测到的炮弹轨迹与坦克未来的位置（考虑坦克自身的矩形碰撞箱）发生碰撞时，触发躲避。

躲避算法流程 (按尝试顺序)：
AI 会依次模拟以下动作，一旦找到一个“安全”的动作，立即生成指令队列并执行。

原地旋转躲避 (tryRotation)：

逻辑：尝试顺时针或逆时针旋转。
原理：坦克的碰撞箱是矩形，旋转可以改变受弹面积（即“侧身”躲子弹）。
检查：模拟旋转后的每一帧，检查是否与预测的炮弹轨迹重叠。
旋转并移动 (tryRotatingAndMoving)：

逻辑：判断炮弹来袭方向，尝试向炮弹轨迹的垂直方向（左侧或右侧）移动。
组合：先旋转到垂直于炮弹轨迹的角度，然后直行（前进或后退）。
曲线移动 (tryRotatingWithMoving)：

逻辑：一边旋转一边移动（模拟弧形运动）。
场景：当单纯旋转或直线移动都无法避开时使用。
安全检查 (checkFeasible & checkWillDie)：

任何躲避动作的模拟，都会检查：
是否撞墙（地形碰撞）。
是否会撞上其他预测的炮弹（避免躲了一发撞上另一发）。
3.2. 中等优先级：攻击策略 (AttackStrategy)
当没有迫切的生存威胁时，AI 尝试攻击。

瞄准算法 (tryAiming)：

直射检查：
计算自身与敌人的连线。
检查连线上是否有障碍物（墙壁）。
如果没有障碍物，直接锁定敌人角度。
反弹射击 (Bounce Shot)：
如果无法直射，AI 会暴力搜索发射角度（每隔一定角度尝试一次）。
利用 ballisticPredict 模拟该角度发射出的炮弹轨迹。
如果模拟轨迹的任何一段（包括反弹后的）击中敌人，则锁定该角度。
优化：排除了 0, 90, 180, 270 度附近的无效角度。
执行：

生成 AttackStrategy，旋转炮口对准计算出的角度，然后开火。
开火后立即切换回 ContactStrategy 或 DodgeStrategy。
3.3. 最低优先级：接触策略 (ContactStrategy)
当不躲避也不攻击时，AI 主动寻找敌人。

A 寻路算法 (AStar)*：

地图栅格化：将连续的坐标系映射为离散的网格（Grid）。
障碍物处理：根据地图上的 Block 标记不可达区域。
路径生成：计算从当前位置到敌人位置的最短路径。
移动控制：
获取路径上的下一个节点中心。
计算目标向量与当前朝向的夹角。
PID类控制：如果角度差大，先原地旋转；如果角度合适，前进；如果卡住（stuckSteps > 5），尝试倒车和乱动以脱困。
安全移动 (safeToMove)：在 ContactStrategy 的每一步移动前，都会调用 AgentSmith::safeToMove 再次确认这一步迈出去会不会正好撞上子弹。如果会死，就立刻停车。
4. 关键数据结构与辅助函数
为了移植，需要特别注意以下数据定义：

Object::PosInfo：包含 pos (x, y 向量) 和 angle (朝向)。
DodgeCommand：指令结构体，包含操作类型（如 DODGE_CMD_ROTATE_CW）、持续步数、目标时间步。
util::Vec：向量类，支持加减、点积、叉积、求模等几何运算。
碰撞检测函数：
util::checkRectRectCollision：矩形与矩形碰撞（用于坦克 vs 墙，坦克 vs 炮弹轨迹段）。
util::checkRectCircleCollision：矩形与圆碰撞（用于坦克 vs 炮弹当前位置）。
5. 移植时的注意事项 (给接收方 AI 的提示)
坐标系与角度：注意原环境是使用 360 度制还是弧度制（代码中似乎混用，PosInfo 用角度，三角函数需弧度）。注意 Y 轴方向（通常游戏开发中 Y 轴向下或向上）。
物理步长 (Step)：SmithAI 强依赖于 globalStep。预测逻辑假设物理引擎是确定性的（Deterministic）。移植时需确保预测步长与新环境的物理更新频率一致。
碰撞箱尺寸：代码中硬编码了 Tank::TANK_WIDTH, Tank::TANK_HEIGHT, Shell::RADIUS。在新环境中需替换为实际尺寸。
A 网格精度*：A_STAR_GRID_SIZE 决定了寻路的精细度。过大导致无法穿过狭窄通道，过小导致计算量爆炸。
总结一句话： SmithAI 是一个基于确定

性物理预测的反应式 AI**。它不使用机器学习，而是通过精确计算未来的物理状态来寻找生存空间（躲避）和杀伤路径（反弹射击），并在安全时使用 A* 算法逼近敌人。******