# 文件名: player.py
# 类名: Player
# 函数名: tick_normal
# 描述: 在正常模式下更新小鸟的状态。处理小鸟的运动和旋转。
# 输入: 无
# 输出: 无
# 潜在用途: 在深度学习模型中，用于计算每一步后小鸟的新位置和状态。
# 相关数值:
#   - vel_y: Y轴速度
#   - max_vel_y: Y轴最大速度
#   - min_vel_y: Y轴最小速度
#   - acc_y: Y轴加速度
#   - rot: 当前旋转角度
#   - vel_rot: 旋转速度
#   - rot_min: 最小旋转角度
#   - rot_max: 最大旋转角度
#   - flap_acc: 拍动翅膀时的加速度

# 文件名: pipes.py
# 类名: Pipes
# 函数名: tick
# 描述: 更新所有水管的位置。如果需要，生成新的水管并移除旧的水管。
# 输入: 无
# 输出: 无
# 潜在用途: 在深度学习模型中，用于更新环境状态，特别是水管的位置。
# 相关数值:
#   - vel_x: 水管的水平移动速度 (-5)

# 文件名: score.py
# 类名: Score
# 函数名: add
# 描述: 增加分数，并播放得分音效。
# 输入: 无
# 输出: 无
# 潜在用途: 在深度学习模型中，用于更新和记录得分，可能用于奖励函数的一部分。

# 文件名: entity.py
# 类名: Entity
# 函数名: collide
# 描述: 检测两个实体之间是否发生了碰撞。
# 输入: 
#   - other: 另一个实体对象
# 输出: 
#   - bool: 表示是否发生碰撞
# 潜在用途: 在深度学习模型中，用于检测小鸟是否与其他对象（如水管或地板）发生碰撞。

# 文件名: floor.py
# 类名: Floor
# 描述: 表示游戏中的地板。
# 相关数值:
#   - vel_x: 地板的水平移动速度 (4)
