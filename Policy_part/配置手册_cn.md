# UMR PD Controller 配置手册

本文档用于调试 `pd_controller_ros2_new2_agilex.py` 在 UMR 底盘上的转向手感。当前控制器的核心目标是：小转向能越过底盘死区，大转向不过冲，线速度在转弯时适当降低。

## 相关文件

- 控制器：`pd_controller_ros2_new2_agilex.py`
- 速度上限配置：`config/robot.yaml`
- 输出话题：`/cmd_vel`
- 输入 waypoint：`/waypoint`

如果 navflow、waypoint publisher 和 PD controller 都在 orin26 本机跑，建议它们启动前都设置：

```bash
export ROS_LOCALHOST_ONLY=1
```

否则多网卡环境下可能出现 `/waypoint` 已发布但 PD controller 收不到的情况。

## 当前推荐初始值

`pd_controller_ros2_new2_agilex.py`：

```python
TURN_KP = 1.2
MIN_TURN_W = 0.18
TURN_DEADBAND_ANGLE = 0.03
MAX_DELTA_W = 0.12
TURN_SPEED_DECAY = 0.8
```

`config/robot.yaml`：

```yaml
max_v: 1.0
max_w: 0.45
```

如果同时使用 `tools/ros2_cmd_vel_to_agilex_ws.py` 转发到底盘，还要确认 bridge 的 `--max-angular` 不小于 `config/robot.yaml` 的 `max_w`。例如：

```bash
tools/agilex_api/run_navflow_cmd_vel_ws_bridge.sh --publish --max-linear 0.8 --max-angular 0.45
```

## 参数含义

### `TURN_KP`

角速度比例增益。控制 waypoint 横向偏差转成 `angular.z` 的力度。

- 调大：转弯更积极，入弯更快。
- 调小：转弯更柔和，但可能跟不住弯。
- 推荐步长：每次 `+/- 0.2`。
- 常用范围：`0.8 ~ 1.8`。

### `MIN_TURN_W`

最小有效转向角速度，用来补偿 UMR 左右旋转死区。

- 调大：小弯也能明显转起来。
- 调小：小弯更细腻，但可能又落入底盘死区。
- 推荐步长：每次 `+/- 0.03 rad/s`。
- 常用范围：`0.12 ~ 0.28 rad/s`。

如果机器人收到小角速度不动，优先调大这个值。  
如果小弯突然一拐一拐，优先调小这个值。

### `TURN_DEADBAND_ANGLE`

转向死区补偿触发阈值，单位是弧度。只有 `abs(heading_error)` 大于这个值时，才会启用 `MIN_TURN_W`。

- 调大：更小的横向误差不会触发转向补偿，直行更稳。
- 调小：轻微偏差也会主动修正。
- 推荐步长：每次 `+/- 0.01 rad`。
- 常用范围：`0.02 ~ 0.08 rad`。

如果直线行驶时左右抖动，调大它。  
如果轻微偏航长期修不回来，调小它。

### `MAX_DELTA_W`

每个控制周期允许角速度变化的最大值。当前 `RATE=10 Hz`，所以 `MAX_DELTA_W=0.12` 大约对应 `1.2 rad/s^2` 的角加速度限制。

- 调大：转向响应更快。
- 调小：转向更丝滑，但可能反应慢。
- 推荐步长：每次 `+/- 0.03`。
- 常用范围：`0.06 ~ 0.18`。

如果入弯太慢，调大它。  
如果转向命令跳变明显、车身一顿一顿，调小它。

### `TURN_SPEED_DECAY`

转弯时线速度衰减强度。角速度越大，线速度越低。

- 调大：转弯时更慢，更稳。
- 调小：转弯时保持速度，更激进。
- 推荐步长：每次 `+/- 0.2`。
- 常用范围：`0.4 ~ 1.4`。

如果转弯外甩、弯道冲过头，调大它。  
如果转弯太慢、效率低，调小它。

### `max_w`

角速度绝对上限，来自 `config/robot.yaml`。

- 调大：允许更急的转弯。
- 调小：限制急转，降低风险。
- 推荐步长：每次 `+/- 0.05 rad/s`。
- UMR 初始建议：`0.45`。
- 如果底盘和环境允许，可试：`0.55 ~ 0.65`。

注意：bridge 的 `--max-angular` 会再次限幅。如果 `max_w=0.65`，但 bridge 仍是 `--max-angular 0.4`，最终到底盘还是最多 `0.4`。

### `max_v`

线速度绝对上限，来自 `config/robot.yaml`。

- 调大：直线更快。
- 调小：整体更稳。
- 推荐步长：每次 `+/- 0.1 m/s`。

调转弯手感时，先不要改 `max_v`。先把转向参数调顺，再决定是否提高速度。

## 按现象调参

### 小弯不转，或者转得很晚

优先顺序：

1. `MIN_TURN_W += 0.03`
2. `TURN_KP += 0.2`
3. 如果仍不够，再把 `max_w += 0.05`

观察目标：轻微左右 waypoint 时，`w` 至少达到底盘能响应的值，例如 `0.18 ~ 0.25 rad/s`。

### 小弯转向太突兀

优先顺序：

1. `MIN_TURN_W -= 0.03`
2. `MAX_DELTA_W -= 0.03`
3. `TURN_DEADBAND_ANGLE += 0.01`

观察目标：轻微横向偏差时不会突然甩头，`w` 有平滑过渡。

### 直线左右摆

优先顺序：

1. `TURN_DEADBAND_ANGLE += 0.01`
2. `TURN_KP -= 0.2`
3. `MAX_DELTA_W -= 0.03`

观察目标：直线 waypoint 的 `dy` 小幅变化时，`w` 不频繁正负切换。

### 入弯慢，跟不住弯

优先顺序：

1. `TURN_KP += 0.2`
2. `MAX_DELTA_W += 0.03`
3. `max_w += 0.05`

观察目标：中等弯道的 `w` 能较快升到 `0.3 ~ 0.45 rad/s`。

### 急弯冲过头或外甩

优先顺序：

1. `TURN_SPEED_DECAY += 0.2`
2. `max_w -= 0.05`
3. `max_v -= 0.1`

观察目标：急弯时 `v` 自动降下来，例如 `w=0.45` 时 `v` 在 `0.4 ~ 0.6 m/s`。

### 转向响应太慢但又不想更激进

优先调：

1. `MAX_DELTA_W += 0.03`

不要先调 `TURN_KP`。`TURN_KP` 会改变目标角速度，`MAX_DELTA_W` 只改变到达目标角速度的速度。

## 推荐调参流程

每次只改一个参数。改完后用 fake waypoint 做三组测试，再上车低速实测。

### 1. 轻微左转

```bash
export ROS_LOCALHOST_ONLY=1
source /opt/ros/humble/setup.zsh
source /home/orin26/miniforge3/etc/profile.d/conda.sh
conda activate foundation_nav
cd /home/orin26/code/Urban-Nav-SR/Policy_part

ros2 topic pub --rate 10 /waypoint std_msgs/msg/Float32MultiArray \
"{data: [0.5, 0.06, 1.0, 0.12, 1.5, 0.20, 2.0, 0.26]}"
```

当前参考输出：

```text
v ~= 0.65
w ~= +0.18
```

### 2. 轻微右转

```bash
ros2 topic pub --rate 10 /waypoint std_msgs/msg/Float32MultiArray \
"{data: [0.5, -0.06, 1.0, -0.12, 1.5, -0.20, 2.0, -0.26]}"
```

当前参考输出：

```text
v ~= 0.65
w ~= -0.18
```

### 3. 急左转

```bash
ros2 topic pub --rate 10 /waypoint std_msgs/msg/Float32MultiArray \
"{data: [0.5, 0.20, 1.0, 0.50, 1.5, 0.90, 2.0, 1.20]}"
```

当前参考输出：

```text
v ~= 0.48
w ~= +0.45
```

## 调参记录模板

每次实车测试建议记录：

```text
日期:
场地:
参数:
  TURN_KP:
  MIN_TURN_W:
  TURN_DEADBAND_ANGLE:
  MAX_DELTA_W:
  TURN_SPEED_DECAY:
  max_v:
  max_w:
现象:
  直线:
  小弯:
  急弯:
  是否左右摆:
  是否外甩:
下一步:
```

## 安全注意

- 实车测试前确认 `/cmd_vel` bridge 是否开启。如果 bridge 开启，fake waypoint 会让底盘运动。
- 第一次改参数后，先不要提高 `max_v`。
- 如果开启 WebSocket bridge，确认 `--max-angular` 和 `max_w` 一致，否则你以为改大了角速度，实际可能被 bridge 截断。
- 如果 PD controller 收不到 `/waypoint`，先确认所有相关 pane 都设置了 `ROS_LOCALHOST_ONLY=1`，并且重新启动节点。
