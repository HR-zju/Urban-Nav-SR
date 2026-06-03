# nav_gps_pose_alignment.py 使用说明

以当前数据集为例：

```text
/Users/minruiliu/Desktop/SR/0602/nav_dataset_20260601_083907
```

这个脚本用于把 OpenStreetMobile 采集到的 ARKit 局部轨迹 `pose0.csv` 和 GPS `gps0.csv` 对齐到本地 ENU 坐标系，输出可直接用于轨迹分析和可视化的 `aligned_pose_enu.csv`。

## 1. 基本用法

运行：

```bash
python nav_gps_pose_alignment.py \
  /Users/minruiliu/Desktop/SR/0602/nav_dataset_20260601_083907 \
  --max-h-accuracy 20
```

运行成功后会输出：

```text
wrote /Users/minruiliu/Desktop/SR/0602/nav_dataset_20260601_083907/aligned_pose_enu.csv
wrote /Users/minruiliu/Desktop/SR/0602/nav_dataset_20260601_083907/alignment_metadata.txt
```

## 3. 参数说明

```bash
python tools/nav_gps_pose_alignment.py DATASET_DIR [--max-h-accuracy VALUE] [--pose-axes AXES]
```

参数含义：

| 参数 | 说明 |
|---|---|
| `DATASET_DIR` | NAV REC 数据集目录，例如当前的 `nav_dataset_20260601_083907` |
| `--max-h-accuracy 20` | 只使用 GPS `h_accuracy <= 20 m` 的样本参与对（防止剧烈运动跳变） |
| `--pose-axes x,-z` | 指定 ARKit 的水平面轴，默认就是 `x,-z` |

为什么默认是 `x,-z`：

ARKit 使用 `worldAlignment = .gravity` 时，`y` 轴是竖直方向；因此水平平面使用 `x` 和 `-z`。脚本默认已经按这个规则处理。

## 2. 输出文件

### aligned_pose_enu.csv

这是主要结果文件，包含对齐后的 ARKit 轨迹和插值后的 GPS 轨迹。

主要字段：

| 字段 | 含义 |
|---|---|
| `timestamp [ns]` | 时间戳 |
| `east,north,up` | ARKit pose 对齐到 GPS ENU 后的位置 |
| `gps_east,gps_north,gps_up` | 同一时间插值得到的 GPS ENU 位置 |
| `pose_horizontal_a,pose_horizontal_b` | 原始 ARKit pose 投影到水平面的两个坐标 |
| `pose_x,pose_y,pose_z` | 原始 ARKit 坐标 |
| `qw,qx,qy,qz` | 原始 ARKit 姿态四元数 |

示例首行：

```text
timestamp [ns],east,north,up,gps_east,gps_north,gps_up,...
66477740380791,-2.760801513,-1.269112962,0.122666607,-0.026034416,-0.088230179,0.122666607,...
```

### alignment_metadata.txt

这是对齐参数和质量指标文件。

当前数据集运行后的结果：

```text
alignment_type=arkit_yaw_translation_2d
aligned_samples=34662
pose_horizontal_axes=x,-z
pose_vertical_axis=y
yaw_deg=102.618309490
translation_east=-2.544665024
translation_north=-0.772755320
rmse_m=2.850412835
max_h_accuracy=20.000
```

关键指标解释：

| 指标 | 说明 |
|---|---|
| `aligned_samples` | 成功对齐的 pose 样本数 |
| `yaw_deg` | ARKit 局部水平坐标转到 GPS ENU 坐标时估计出的旋转角 |
| `translation_east/north` | 平移量 |
| `rmse_m` | 对齐后的均方根误差，单位为米 |
| `max_h_accuracy` | 使用的 GPS 水平精度阈值 |



## 3. 常见用法

### 标准推荐用法

```bash
python tools/nav_gps_pose_alignment.py \
  /Users/minruiliu/Desktop/SR/0602/nav_dataset_20260601_083907 \
  --max-h-accuracy 20
```

### 放宽 GPS 过滤

如果 GPS 样本太少，可以临时放宽 GPS 水平精度阈值：

```bash
python tools/nav_gps_pose_alignment.py \
  /Users/minruiliu/Desktop/SR/0602/nav_dataset_20260601_083907 \
  --max-h-accuracy 50
```

但放宽后可能引入低质量 GPS 点，使 `rmse_m` 变大。

### 显式指定 ARKit 水平轴

```bash
python tools/nav_gps_pose_alignment.py \
  /Users/minruiliu/Desktop/SR/0602/nav_dataset_20260601_083907 \
  --max-h-accuracy 20 \
  --pose-axes x,-z
```
