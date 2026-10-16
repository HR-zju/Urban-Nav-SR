# 当前 r10 服务交接

2026-10-16 只读健康核验：PID3928，GPU6，ready=true、busy=false，累计请求1（启动预热）。
这是状态快照，不保证进程以后仍在线；查询健康不执行模型。此版本只用于推理与无运动shadow。

## 当前实例

- 服务节点：原SSH跳板进入的 `ssh-d.lpai.lixiang.com:32660`，不是Orin。
- 远端run：`/lpai/volumes/mind-eb-bd-ga/suh/UrbanNavigation/runs/r10_per_image_service_20261016`。
- HTTP：服务器本机 `http://127.0.0.1:18716`；只监听loopback。
- 解释器：`/lpai/volumes/mind-eb-bd-ga/suh/UrbanNavigation/envs/urbannav_ms_swift_v31/bin/python`。
- 启动配置：run内 `model_config.json`、`runtime_env.json`、`warmup.json`。
- 实际源码：run内 `client/urbannav_deploy`；`client_commit.json` 将其哈希绑定到实现提交25f909e。
- 视觉策略：缓存内部逐图编码，lifecycle中 `vision_policy.batch_unit=one_image`。
- 积分：8步、零初始动作；标准正态诊断没有改动服务。
- 模型digest：`89a5d147c5455332e7b754280eef6af3bd0d5101e12d30804260220a34f2a9d0`。
- engine digest：`824da2bc012467096aa1f155b52bb4d6bb9ab7caeb2315d6d2c94bf549b01888`。

## 不触发推理的查看方法

登录推理服务器后执行：

```bash
curl --fail --silent http://127.0.0.1:18716/readyz
curl --fail --silent http://127.0.0.1:18716/v1/requests/startup-warmup
```

第二条只取回启动预热结果，不能把该历史轨迹送给机器人。
诊断失败看run内 `service.log`、`supervisor.log`、`lifecycle.jsonl` 和存在时的 `exit.json`。
旧实例PID1601已正常退出；不要复用旧PID或旧18714端口说明进行操作。

## 提交新的观测

已有客户端或 `POST /v1/predict` 消费JSON：

```json
{"request_id":"唯一请求标识","row":"此处应为完整的target-free数据行对象"}
```

上面只说明外层字段，字符串占位不是可提交样例。完整输入的结构参考run内 `warmup.json` 的row，
新观测必须提供自己的五帧FPV、当前BEV/GUI、instruction、角色及真实时间；图片路径必须位于服务器允许的媒体根。
本接口不会自动将任意MP4转换成五帧配对；clip需先经已有replay输入准备，保留帧与时间身份。

返回 `[32,3]` 的x_forward/y_left/yaw_rel，单位m/m/rad，参考当前观测机体系，时间点0.125至4秒。
返回轨迹是候选参考，不是底盘速度命令。忙时429，非法输入400，ID冲突409，模型错误500。
断线后先查 `GET /v1/requests/<request_id>`，不要自动更换ID重算。

## 恢复边界

只在确认当前实例已退出后恢复。重新核对GPU、端口和冻结源码；沿用维护环境及runtime_env中的依赖。
使用新的run_root、独立lifecycle和日志，入口为 `python -m urbannav_deploy.serve_warm`，
参数为 `--config`、`--warmup-request`、`--lifecycle`。旧记录保留，不覆盖请求与故障产物。
run内 `prepare_restore.py` 是此次带旧PID身份检查的一次性迁移记录，不能当作通用重启脚本再次执行。

完整工程证据与仍未通过的质量项见 [采集前验收表](predeployment_acceptance.md)。
