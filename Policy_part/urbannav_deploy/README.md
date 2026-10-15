# UrbanNav 部署服务组合模块

模型、采集和车端控制可分别替换。包本身不依赖 Torch/CUDA/ROS，模型在 ms-swift 服务中驻留；
`pip install -e Urban-Nav-SR/Policy_part` 只安装 `urbannav_deploy` 包，不打包旧机器人脚本或模型。

本轮实际结果：完整59.386秒clip共117/117窗口推理成功，共享GPU下1.904Hz、客户端P95 0.713秒，严格500ms deadline未通过；不得宣称独占性能或实物通过。

当前状态：因果回放规划、HTTP 模型 transport、异步 latest-only 会话已实现并有定向 CPU 边界验证。
完整 clip 实测由 ms-swift 回放验收记录承载。真实相机/GUI、里程计及 FWTW 控制 adapter 尚未接通，
不将接口或 recording fixture 当成实体控制验收。

## 接口与责任

| 边界 | 当前实现 | 替换时必须保留 |
| --- | --- | --- |
| 观测源 | 原视频＋源时间戳＋BEV索引的因果回放 | immutable帧身份、实际采集时钟、姿态/地图来源；不可用接收时间冒充采集时间 |
| 五帧同步 | `replay.plan_clip`，FPV5Hz/BEV2Hz，未满窗口和gap显式拒绝 | 禁止未来帧、重复补帧和未来地图，整数ns保留到模型适配边界 |
| 模型请求 | ms-swift `service_request_adapter.build_service_row` | canonical prompt、16query、当前observations与全新digest；无监督标签 |
| 推理 | `transport.InferenceClient` / `HttpInferenceClient` | 唯一request id，不对超时自动POST重试；GET取回不重算 |
| 实时会话 | `session.DeploymentSession` | 一条in-flight＋最多一条pending，旧pending被新输入替换并记录 |
| 车端控制 | `session.ControlAdapter`，由现场adapter实现 | 使用当前里程计重定位轨迹、显式控制权、物理限速和停车；未实现时只用recording sink |

`DeploymentSession.tick()` 由应用在独立控制时钟调用。模型在后台线程运行，tick不等待模型。
每个tick显式传入 `autonomy_owned`，默认False；无权、模型失败、源时钟异常、轨迹过期或退出均
调用 `safe_stop`。模型响应必须匹配request id和观测时钟。车端adapter继续负责几何/轨迹合同验证、
里程计健康和实际命令送达，底盘自身还应有失联watchdog；Python进程被杀不能靠finally实现停车。
记录回调必须非阻塞；不得在control tick里做模型调用、图片编码、阻塞网络或日志flush。

## 两种调度不能混淆

- 验收模式：`replay_r10_clip.py` 是 lossless scheduled replay；所有有效窗口各执行一次，
  source clock不停，慢请求形成真实可见的dispatch lag/deadline miss。它回答全输入覆盖与持续性能。
- 实时模式：`DeploymentSession` 丢弃尚未开始的旧pending，只处理最新输入；每个丢弃有事件。
  它不为了补齐历史样本而执行过期动作。模型结果到达后仍需在独立tick检查source age。

回放的“源帧到结果年龄”是原时钟映射计算，不是实际相机曝光、传输、驱动执行的端到端测量。

## 应用生命周期

1. 现场配置确定相机/GUI/里程计与控制时钟映射，模型endpoint/model identity、年龄阈值及车辆限制。
2. 创建 transport、vehicle adapter和 telemetry；`session.start()`只检查已有模型并创建一个worker。
3. 采集worker构造真实target-free row，`submit(request_id, row)`；每次输入必须有递增source timestamp。
4. 独立control timer调用 `tick(source_now_sec, autonomy_owned=...)`，使用现场控制权状态。
5. 异常或退出调用 `close()`；先停车，清空pending，再有界等待。返回False表示in-flight尚未终止，
   必须检查原request，不能重新提交推理；不能启动新会话假装旧工作已结束。

不提供通用的默认运动ROS topic或速度值：旧项目存在不同平台接口，只有现场身份确认后才绑定。
模型侧不应导入机器人SDK，车端控制模块也不应导入模型权重。

## 已有验证与后续门禁

已通过的定向fixture：5Hz/2Hz因果选取、历史不足、缺帧清空、错误时钟、latest-only替换、
控制权撤销、轨迹过期、未来时间拒绝、in-flight退出无重试。没有新增GPU调用。

下一步依次接真实采集/里程计的shadow adapter、检查Orin环境并在目标设备构建匹配engine、
保存真实端到端clock ledger；物理驱动与停车/接管必须现场验收。回放或CPU fixture不替代这些门禁。

资源准入：`admission.inspect_service_gpu` 只读检查物理GPU和compute进程；验收runner在prepare和start均检查，无法映射或共享GPU时拒绝新增推理。这是完整回放后补齐的门禁，未倒填到历史运行。

## 配置化启动入口（2026-10-15）

`python -m urbannav_deploy --config <path> --check` 只校验 JSON，不导入适配器、
不连接模型或设备。安装包后等价入口为 `urbannav-deploy`。
示例 `Policy_part/configs/deployment.shadow.example.json` 中的
`site_adapters.observations:create_source` 是明确的待实现站点适配器，不是已接通采集。
因此示例可以 `--check`，不能直接作为实机启动配置。

去掉 `--check` 后才构造配置指定的可信本地 factory。统一签名为
`factory(*, options: dict, record: callable)`；factory 应只构造对象，不启动采集或运动。
配置具有执行 Python 代码的能力，只使用维护者提供的文件，不加载外部不可信配置。

- source 实现 `ObservationSource`：`start/poll/now/exhausted/close`；poll 非阻塞，
  返回最新完整 `(request_id, row)` 或 None。临时断流不是 exhausted；now 在断流期间仍推进。
  图片解码、编码、传输和五帧/地图配对应由采集线程执行，不进入控制 tick。
- inference 实现 `InferenceClient`；内置 HTTP factory 不创建模型、不自动重试 POST。
- 默认 shadow 仅构造 `ShadowController`，记录候选计划身份和年龄，不导入车端驱动。
- live 必须明确提供 controller 和可调用的 ownership factory；每个 tick 重新获取控制权。
  它们仍需实现真实坐标、里程计、限速、接管及停车合同，配置通过不等于现场验收。
- `DeploymentApplication` 使用独立后台推理线程；主线程按 control_hz 调用采集 poll 和控制 tick。
  超过控制 deadline 时跳过已错过的槽位，不密集补发旧控制 tick。SIGINT/SIGTERM 请求协作退出。
  退出先停止控制并清空 pending，再有界等待模型并关闭采集。未结束的 in-flight 必须人工检查原请求。
- CLI 的有界日志队列由独立线程写 stdout JSONL，stderr 输出退出时的日志丢弃数、错误和终止状态。
  返回码 2 表示推理未终止或日志未完整写出；运行异常返回非零。日志丢失不能视为完整验收证据。
  adapter 的 close 必须能安全处理部分初始化；拥有阻塞资源的 adapter 必须自行设定有限超时。

本次只新增 8 个 CPU fixture 验证，分两次运行各一次：慢推理下独立控制与采集间隙、
无控制权时模型故障可见、启动失败清理、无连接配置检查、部分构造失败清理、日志失败可见、
控制逾期不补发、shadow 组合退出关闭采集一次。没有重跑旧模型测试，也没有连接 Orin。
真实 source/vehicle adapter、目标硬件部署、实际端到端时延和物理停车仍待验收。

### 已检查的现有采集代码与复用边界

- `ms-swift/swift/sidewalk_nav/runtime/streaming.py::iter_opencv_capture_requests`
  支持五帧 offsets，但时间来自 ordinal/FPS。不能用这个时钟声称实际相机曝光到控制执行的时延。
- `history_buffer.py::CausalHistoryBuffer` 固定为 4 秒/九帧 profile；本次 r10 五帧合同不能直接替换成它。
- `condition_sync.py::ConditionSynchronizer.append_bev/snapshot` 可复用地图因果选择，
  需由真实生产端提供映射到 FPV 时基的截图时间、GUI 来源和 immutable 图片路径。
- 实际相机、iPhone/iPad 高德采集程序及其传输接口尚未提供，所以 source factory 仍是接入点，
  没有用假定 URL、视频名义时钟或历史 BEV 充当真实采集。需要明确 producer 后再接线。

离线交付构建已通过：使用现有 UrbanNav Python 3.12.13，以 no-index/no-deps/no-build-isolation
构建 wheel，并在隔离目录从 wheel 内容执行配置检查；未安装到维护环境、未连接模型或硬件。
这证明包内容和入口可交付，不证明现场适配器已部署。源码 SHA256 与 wheel SHA256 见
`ResearchWorkflow/runs/r10-full-clip-20261015.json` 指向的 package/receipt.json。


### 内存帧传输接口

`transport.frame_reference(compressed_bytes, capture_id)` 生成带内容哈希的帧引用。
服务启用 memory_frame_bytes 后，`HttpInferenceClient.predict(..., frames=...)` 可以上传新原图；
后续请求引用驻留原图。source.poll 可返回 `(request_id, row, frames)`，模型 worker 负责发送。
采集时间保留在 row 中；入队或上传时间不代替采集时间。被队列丢弃的请求没有上传成功，
生产者需保留未确认的必需原图；缺帧错误直接暴露，不触发自动 POST 重试。
此接口已通过 CPU worker、真实 loopback HTTP fixture 与真实模型内存输入逐值一致性验证。
当前源码已离线构建到 r10-full-clip-20261015/package_cache，receipt.json 绑定 Python 源码与 wheel 哈希；
解包后的配置入口检查通过，未安装到维护环境或连接设备。旧 package 目录保留原版本记录。
