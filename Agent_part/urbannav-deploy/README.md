# UrbanNav deployment agent

此目录是 Urban-Nav-SR 仓库内 Agent_part 下的部署模块，由父仓统一版本管理，包名保持 urbannav_deploy。
从共享 Urban-Nav-SR/Policy_part 复制本轮部署模块，来源、源码哈希和保留的未提交改动见 SOURCE_PROVENANCE.json。

## 开发范围

后续部署客户端、双频调度与无运动 shadow 的开发仅修改本目录。
共享 Policy_part 保持只读；模型推理通过既有 HTTP 服务调用。
本目录不是独立 Git 仓库或 submodule；代码随 Urban-Nav-SR 提交和推送。此前独立仓提交 ca126f6 至 4a88c29 为迁移前历史证据标识，现有运行记录中的这些标识保持不变。

## 使用

在本目录执行 `pip install -e .` 安装客户端；维护环境暂未安装此副本。
无需安装的配置检查：`python -m urbannav_deploy --config configs/deployment.shadow.example.json --check`。
示例中的真实 source adapter 尚未实现，配置检查不连接设备。

详细接口与历史证据见 [模块说明](urbannav_deploy/README.md)。其中旧工作区路径描述迁移前的记录。
本次复制保留此前验证结果；不因迁移重新运行模型或设备测试。

## 离线事件统计

`python -m urbannav_deploy.telemetry --events events.jsonl --output summary.json`

输入是部署客户端 EventWriter 的单次会话日志。分别报告本地控制循环、响应到达频率，
以及按服务进程和模型实例分组的 query 更新、轨迹完成频率；不会跨主机相减时钟。
平均频率为 `(N-1)/(末次时间-首次时间)`，同时报告最大间隔；不足两点时频率为空。
query 复用不计为更新。缺少时间或服务身份会明确显示样本不足或缺失计数。
过期、控制权丢失、模型失败、待处理输入丢弃计数来自日志，不自动认定为断连原因。
还需另查 stderr 中 `telemetry_closed` 状态确认写入完成及丢失情况；日志统计不等于实物验收。

## 按时间投递已保存观测

source factory 可配置为 `urbannav_deploy.replay_source:create_source`，options 为
`requests_jsonl`、`source_start_sec` 和 `end_offset_sec`。JSONL 每行包含
`request_id`、相对于采集起点的 `offset_sec`、已有模型合同的 `row`。
row 中的图片路径必须已存在于推理服务端。source 时钟等于采集起点加本地经过时间；
投递晚到时只提交最新到期观测，并通过 `replay_delivery.skipped_slots` 记录跳过数量。
该适配器用于记录数据的双频服务验证，真实 FPV／高德采集仍需独立接口。

当前已运行实例、只读健康查询和输入合同见 [服务交接](docs/service_handoff.md)。

## 预热后开放模型服务

在服务器已有 ms-swift 维护环境中运行：

```bash
python -m urbannav_deploy.serve_warm --config model_config.json \
  --warmup-request warmup.json --lifecycle lifecycle.jsonl
```

`warmup.json` 是带独立 request_id 的合法历史观测，属于启动输入，不交给控制器。
入口在视觉缓存内部固定每张图独立编码，避免缓存首次编码的批次影响 BF16 数值。
完成一次完整前向后清除预热 query 状态，再绑定配置中的 loopback 端口。
预热失败时原样报告异常；业务端口尚未开放。健康状态的累计请求包含预热请求。
模型实现由既有环境提供，本模块只拥有启动及客户端逻辑。每次恢复使用独立 run_root 和 lifecycle 文件保留历史证据。

## 当前证据与门禁

完整收尾状态见 [采集前验收表](docs/predeployment_acceptance.md)。工程服务可用，导航质量尚未放行。

- [双频记录输入运行](docs/r10_dual_rate_20261015.md)：query 1.68 Hz、轨迹平均 5 Hz、独立控制 20 Hz，包含起步丢弃和过期记录。
- [预热与在线刷新](docs/r10_service_ready_20261016.md)：预热后开放端口；五条新增刷新用例通过。启动成本单独记录。
- [视觉缓存数值定位与修复](docs/r10_cache_batch_diagnosis_20261016.md)：实际历史批次可改变特征；逐图策略在同一输入的三个缓存历史下条件与轨迹逐值一致，验证范围为单样本。
- [已保存轨迹审阅](docs/r10_saved_output_review_20261016.md)：55 对当前输入匹配，非零年龄输出差异及重复 fresh 差异仍需解释。

HTTP ready 只表示模型完成启动预热，可以接受推理请求。真实采集、坐标对齐、导航语义及实物门禁分别验收。
本模块默认示例为 shadow，不绑定任何运动发布器；模型侧双频优化还不能作为实车放行依据。
