# 可插拔 Open-loop Benchmark V1

## 目的与责任边界

本实现将离线 benchmark 固定为三个独立轴：

```text
冻结数据集插件 × target-free 模型适配器 × 冻结指标插件
```

RouteServer 是 contract、数据集构造、隐藏真值、指标、scorecard 和 HTML review 的唯一 owner；`ms-swift` 只把严格公共输入变成物理模型请求并写 prediction/failure ledger。`CityWalker` 是只读参考仓：RouteServer 不 import 或修改其训练代码；仅当用户明确指定 config/checkpoint 时，ms-swift 的独立 native adapter 才在独立进程中加载其模型实现。

V1 不替代已有 `urbannav_openloop_benchmark_v2_1`。后者仍是现有 Stage I/II 实验的合同；新合同只解决“不同数据集、不同模型输出节奏、不同指标套件”不能被既有 stage/task 限制表达的问题。

## Artifact 边界

| Artifact | 是否可交给模型 | 内容 |
| --- | --- | --- |
| `inputs.jsonl` | 是 | messages、images、公共坐标/point goal 与 modality stamp |
| `references.jsonl` | 否 | future trajectory、source timestamps、mask、arrival、scenario、subgroup |
| prediction/failure JSONL | 否 | 输入 digest 绑定的模型输出或显式失败 |
| `native_adapter_trace.jsonl`（可选） | 否 | 仅 public-only 的原生模型输入选择、尺度/坐标投影与受控失败码；只用于 review coverage，不进入 scorecard |

`BenchmarkInputV1` 递归拒绝 `trajectory_ego_t`、`future_trajectory`、`arrival_label`、`scenario_labels`、`reference` 等私有键。每一个 prediction/failure 都必须携带 exact public input 的 SHA256 digest；prediction 还必须声明 `prediction_geometry_id` 并与 profile 匹配。runner 在计算任何指标前验证这些条件。

如果需要把外部原生模型（例如未来 CityWalker→UrbanNav）的结果写入同一 benchmark，模型侧还可额外写
`citywalker_to_urbannav_adapter_contract_v1` JSONL。每行必须以 `sample_id + benchmark_input_digest` 绑定 frozen
public input，且严格只允许记录下列 adapter 审计事实：native adapter/checkpoint stamp、实际选用的 image index/role、
公开 history transform、公开 scale estimator/value、输入/输出坐标 projection、native output schedule/count，以及
`eligible | input_ineligible | native_model_error | output_contract_error` 与受控的 first-failure `reject_reason`。其中
`input_ineligible` 只能使用 `insufficient_public_fpv`、`missing_public_history`、`missing_public_goal`、`invalid_scale` 或
`unsupported_public_input`；其余两个失败状态分别只能使用 `native_model_error`、`output_contract_error`。schema `extra=forbid`，
因此 future trajectory、hidden reference timestamp、GT-derived scale 和 evaluator L2 均无法写入。RouteServer 还会检查
选帧 index/role 与 frozen input 一致。该 sidecar 不改变 input、reference、prediction、canonical scorecard，也不能拿
“仅适配成功子集”替换冻结分母。

## CityWalker 数据集插件

`mcp_server.data.benchmark.citywalker_dataset_plugin` 直接解析 reference Teleop 的三行记录：GPS timestamp、6DoF pose/frame id、六个可重叠 category flag。官方 sequence split 为 train `01–08`、validation `09`、test `10–18`；train anchor stride 为 5，validation/test 为 1，与 reference `TeleopDataset` 保持一致。

reference 实现中 point goal 每次 `__getitem__` 随机抽取；V1 以 `SHA256(target_seed, sequence_id, pose_start)` 创建本地 RNG，抽取结果编码在 `sample_id` 中。历史图像、历史 local xy 和 point goal 是公共输入；未来五个 pose、其来自 GPS 的真实相对 timestamp、arrival draw、category 都留在 reference。reference 逐字段记录 provenance：trajectory/arrival 是确定性 source+sampler 产物，scenario 是 CityWalker reference annotation，不能把两者都称为人工 GT。

坐标严格写为 `citywalker_current_pose_x_right_y_front_v1`。其变换复现 reference 的 `inv(T_current) @ T_future` 加 `[y, -x]` 轴变换；实现中用 Rodrigues 公式，避免引入 reference repo 或 SciPy runtime。

## 时间、指标和 coverage

profile 的 `BenchmarkTimePolicyV1.horizons_s` 是主指标唯一时间轴。两条轨迹仅在其各自可覆盖 horizon 上做线性 xy 插值，禁止外推：

\[
C_t = N_{\mathrm{prediction\ covers\ }t} / N_{\mathrm{reference\ covers\ }t}.
\]

因此 report 同时给出 conditional L2、横向/纵向误差与 coverage。没有 prediction 不是零误差，也不会从分母消失。CityWalker suite 额外给出非退化 GT 位移下的 AOE/MAOE、arrival（模型未输出 score 时为 `N/A`）和六个可重叠 scenario slice。切片不互斥，不能与 overall 相加。

`reference_schedule` 只允许原生 CityWalker adapter 使用：它不携带 GT timestamp，runner 才将它的输出索引对齐隐藏 reference schedule。UrbanNav CFM adapter 必须写 `explicit` timestamp。

## 当前 UrbanNav CFM 模型直接评测 CityWalker

`ms-swift` 的 `UrbanNavStageIIBenchmarkAdapterV1` 可把 CityWalker public prompt 直接交给已有 `[TRAJ_i]` + action head，不需要在 CityWalker 重新训练。该 arm 必须标为 `adaptation_mode=zero_shot`；结果只能说明 frozen CityWalker open-loop transfer，不说明 CityWalker fine-tune 成果。

它把 checkpoint 输出的 metric FLU `(x_forward, y_left)` 显式转换为 CityWalker `(x_right, y_front)`：

\[
(x_{right}, y_{front})=(-y_{left}, x_{forward}).
\]

prediction 中固定记录 `coordinate_projection_id=flu_to_citywalker_xright_yfront_v1`。任何新的标定、图像外参或输出 frame 必须新增版本化 adapter/profile，而不是静默改变这个映射。

CityWalker 原生模型也可使用 `citywalker_reference_callback_v1`；callback 只能收到 `BenchmarkInputV1`，并输出 native xy/可选 arrival score，不获得 hidden future/timestamp/reference。

## UrbanNav 自有 benchmark

`UrbanNavDatasetPluginV1` 仅投影已有、冻结的 legacy Stage-II trajectory input/reference；不会重新构造 trajectory。它保留模型可见 prompt/media，将 hidden `trajectory_ego_t` 和 `action_mask` 写为 generic reference。共同的 fixed-time geometry suite 用于比较；UrbanNav suite 仅当模型 ledger 明确写出 direction readout 时才计算方向诊断，否则显式 `N/A`。

投影入口是 `tools/evaluation/build_urbannav_plugin_benchmark.py`，接收 legacy input/reference JSONL，固定 split/profile 后写出同一类 `inputs.jsonl`、`references.jsonl` 和 `package.json`；因此 CityWalker benchmark 和 UrbanNav 自有 benchmark 进入完全相同的 adapter/evaluator 接口。

除 legacy projection 外，`language_waypoint_dataset_plugin` 提供新的条件消融 package：
`fpv_only`、`fpv_language`、`fpv_waypoint` 与 `fpv_language_waypoint`。它要求语言文本来自
验收的 anchor-visible annotation，waypoint 来自独立 planner/GPS route，未来 odometry 永远只写入
hidden reference。具体 typed contracts、归一化和 CLI 见
[LANGUAGE_WAYPOINT_BENCHMARK_V1.md](./LANGUAGE_WAYPOINT_BENCHMARK_V1.md)。该模块目前只完成
preprocessing/adapter smoke；不表示已有真实 waypoint 数据或 condition-aware 模型训练结果。

## CityWalker 原生模型评测 UrbanNav

`CityWalkerToUrbanNavAdapterV1` 是独立于 `citywalker_reference_callback_v1` 的反向 adapter；后者仍只适用于“外部调用者已经算好 CityWalker trajectory”的薄序列化桥，不能承担本节的 input-adaptation、尺度审计或 native inference。

反向 adapter 的 admission 完全从 `BenchmarkInputV1` 的 public fields 完成，**不会读取 `references.jsonl`**：

1. `public_context.coordinate_frame` 必须为 `current_ego_x_forward_y_left_v1`；
2. `history_xy_m` 与 `history_image_indices` 必须显式一一对应，按时间取最后五个 index；所有选中 role 必须是 FPV；
3. `target_xy_m` 必须存在且为有限的 FLU 二维坐标；
4. 尺度固定为这五个公开 history 相邻正位移（阈值 `0.01 m`）的中位数，不能使用 CityWalker 原始 `TeleopDataset.step_scale`，因为后者由整段 pose trajectory 计算，可能包含未来位置；
5. 坐标显式映射为 `(right, front)=(-left, forward)`；history 按 CityWalker `input_target` 规则相对于 current 平移并旋转，goal 保持 native 数据集所用的 unrotated current-local xy；六行均除以上述 public scale；
6. native output 必须恰为五个 CityWalker xy waypoint，时间合同固定为 `[1,2,3,4,5] s`，再映射回 UrbanNav FLU `(forward,left)=(front,-right)`。

缺字段、帧数不足、history/image 对齐不明确或无稳定公开位移时，row 保留在完整 source denominator 的 audit funnel，但不进入 derived package。`compatibility_audit.json` 只含 `sample_id`、public input digest、admission reason、selected public index/role 和 public scale；没有 hidden trajectory、GT scale、L2 或 source reference timestamp。

先从已有 generic UrbanNav package 做 audit；该命令不会加载 CityWalker 模型：

```bash
cd Agent_part/RouteServer
conda run -n UrbanNav env PYTHONPATH=.:src python \
  tools/evaluation/build_urbannav_citywalker_compatible_benchmark.py \
  --source-package-json /volumes/.../urbannav_validation/package.json \
  --output-root /volumes/.../urbannav_citywalker_audit --audit-only
```

确认 coverage 后，不带 `--audit-only` 生成新的 `inputs.jsonl/references.jsonl/package.json`。derived profile 是
`urbannav_citywalker_native_openloop_v1`，只冻结 CityWalker 原生能显式覆盖的 `1 s/2 s`，使用 common geometry + UrbanNav
metric suite。后者对 direction 返回 `N/A`，因为 native CityWalker 不输出 UrbanNav direction readout；arrival 同样不从 native
arrival logit 或 trajectory endpoint 伪造，保持 `N/A`。

模型侧执行需要用户指定只读 CityWalker checkout、**精确** YAML config 和 checkpoint：

```bash
python -m swift.sidewalk_nav.cli.infer_citywalker_to_urbannav_benchmark \
  --citywalker-root /workspace/CityWalker --config /workspace/CityWalker/config/finetune.yaml \
  --checkpoint /volumes/.../citywalker.ckpt --device cuda:0 \
  --input-jsonl /volumes/.../urbannav_citywalker_native/inputs.jsonl \
  --prediction-jsonl /volumes/.../run/predictions.jsonl \
  --failure-jsonl /volumes/.../run/failures.jsonl \
  --adapter-trace-jsonl /volumes/.../run/native_adapter_trace.jsonl \
  --run-audit-json /volumes/.../run/run_audit.json
```

该 loader fail-closed 地检查 `context_size=5`、`len_traj_pred=5` 和 `input_target` coordinate embedding；它只以 five public
FPV、six public coordinate rows运行 native inner model，且每个 input 产生 prediction 或 failure，并产生一条 trace。随后仍只能由
RouteServer 使用同一套 runner 评分：

```bash
conda run -n UrbanNav env PYTHONPATH=.:src python \
  tools/evaluation/evaluate_benchmark_plugin.py \
  --package-json /volumes/.../urbannav_citywalker_native/package.json \
  --prediction-jsonl /volumes/.../run/predictions.jsonl \
  --failure-jsonl /volumes/.../run/failures.jsonl \
  --adapter-trace-jsonl /volumes/.../run/native_adapter_trace.jsonl \
  --require-complete-adapter-traces --output-root /volumes/.../run/scorecard
```

目前本仓只完成了 synthetic contract/package/ledger smoke；尚未给定实际 CityWalker config/checkpoint 路径，因此尚未声称完成
native GPU smoke 或完整 UrbanNav validation。

## 使用顺序

先构建 package，之后任意 model adapter 只读取其 `inputs.jsonl`：

```bash
cd Agent_part/RouteServer
conda run -n UrbanNav env PYTHONPATH=.:src python \
  tools/evaluation/build_citywalker_plugin_benchmark.py \
  --pose-label-dir /media/ubuntu/data2/CityWalker/prepared/teleop/pose_label \
  --image-root-dir /media/ubuntu/data2/CityWalker/prepared/teleop/obs \
  --output-root /volumes/.../citywalker_test_v1 --split test \
  --package-id citywalker_test_v1
```

ms-swift direct CFM inference writes prediction/failure ledgers (checkpoint digest 必须是实际 immutable digest)：

```bash
python -m swift.sidewalk_nav.cli.infer_benchmark_plugin \
  --model <base-model> --checkpoint-dir <stage-ii-checkpoint> \
  --input-jsonl /volumes/.../citywalker_test_v1/inputs.jsonl \
  --prediction-jsonl /volumes/.../predictions.jsonl \
  --failure-jsonl /volumes/.../failures.jsonl \
  --run-audit-json /volumes/.../run_audit.json \
  --checkpoint-digest <sha256> --adaptation-mode zero_shot \
  --output-geometry-id citywalker_current_pose_xy_v1 --action-horizon 16
```

最后由 RouteServer 做唯一 canonical metric/report：

```bash
conda run -n UrbanNav env PYTHONPATH=.:src python \
  tools/evaluation/evaluate_benchmark_plugin.py \
  --package-json /volumes/.../citywalker_test_v1/package.json \
  --prediction-jsonl /volumes/.../predictions.jsonl \
  --failure-jsonl /volumes/.../failures.jsonl --output-root /volumes/.../scorecard
```

输出 `scorecard.json/csv/html` 与 `review.html`。review 按 stable sample id 保留 FPV 本地链接、GT/pred local-xy SVG 与 prompt，用于错误定位，不是训练或 benchmark 成功的证据。CityWalker package 的同一通用 renderer 使用 CityWalker 风格的两栏界面：左侧为按时间排序的 FPV（深色边框为 current frame），右侧是等比例局部坐标图；蓝色为公开 history、紫色为公开 point goal、绿色为 evaluator-only GT、红色为 prediction、深色点为 current origin。它只复用可视化语义，不 import、复制或依赖 CityWalker 代码；其他 package 缺失 `history_xy_m` 或 `target_xy_m` 时相应图层为空。

从 V1 review time split 起，轨迹图的实线只画到当前 metric score horizon；超过该 horizon 的可用 GT/pred
以浅色虚线展示。填充圆点明确标为 `GT @ 1s/2s`、`Pred @ 1s/2s`，而可用的完整 terminal 是单独的空心标记与
`terminal @ t` 标签。因此 CityWalker 五点 reference 的 terminal 不会再被视觉上误读成 2 秒 GT。此处只改变 review，
不改变任何 L2/coverage 计算。

当且仅当模型侧已经产生 trace sidecar 时，两个 review CLI 都可附加：

```bash
  --adapter-trace-jsonl /volumes/.../native_adapter_trace.jsonl \
  --require-complete-adapter-traces
```

它会额外写 `adapter_trace_funnel.json`，并将单 case 页面扩展为三栏：**model-visible UrbanNav input**（选中的
CityWalker 五帧以蓝框标出，其他仍可见但灰显）、**native-adapter audit**（完全 public-only）、以及
**evaluator-only geometry**（GT/pred/L2）。顶部 funnel 始终以完整 frozen denominator 依次报告 trace present、足够公开
FPV、公开 history/goal、公开 scale、input eligibility、native/output failure、prediction yielded 与每个 horizon jointly
scoreable；它是诊断产物，不是替代 scorecard 的新指标。
当前已经完成的 UrbanNav Stage-II→CityWalker zero-shot run 没有 native trace，因此仍使用二栏 generic review；本轮
只是为之后的真实 CityWalker→UrbanNav native adapter 预先冻结可审计 review contract，尚未声称已经运行该 baseline。

## 后验 slice review 与坐标诊断

当正式 ledger 已完成、需要区分 zero-shot transfer error 与输出 adapter 问题时，使用
`build_benchmark_plugin_slice_review.py`。它是严格的 evaluator-only 步骤：读取 hidden reference 来计算
rank strata，但绝不把 `scenario_labels`、GT、L2 或人工结论回写给模型输入。

```bash
conda run -n UrbanNav env PYTHONPATH=.:src python \
  tools/evaluation/build_benchmark_plugin_slice_review.py \
  --package-json /volumes/.../citywalker_validation/package.json \
  --prediction-jsonl /volumes/.../predictions.jsonl \
  --output-root /volumes/.../slice_audit \
  --slice-label turn --slice-label crossing \
  --slice-label action_target_mismatch --slice-label other \
  --horizon-s 2.0 --per-slice 20 --seed 20260808 \
  --copy-media --require-readable-media
```

工具固定地从每个 overlapping slice 的低/中/高 L2 rank band 抽样；同一个 stable sample 可有多个
`selection_reasons`，但 HTML 只生成一张卡。输出包含 portable `review.html`、local `media/`、
`selection.json`、待人工填写的 `diagnosis_template.jsonl`，以及两类非 official score：

- `coordinate_diagnostic.json`：在八种 global signed-permutation 中检查 identity 是否 material 地劣于某个
  axis/sign 变换。缺省门槛是绝对改善至少 `0.05 m` 且相对改善至少 `1%`；它只能触发 adapter 审计，不能追溯改变
  canonical score。
- `reference_motion_diagnostic.json`：按 hidden-reference 位移 `<0.25`、`0.25–1`、`1–3`、`>=3 m` 分桶，显示
  GT/pred radial displacement 与 L2。它不是 stop metric 或新监督标签，而是用来发现“近静止状态预测持续前行”这类
  transfer/interface 失配。

review 必须先完成逐案 diagnosis，再决定是修 adapter/prompt、冻结能力匹配的 2 秒 profile，还是注册独立的
CityWalker fine-tune 对照；不能只凭某一个 scenario AMDE 归因于模型能力。

## V1 验证状态

本轮已完成 schema、package IO、确定性 CityWalker builder、CityWalker/UrbanNav metrics、generated mirror、target-free adapter、slice review/coordinate diagnostic 和 focused tests。`citywalker_zeroshot_stage_ii_cfm_validation_20260808` 已在冻结 validation
sequence `09` 完成 568-case Stage-II checkpoint inference、canonical scorecard 与 post-hoc review；精确数值、
receipt 和限制见 ms-swift 的实验记录。尚未运行 CityWalker hidden test、CityWalker fine-tune 或训练，因此该真实
validation 结果只能是可复现 zero-shot baseline，不是最终 benchmark 或 target-domain adaptation 结论。
