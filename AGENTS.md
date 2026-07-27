# Urban-Nav-SR 仓库说明

本仓是研究代码组合层的两个主框架之一，负责 navigation runtime、canonical contracts、
annotation 以及 benchmark/evaluation。

## 组合层路由

- 跨仓工作先阅读 `../ResearchWorkflow/AGENTS.md`、
  `../ResearchWorkflow/workstreams/core-framework/STATUS.html` 和
  `../ResearchWorkflow/policies/policy-matrix.json`。
- 项目自有 Python 修改遵循
  `../ResearchWorkflow/policies/python-code-quality.md`。
- dataset、benchmark 和 evaluator 修改遵循
  `../ResearchWorkflow/policies/benchmark-evaluation.md`。
- `Agent_part/RouteServer/AGENTS.md` 对该子目录提供更严格说明，并继续作为其领域规则的
  本地权威。

## 仓库责任

本仓负责：

- RouteServer tools、providers、route sessions 和 runtime traces。
- Canonical reasoning、tool-use、data 和 evaluation contracts。
- Dataset construction、teacher annotation、validation、audit 和 export。
- UrbanNav ground truth、benchmark protocols、metrics 和 evaluation runners。

本仓不负责：

- ms-swift trainer internals、LM loss masking 或 checkpoint sidecars。
- SideWalkNavigation portfolio/release manifests。
- InternNav 或其他参考仓的上游实现。

与 ms-swift 通过 versioned canonical schemas、generated mirrors、exporters 和
cross-repository contract tests 交换行为。Runtime/audit traces、trainable views
和 evaluation views 必须保持为不同产物。

## 代码与证据

- 遵循共享的文档、架构和 focused-test 规范。
- 当前 RouteServer `pyproject.toml` 未定义 formatter。应匹配本地风格，不能通过偶然修改
  引入或声称存在仓库级 formatter。
- 用于正式证据的 evaluator 必须 offline 且 deterministic。
- 在当前 workflow run manifest 中记录精确 data、model、command、environment、
  commit 和 artifact。
- service model listing、synthetic fixture 或 mini smoke 不能当作 semantic 或
  frozen-benchmark validation。

## Git 与外部门禁

- 保留无关 worktree 变更，并按责任域组织 commit。
- 外部数据传输、paid annotation、push、tag、publication 和 release 需要明确授权。
- 禁止在 code、documentation 或 workflow evidence 中记录 credential 或 private raw data。
