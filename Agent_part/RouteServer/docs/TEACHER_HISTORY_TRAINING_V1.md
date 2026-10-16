# Teacher-owned history training views v1

## 2026-10-16 决策

用户明确选择 teacher Decision 作为最终 Stage I 监督，使用独立版本合同。
新合同 `urbannav_teacher_stage_i_v1` 在 UrbanNav-Contracts 中定义；旧 GT-owned
ReasoningTrainingViewV4 和 StageIFullCoTViewV2 保持原样。

GT 轨迹及 action_mask 继续在 Stage II sidecar 中监督。新 Stage I exchange 不携带
轨迹、GT方向或伪造的 strong-direction stamp；teacher/GT 差异不能靠覆盖任一方消除。

## 合并规则

1. 沿用已有自身ID和时间映射，读取冻结源集合与后续补标；历史QC状态不用于丢弃旧回答。
2. 覆盖缺口的7649条 teacher Decision，再应用189条定向复核的已解决结果。
3. corrected 仅替换具名section；consistent 保留原文；pending/解析异常保留独立清单。
4. 保留原始来源、ID、时间差、修订链和raw回答位置；不覆盖旧文件。
5. 沿用原有clip split，媒体路径相对 corpus root；纳秒源时钟与相对时间同时保留。

## 方向语义

Decision采用ego-metric 12-way，旧teacher动作50至90度按定义归入50plus，并记录映射。
这属于词表投影，不是从GT计算新动作。

旧LocalTraversability的left_90/right_90是具名可通行性探测方向，不能直接声称它们代表
整个50plus角域。v1通过context_direction_profile显式保留原grounding词表，而Decision
使用12-way。这个兼容合同不等于所有grounding已经迁成ego_12way probe profile；
需要统一全部探测方向时，应另行确定其语义迁移，不批量改名。

## 消费者

`ms-swift/swift/sidewalk_nav/data/stage_i/teacher_stage_i_manifest.py` 将exchange转换为
train/validation/test JSONL；原stage_i_manifest入口也按schema显式分派。
复用六section renderer和V3.1 reasoning/decision loss transport，不复用GT composer。
系统/用户输入只包括新图像、角色、物理时间、instruction和公共词表，旧答案和来源不进入prompt。

本轮不更改旧epoch配额compiler、不启动训练、不重新导出VQA，也不宣称新数据全量视觉校准。
label_status区分迁移teacher与局部复核teacher；待处理项仍保留在pending.jsonl。

## 执行与限制

189条视觉复核在32011的GPU2–5运行，tmux为vla-decision-review-20261016。
复核结束后vla-teacher-export-20261016串行执行合并和消费者转换。
服务器根目录：`/lpai/volumes/mind-eb-bd-ga/suh/UrbanNavigation/annotation_runs/vla_decision_review_20261016`。
完成以reviews/report和export/finished.json为准，代码通过测试不是远端完成证明。
