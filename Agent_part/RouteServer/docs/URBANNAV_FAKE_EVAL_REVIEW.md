# UrbanNav Fake Evaluator Review Set

这是一组手写的 5 条 fake GT/pred，用来人工 review Step 3 的 evaluator 语义。它不依赖真实地图 API，也不代表正式 benchmark 分布。

## Files

```text
tests/fixtures/urbannav_fake_gt_5.jsonl
tests/fixtures/urbannav_fake_pred_5.jsonl
```

运行：

```bash
cd /home/ubuntu/WorkSpace/project/SR/Code/Urban-Nav-SR/Agent_part/RouteServer

PYTHONPATH=src python tools/evaluation/eval_urbannav_destination_route.py \
  --ground-truth tests/fixtures/urbannav_fake_gt_5.jsonl \
  --predictions tests/fixtures/urbannav_fake_pred_5.jsonl \
  --output-json /tmp/urbannav_fake_summary.json \
  --output-jsonl /tmp/urbannav_fake_rows.jsonl
```

## Per-case Expected Behavior

| case | 目的 | expected |
| --- | --- | --- |
| `fake-001` | rank-1 目的地命中，路线距离等于参考距离 | `destination_success=true`, `hit@5=true`, `route_success=true`, `route_ratio=1.0`, `spl=1.0` |
| `fake-002` | 命中非 rank-1 但仍可接受的目的地 `poi-b2` | `destination_success=true`, `hit@5=true`, `route_success=true`, `matched_destination_rank=2`, `reference_route_distance_m=200`, `route_ratio=1.2`, `spl=0.833333...` |
| `fake-003` | Top-1 选错，但 Top-5 里有正确目标 | `destination_success=false`, `hit@5=true`, `route_success=false`, `spl=0` |
| `fake-004` | 目的地命中，但路线终点离 matched destination 太远 | `destination_success=true`, `hit@5=true`, `route_success=false`, `spl=0` |
| `fake-005` | 目的地命中，但无显式候选列表且无路线 | `destination_success=true`, `hit@5=true`, `route_success=false`, `spl=0` |

## Expected Summary

```json
{
  "total_cases": 5,
  "destination_success": 0.8,
  "hit@5": 1.0,
  "route_success": 0.4,
  "median_route_ratio": 1.1,
  "spl": 0.3666666666666667,
  "missing_prediction_count": 0
}
```

说明：

- 如果没有显式 `candidate_destinations`，loader 会把 `chosen_destination` 作为 Top-1 fallback，因此 `fake-005` 的 `hit@5=true`。
- `fake-002` 是最关键样例：预测的是 rank-2 acceptable destination，因此路线分母必须使用 `poi-b2.reference_route_distance_m=200`，不能使用 rank-1 的 150。

## Actual Review Result

已用当前 evaluator 实际运行：

```bash
PYTHONPATH=src PYTHONPYCACHEPREFIX=/tmp/pycache_urbannav_fake \
python tools/evaluation/eval_urbannav_destination_route.py \
  --ground-truth tests/fixtures/urbannav_fake_gt_5.jsonl \
  --predictions tests/fixtures/urbannav_fake_pred_5.jsonl \
  --output-json /tmp/urbannav_fake_summary.json \
  --output-jsonl /tmp/urbannav_fake_rows.jsonl
```

实际 summary：

```json
{
  "total_cases": 5,
  "destination_success": 0.8,
  "hit@5": 1.0,
  "route_success": 0.4,
  "median_route_ratio": 1.1,
  "spl": 0.3666666666666667,
  "missing_prediction_count": 0
}
```

逐条人工检阅表：

| case | DS | hit@5 | RS | route_ratio | SPL | matched | 人工结论 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `fake-001` | true | true | true | 1.0 | 1.0 | rank 1 `poi-a` | 正常通过；rank-1 目的地和路线都正确。 |
| `fake-002` | true | true | true | 1.2 | 0.8333333333333334 | rank 2 `poi-b2` | 正常通过；这是多合理目的地样例，路线分母使用 `poi-b2` 的 200m。 |
| `fake-003` | false | true | false | null | 0.0 | none | 符合预期；Top-1 目的地错，但 Top-5 候选召回了 GT。 |
| `fake-004` | true | true | false | null | 0.0 | rank 1 `poi-d` | 符合预期；目的地命中，但路线终点偏离约 866.68m。 |
| `fake-005` | true | true | false | null | 0.0 | rank 1 `poi-e` | 符合预期；目的地命中，但缺少路线，记录 `missing_route_polyline`。 |

人工检阅结论：

```text
PASS
```

当前 5 条 fake 样例覆盖了 evaluator 最关键的边界：

- rank-1 正常命中。
- 非 rank-1 acceptable destination 命中。
- Top-1 错但 Top-5 召回正确目标。
- 目的地正确但路线终点错误。
- 目的地正确但路线缺失。
