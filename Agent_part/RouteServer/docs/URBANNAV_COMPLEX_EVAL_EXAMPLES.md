# UrbanNav Complex Evaluation Examples

本文档展示一些比 fake smoke 更复杂的评估情况，重点说明：

- 预测选择的目的地不是 rank-1 GT 时，什么时候仍算成功。
- 命中合理目的地但路线绕路时，路线效率如何扣分。
- top-1 目的地错误但 top-5 召回正确目标时，`hit@5` 和 `destination_success` 如何区分。
- 没有稳定 POI id 时，坐标 fallback 如何生效。
- route-update 样例如何沿用同一套主指标。

对应文件：

```text
tests/fixtures/urbannav_complex_gt_6.jsonl
tests/fixtures/urbannav_complex_pred_6.jsonl
```

复现命令：

```bash
cd Agent_part/RouteServer

PYTHONPATH=src python tools/evaluation/eval_urbannav_destination_route.py \
  --ground-truth tests/fixtures/urbannav_complex_gt_6.jsonl \
  --predictions tests/fixtures/urbannav_complex_pred_6.jsonl \
  --output-json /tmp/urbannav_complex_examples_summary.json \
  --output-jsonl /tmp/urbannav_complex_examples_rows.jsonl

PYTHONPATH=src python tools/evaluation/build_urbannav_review_html.py \
  --ground-truth tests/fixtures/urbannav_complex_gt_6.jsonl \
  --predictions tests/fixtures/urbannav_complex_pred_6.jsonl \
  --eval-rows /tmp/urbannav_complex_examples_rows.jsonl \
  --output-dir /tmp/urbannav_complex_examples_html \
  --title "UrbanNav Complex Eval Examples"
```

HTML review：

```text
/tmp/urbannav_complex_examples_html/index.html
```

## Summary

本组 6 条样例的评估结果：

| Metric | Value |
| --- | ---: |
| `total_cases` | 6 |
| `destination_success` | 0.6667 |
| `hit@5` | 1.0 |
| `route_success` | 0.6667 |
| `median_route_ratio` | 1.0077 |
| `spl` | 0.6308 |

`hit@5=1.0` 说明每条样例的候选列表里都召回了合理目标；但只有 4/6 的 top-1
`chosen_destination` 命中 acceptable destination，所以 `destination_success=0.6667`。

## Case Table

| Case | 情况 | 预测选择 | Matched rank | DS | Hit@5 | RS | Route ratio | SPL |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `complex-001` | 非 rank-1 但同样合理 | 咖啡店 Drink B | 2 | 1 | 1 | 1 | 1.0 | 1.0 |
| `complex-002` | 非 rank-1 合理目标，但路线绕路 | 安静咖啡 B | 2 | 1 | 1 | 1 | 1.25 | 0.8 |
| `complex-003` | top-1 错，top-5 有正确目标 | 错误餐厅 | - | 0 | 1 | 0 | - | 0 |
| `complex-004` | 无 stable id，name/type/coordinate fallback | 中国银行24小时自助银行 | 1 | 1 | 1 | 1 | 1.0154 | 0.9848 |
| `complex-005` | route update，选择非 rank-1 但合理更新目标 | 便利店 Update B | 2 | 1 | 1 | 1 | 1.0 | 1.0 |
| `complex-006` | 在 ranked 里但不 acceptable，因“最近 ATM”约束失败 | 远处 ATM | - | 0 | 1 | 0 | - | 0 |

## 1. 非 rank-1 但同样合理

`complex-001` 的需求是：

```text
我想买点喝的，便利店或咖啡店都可以。
```

GT 中有两个 acceptable destinations：

```text
rank 1: 便利店 Drink A, reference_route_distance_m = 100
rank 2: 咖啡店 Drink B, reference_route_distance_m = 108
```

预测选择 rank-2 的咖啡店：

```text
chosen_destination = drink-b
```

因为 `drink-b` 在 `acceptable_destinations` 里，所以：

\[
DS = 1
\]

路线也到达 `drink-b`，并且预测距离等于 `drink-b` 自己的参考距离：

\[
route\_ratio = \frac{108}{108} = 1
\]

这里不会用 rank-1 的 100m 作为分母。

## 2. 非 rank-1 合理目标，但路线绕路

`complex-002` 预测选择 rank-2：

```text
安静咖啡 B
```

它是 acceptable destination，所以：

\[
DS = 1,\quad RS = 1
\]

但预测路线距离是 250m，而该 matched destination 的参考路线是 200m：

\[
route\_ratio = \frac{250}{200}=1.25
\]

\[
SPL = \frac{200}{\max(200,250)}=0.8
\]

这说明目的地选择正确，但路线效率被扣分。

## 3. top-1 错，但 top-5 召回正确目标

`complex-003` 的 top-1 是错误餐厅：

```text
chosen_destination = food-wrong
```

它不在 `acceptable_destinations`，所以：

\[
DS=0,\quad RS=0,\quad SPL=0
\]

但候选列表里包含正确目标 `food-a`：

```text
candidate_destinations = [food-wrong, food-a]
```

因此：

\[
Hit@5=1
\]

这类样例用于区分“检索召回能力”和“最终选择能力”。

## 4. 无 stable id 的坐标 fallback

`complex-004` 的 GT 没有 `provider_poi_id`，只有：

```text
name = 中国银行24小时自助银行
type = atm
lat/lng = 30.0, 120.001
```

预测也没有 stable id，但 name/type 一致，坐标误差约 1.47m：

```text
destination_match_method = name_type_coordinate
endpoint_error_m = 1.47
```

因此仍算目的地命中。

这个 fallback 只在 stable id 缺失时使用，并且要求 name/type 兼容，避免仅凭坐标误匹配。

## 5. route update 中选择非 rank-1 合理目标

`complex-005` 是 route update：

```text
previous_instruction = 先去咖啡店。
updated_instruction = 我临时想改去可以买水的地方。
current_origin = 30.0005, 120.0005
```

评估时使用更新后的：

\[
q_{\mathrm{eff}} = q_{\mathrm{updated}}
\]

\[
o_{\mathrm{eff}} = o_{\mathrm{current}}
\]

预测选择 rank-2 的 `update-b`，它在 `acceptable_destinations` 内，所以仍然成功：

```text
matched_destination_rank = 2
destination_success = true
route_success = true
```

## 6. ranked 里有，但不 acceptable

`complex-006` 是：

```text
帮我找最近的 ATM。
```

ranked candidates 里有：

```text
rank 1: 近处 ATM, 100m
rank 2: 远处 ATM, 430m
```

但 `acceptable_destinations` 只有 rank-1，因为“最近的 ATM”约束更强。预测选择
rank-2 的远处 ATM：

```text
chosen_destination = atm-far
```

虽然它是同类型目标，也在 `ranked_destinations` 里，但不在 `acceptable_destinations`：

\[
DS=0
\]

候选里包含正确的近处 ATM，因此：

\[
Hit@5=1
\]

这说明 `ranked_destinations` 是审计和分析列表，不等价于正确答案集合。真正用于目的地成功判断的是
`acceptable_destinations`。
