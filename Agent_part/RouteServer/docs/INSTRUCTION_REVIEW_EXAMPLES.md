# Instruction Review Examples

本文档用于 review 第一版重构后的需求指令类型。当前只 review 指令、类别和目标类型映射，不包含 POI 检索结果、目的地真值或路线真值。

## POI/品牌导航 / POI and Brand Navigation (`poi_navigation`)

合并来源：CitySeeker `question1` 基础 POI + `question4` 品牌特定导航。

| subcategory | 中文指令 | English instruction | 目标类型 / 品牌约束 | CitySeeker 来源 |
| --- | --- | --- | --- | --- |
| `bank` | 请帮我找最近的银行。 | Please find the nearest bank. | `bank` | `question1` |
| `restaurant` | 请帮我找最近的餐厅。 | Please find the nearest restaurant. | `restaurant` | `question1` |
| `convenience_store` | 请帮我找最近的便利店。 | Please find the nearest convenience store. | `convenience_store` | `question1` |
| `healthcare_facility` | 请帮我找最近的医疗设施。 | Please find the nearest healthcare facility. | `hospital`, `clinic`, `pharmacy` | `question1` |
| `hotel` | 请帮我找最近的酒店。 | Please find the nearest hotel. | `hotel` | `question1` |
| `school` | 请帮我找最近的中小学。 | Please find the nearest primary and secondary school. | `school` | `question1` |
| `shopping_mall` | 请帮我找最近的购物中心。 | Please find the nearest shopping mall. | `shopping_mall`, `mall` | `question1` |
| `starbucks` | 我想喝杯星巴克，帮我找最近的一家。 | Please find the nearest Starbucks. | `cafe`; `Starbucks`, `星巴克` | `question4` |
| `kfc_or_mcdonalds` | 附近有肯德基或麦当劳吗？ | Please find the nearest KFC or McDonald's. | `fast_food/restaurant`; `KFC`, `McDonald's`, `肯德基`, `麦当劳` | `question4` |
| `bank_of_china` | 请帮我找最近的中国银行。 | Please find the nearest Bank of China. | `bank`; `Bank of China`, `中国银行` | `question4` |
| `apple_or_huawei` | 请帮我找最近的苹果或华为门店。 | Please find the nearest Apple or Huawei store. | `electronics_store/mobile_phone_shop`; `Apple`, `Huawei`, `苹果`, `华为` | `question4` |
| `7_eleven_or_familymart` | 请帮我找最近的 7-Eleven、全家、罗森或便利蜂。 | Please find the nearest 7-Eleven, FamilyMart, Lawson, or Bianlifeng. | `convenience_store`; `7-Eleven`, `FamilyMart`, `Lawson`, `Bianlifeng`, `全家`, `罗森`, `便利蜂` | `question4` |

## 隐式需求导航 / Implicit Need Navigation (`implicit_need_navigation`)

合并来源：CitySeeker `question6` Latent POI + `question7` Abstract Needs。这里既包括“洗手间/ATM”等不总是独立可搜的设施，也包括“饿了/渴了/手机没电”等不直接出现 POI 名词的需求。

| subcategory | 中文指令 | English instruction | 推断目标类型 | CitySeeker 来源 |
| --- | --- | --- | --- | --- |
| `restroom` | 我想找个洗手间。 | Please find the nearest restroom. | `toilets`, `restroom`, `shopping_mall`, `fast_food`, `public_facility` | `question6` |
| `atm` | 我需要取点现金，帮我找附近可以取钱的地方。 | Please find the nearest ATM. | `atm`, `bank` | `question6` |
| `parking_lot` | 我需要找个地方停车。 | Please find the nearest parking lot. | `parking`, `parking_lot` | `question6` |
| `gym` | 我想锻炼，帮我找附近合适的地方。 | Please find the nearest gym. | `gym`, `fitness_centre`, `park` | `question6` |
| `coffee_shop` | 我想找个地方喝咖啡。 | Please find the nearest coffee shop. | `cafe`, `coffee_shop` | `question6` |
| `cinema` | 我想看电影，帮我找附近合适的地方。 | Please find the nearest cinema. | `cinema`, `movie_theater` | `question6` |
| `hungry` | 我有点饿了。 | I'm feeling hungry and would like something to eat. Could you help me find a nearby place? | `restaurant`, `fast_food`, `food_court` | `question7` |
| `thirsty` | 我有点渴了。 | I'm feeling thirsty and would like something to drink. Could you help me find a nearby place? | `convenience_store`, `cafe`, `beverage_shop` | `question7` |
| `phone_battery_or_recharge` | 手机没电了，找个地方坐着充会儿电。 | I need a place to sit and charge my phone. | `cafe`, `library`, `coworking_space`, `convenience_store`, `mobile_phone_shop` | `question7` |
| `child_play` | 我想带孩子玩，帮我找附近合适的地方。 | I want to bring my child to play and need assistance finding a suitable place nearby. Could you help me? | `playground`, `park`, `children_activity_center` | `question7` |
| `not_feeling_well` | 我身体不太舒服，帮我找附近合适的医疗点。 | I'm not feeling well and need assistance finding a suitable place nearby. Could you help me? | `clinic`, `hospital`, `pharmacy` | `question7` |
| `work_with_wifi` | 我想找个有 Wi-Fi 的地方办公。 | I want to work with Wi-Fi and need assistance finding a suitable place nearby. Could you help me? | `cafe`, `library`, `coworking_space` | `question7` |
| `buy_fruits_vegetables` | 我想买水果蔬菜。 | I want to buy fruits and vegetables and need assistance finding a suitable place nearby. Could you help me? | `supermarket`, `market`, `greengrocer` | `question7` |
| `rest_and_read` | 找个安静的地方让我看看书。 | I want to rest and read and need help finding a suitable place nearby. Could you assist me? | `library`, `bookstore`, `park`, `cafe` | `question7` |

## 多目标导航 / Multi-Stop Navigation (`multi_stop_route`)

来源：CitySeeker `question2` 顺序多目标 + 本仓库扩展的自然语言多目标路线。真值为 `gt_destination_sequence`、`gt_route_legs` 和合并后的 `gt_route`。

| subcategory | 中文指令 | English instruction | 目标顺序 | 来源 |
| --- | --- | --- | --- | --- |
| `restaurant_then_bank` | 请先带我去最近的餐厅，然后去最近的银行。 | Please find the nearest restaurant and bank in order. | `restaurant` -> `bank` | `question2` |
| `restaurant_then_convenience_store` | 请先带我去最近的餐厅，然后去最近的便利店。 | Please find the nearest restaurant and convenience store in order. | `restaurant` -> `convenience_store` | `question2` |
| `clinic_then_hotel` | 请先带我去最近的诊所，然后去最近的酒店。 | Please find the nearest clinic and hotel in order. | `clinic/hospital` -> `hotel` | `question2` |
| `eat_then_buy_daily_goods` | 帮我规划一条路线，先找个地方吃饭，然后再去买点日用品。 | Plan a route where I can eat first and then buy some daily goods. | `restaurant/fast_food` -> `convenience_store/supermarket` | 仓库扩展 |
| `coffee_then_work` | 帮我规划一条路线，先买咖啡，然后找个适合办公的地方。 | Plan a route where I can get coffee first and then find a place to work. | `cafe/coffee_shop` -> `library/coworking_space/cafe` | 仓库扩展 |
| `atm_then_shopping` | 帮我规划一条路线，先取钱，然后去购物。 | Plan a route where I can withdraw cash first and then go shopping. | `atm/bank` -> `shopping_mall/supermarket` | 仓库扩展 |

## 语义偏好导航 / Semantic Preference Navigation (`semantic_preference`)

保留但建议作为后续阶段或人工审核集，因为 `romantic/upscale/comfortable` 这类真值依赖评论、标签或人工判断。

| subcategory | 中文指令 | English instruction | 目标类型 / 语义标签 |
| --- | --- | --- | --- |
| `comfortable_restaurant` | 请帮我找最近环境舒适的餐厅。 | Please find the nearest restaurant with a comfortable environment. | `restaurant`; `comfortable`, `quiet`, `good_environment` |
| `family_friendly_restaurant` | 请帮我找最近适合家庭或多人聚餐的餐厅。 | Please find the nearest group or family-friendly restaurant. | `restaurant`; `family_friendly`, `group_friendly` |
| `restaurant_with_parking` | 请帮我找最近方便路边停车的餐厅。 | Please find the nearest restaurant with roadside parking. | `restaurant`; `roadside_parking`, `parking` |
| `upscale_restaurant` | 请帮我找最近的高档餐厅。 | Please find the nearest upscale restaurant. | `restaurant`; `upscale`, `fine_dining` |
| `outdoor_seating_restaurant` | 请帮我找最近有户外座位的餐厅。 | Please find the nearest restaurant with outdoor seating. | `restaurant`; `outdoor_seating` |
| `romantic_restaurant` | 我想找一家氛围浪漫的餐厅约会。 | Please find the nearest romantic restaurant. | `restaurant`; `romantic` |

## 中途增加途经点 / Add Via Point Mid-Route (`route_update_add_via`)

这是本仓库扩展类型，不来自 CitySeeker。它评估已有路线状态上的增量修改。

| subcategory | 中文指令 | English instruction | 新增途经点 / 工具 |
| --- | --- | --- | --- |
| `add_coffee_stop` | 在去目的地之前，帮我中途加一个咖啡店。 | Add a coffee shop stop before continuing to the destination. | `cafe/coffee_shop`; `add_via_point_to_session` |
| `add_atm_stop` | 在当前路线中途加一个附近的 ATM。 | Add a nearby ATM as a stop on the way. | `atm/bank`; `add_via_point_to_session` |
| `add_convenience_store_stop` | 到最终目的地之前，帮我加一个便利店作为途经点。 | Add a convenience store stop before the final destination. | `convenience_store`; `add_via_point_to_session` |

## 删除的 CitySeeker 类型

| 原类型 | 处理 | 原因 |
| --- | --- | --- |
| 交通枢纽导航 / Transit Hub Navigation | 删除 | 当前任务不重点评估公交/地铁接驳，和一般 POI 检索差异不大 |
| 无障碍设施导航 / Accessibility Needs | 删除 | 真值依赖可靠 accessibility 标签或人工审核，第一版不做 |

## Review Notes

1. `poi_navigation` 是否还需要保留学校、酒店、医疗设施等子类。
2. `implicit_need_navigation` 的每个需求是否能稳定映射到一组可检索 POI。
3. `multi_stop_route` 的目标顺序是否符合我们真实任务场景。
4. `semantic_preference` 是否暂时只做小规模人工审核集。
5. `route_update_add_via` 是否需要区分“追加到终点前”和“插入到指定两站之间”。
