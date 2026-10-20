# UrbanNav final-25K build record

This document records the reproducible controlled-replay UrbanNav 25K build.
Generated datasets, route caches, logs, and collection shards remain under the
ignored `data/` tree; source code, policies, distribution plans, and validation
rules are version controlled.

## Current package: v3, 23 cities

The current package was finalized and independently validated on 2026-07-15:

```text
data/benchmark/urbannav_v3_osm_local_osrm_foot_final25k_v3_23cities
```

It contains:

```text
task_gt.jsonl          25,000 rows
poi_cache.jsonl        31,862 rows
route_cache.jsonl     201,005 rows
task_rejects.jsonl     41,116 audit rows
review sample             500 rows
```

Reject rows describe filtered collection attempts and are not members of the
25K evaluation set. The final validator reports 25,000 valid cases, no invalid
cases, and no missing cache references.

### Exact distribution

| Category | Cases | Share |
| --- | ---: | ---: |
| `poi_navigation` | 10,000 | 40% |
| `implicit_need_navigation` | 8,750 | 35% |
| `multi_stop_route` | 3,750 | 15% |
| `route_update_add_via` | 2,500 | 10% |

The package covers 23 cities. Every city contains 1,087 cases except Toronto,
which contains 1,086. Route-update actions are fixed to:

```text
replace_destination  1,000
add_via              1,000
insert_before_final    500
```

The distribution source of truth is
`config/urbannav_25k_v3_23cities_distribution_plan.json`.

### Quality contract

The package enforces these invariants:

- OSM POI indices include nodes, ways, and relations.
- New-city origins are sampled around POI-density hotspots while sharing one
  city-level pedestrian router.
- Unsupported `city x target_type` instructions are removed before routing.
- Multi-stop labels are unique and target-type groups are disjoint.
- Stops and accepted route legs are at least 20 m apart/long.
- Every later multi-stop leg stores a predecessor-to-candidate route matrix.
- Replay route lookup rejects cached origins that differ by more than 5 m.
- Evaluation requires adjacent predicted route legs to be continuous.
- Add/insert route updates preserve the final destination.
- Duplicate cases and conflicting cache records fail finalization.

The shared policy is
`config/urbannav_25k_v2_quality_policy.json`; the policy version is retained
because v3 extends rather than replaces the v2 quality contract.

### Build entry point

Run from `Agent_part/RouteServer`:

```bash
tools/dataset_processing/run_urbannav_25k_v3_23cities.zsh asset-plan
tools/dataset_processing/run_urbannav_25k_v3_23cities.zsh assets
tools/dataset_processing/run_urbannav_25k_v3_23cities.zsh prepare
tools/dataset_processing/run_urbannav_25k_v3_23cities.zsh smoke
tools/dataset_processing/run_urbannav_25k_v3_23cities.zsh collect
tools/dataset_processing/run_urbannav_25k_v3_23cities.zsh finalize
```

`all` runs the same primary sequence. Collection shards support resume.
Finalization uses seed `20260716` and also consumes the documented v2 and
Shenzhen supplement roots when they are present.

## Historical package: v2, 17 cities

The v2 package remains available at:

```text
data/benchmark/urbannav_v2_osm_local_osrm_foot_final25k_v2
```

Its reproducible entry point is
`tools/dataset_processing/run_urbannav_25k_v2_build.zsh`. It established the
40/35/15/10 category split, exact route-update action quotas, semantic
deduplication, full multi-stop route matrices, and strict continuity checks.
The v3 package adds six cities and POI-hotspot collection without weakening
those constraints.

## Validation

The strict validator command used by both build scripts checks:

```text
JSONL/schema validity
stable case-id uniqueness
POI and route cache references
multi-stop target-type disjointness
minimum stop and leg distance
full predecessor-to-candidate route matrices
multi-leg geometric continuity
exact category, city, and route-update quotas
```

The final package must not be published or benchmarked when
`validation_report.json` contains `valid=false`.
