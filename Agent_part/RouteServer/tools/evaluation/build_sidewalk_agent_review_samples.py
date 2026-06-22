#!/usr/bin/env python3
"""Compatibility wrapper for build_sidewalk_agent_stage2_review_samples.py.

新代码请直接使用:

```bash
PYTHONPATH=src python tools/evaluation/build_sidewalk_agent_stage2_review_samples.py
```
"""

from __future__ import annotations

from build_sidewalk_agent_stage2_review_samples import main


if __name__ == "__main__":
    main()
