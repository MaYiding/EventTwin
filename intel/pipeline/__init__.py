# -*- coding: utf-8 -*-
"""流水线阶段包：ingest → extract → entities → vectorize → resolve → assertions
→ relations → project。runner.py 负责编排与幂等。（v3.1：向量层回归 embedding-8b，
判定归 Jev。）"""
