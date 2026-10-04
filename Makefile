# 企业外部情报系统 —— 常用命令
PY ?= python3
PORT ?= 8600

.PHONY: serve pipeline reset fresh smoke check

serve:            ## 启动 API + Web 观测台（http://127.0.0.1:8600）
	$(PY) -m intel.server --port $(PORT)

pipeline:         ## 全量跑流水线（幂等；LLM 缓存命中时为快速重放）
	$(PY) scripts/run_pipeline.py

reset:            ## 清库重放（保留 LLM 缓存 → 结果逐字一致，秒级）
	$(PY) scripts/run_pipeline.py --reset

fresh:            ## 清库 + 清 LLM 缓存，真实重算（消耗 GPU）
	$(PY) scripts/run_pipeline.py --reset --fresh

smoke:            ## CRUD 增删改查 + 问答接口冒烟（需先 make serve）
	bash scripts/crud_smoke.sh

check:            ## 语法与导入自检
	$(PY) scripts/selfcheck.py
