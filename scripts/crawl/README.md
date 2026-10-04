# 语料采集说明（子 Agent 爬数据 · 可复现）

## 本批语料如何产生

`data/corpus/incoming/batch-0*.json`（共 6 批 59 篇）由 **6 个并行爬虫子 Agent** 于 2026-09-17 采集，
每篇正文均来自对真实 URL 的实际抓取（WebSearch 定位 + WebFetch/页面抓取），无编造内容。

子 Agent 任务书要点（每批相同，仅主题不同）：

1. 用 WebSearch 找候选新闻 URL，WebFetch 打开提取标题/发布日期/正文纯文本；
2. 正文保留产品名、日期、价格、金额等关键事实句（300–3000 字）；
3. 采集目标场景：
   - **同一事件多来源报道**（测事件归并：同一次调价/发布会 2–3 家媒体各一篇）；
   - **同一实体多次动作**（测事件区分：同品牌先后两次调价/两场发布）；
   - **一篇多事件**（测 EventMention 拆分：早报/一周汇总类）；
   - **同交易多阶段**（测"宣布≠交割"：并购 预案→批复→交割）；
4. 输出 JSON 到 `data/corpus/incoming/batch-XX-*.json`，字段：
   `url / title / published_at / fetched_at / source_name / language / content`。

## 各批次覆盖

| 批次 | 主题 | 篇数 | 关键测试场景 |
|---|---|---|---|
| batch-01 | 手机/数码产品发布 | 12 | 华为 Mate 80、iPhone 17、小米17 发布会各 2–3 家媒体报道（同事件归并）；华为/小米各自两场不同发布（区分） |
| batch-02 | 新能源车价格调整 | 12 | 特斯拉 4 次不同调价（同过程多事件）、比亚迪 2 次调价、蔚来电池包调价 2 家媒体、理想优惠 2 家媒体 |
| batch-03 | 新车上市 | 10 | YU7/汉L/萤火虫 上市各 2 家媒体；小米 YU7 vs YU7 GT 区分；两篇多车型汇总 |
| batch-04 | 专利公开与诉讼 | 9 | 大疆诉影石 3 家媒体同事件；汉王诉熵基 2 家媒体；华为专利公开 |
| batch-05 | 并购合作与人事 | 9 | 国泰君安合并海通三阶段（预案/批复/交割）；紫金控股藏格两阶段；上汽×华为两阶段；荣耀换帅 |
| batch-06 | 多事件综合报道 | 7 | IT 早报（单篇约 28 个事件）、苹果发布会汇总、一周新车、半导体综述 |

## 复跑采集（可选）

语料已随仓库提交，**复现流水线不需要重新爬取**（重放同一语料 → 逐字一致结果）。
如需补充新语料，可按上述任务书再派子 Agent，或将新 JSON 放入 `data/corpus/incoming/`
后执行 `python3 scripts/run_pipeline.py --reset`。

也可以直接调 API 提交资料：

```bash
curl -X POST http://127.0.0.1:8600/api/ingest -H 'Content-Type: application/json' \
  -d '{"items":[{"url":"https://example.com/news","title":"示例","published_at":"2026-09-17",
       "source_name":"示例站","content":"正文……"}]}'
```
