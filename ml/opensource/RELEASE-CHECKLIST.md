# 发布记录：EventTwin v1.0（已完成 ✅ 2026-10-02）

## 已发布

| 平台 | 地址 | 内容 | 状态 |
|---|---|---|---|
| GitHub | https://github.com/MaYiding/EventTwin | 双语 README（英文默认+可折叠中文）/ inference.py / MODEL_CARD / VERSIONS / docs×2 / LICENSE(Apache 2.0) | ✅ 页面验证通过 |
| HF 模型 | https://huggingface.co/MaYiding/EventTwin | 权重 model.safetensors(1.1GB, LFS) + tokenizer×2(LFS) + config + **calibration.json(T=0.824)** + 模型卡（YAML/widget/base_model 树生效） | ✅ 7 文件齐全 |
| HF 数据集 | https://huggingface.co/datasets/MaYiding/EventTwin-Data | benchmark-1k.jsonl（909KB，双教师金标）+ train-pairs-24k.jsonl（20MB，软标签）+ 数据卡（CC BY-NC 4.0，viewer 已自动识别） | ✅ 页面验证通过 |

## 版本映射（内外解耦）

公开 **EventTwin v1.0** = 内部学生模型 **L2 v8**；EventTwin-Data v1.0 = 内部 benchmark
`bm_v1_dual` + 训练配方 v8。映射与路线图见仓库 VERSIONS.md。内部 v9（合成扩产）对应
未来公开 v1.1，发布时从 `ml/opensource/` 复用文档骨架改版本号即可。

## 发布过程中的三个实操坑（复用价值）

1. HF 仓库创建时自带 initial commit → 本地推送被判非快进：fetch 后
   `--allow-unrelated-histories` 合并（.gitattributes 冲突取 theirs）再推；
2. HF >10MiB 硬限：`tokenizer.json`(17MB) 也必须 LFS；且**旧提交里的大文件 blob
   会持续触发拒绝**——干净做法是重建单提交历史后 force push（LFS 对象不重传）；
3. 数据集 SSH 远端为 `git@hf.co:datasets/<user>/<name>`（带 datasets/ 前缀），
   首推即自动建库，20MB jsonl 走 LFS。

## 后续可选动作

- [ ] HF 模型页：widget 已生效；可再加 model-index 评测卡（页首指标展示）
- [ ] GitHub：Discussions / CITATION.cff / Release tag v1.0（含 HF 链接）
- [ ] v1.1（内部 v9）：合成通道扩产 → 重训 → 新温度 → 同流程发布
- [ ] 引导语：README 数据集/模型交叉引用已就位，发布推文/帖可从"三条数据定律"切入
