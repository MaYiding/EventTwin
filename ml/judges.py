# -*- coding: utf-8 -*-
"""判别器统一抽象层（L1/L2 框架核心）。

一切"两个事件是否同一事件"的判定需求（评测、训练数据标注、生产 resolve）
都走这里。框架阶段默认 provider=jev（行为与直接调 llm.decide 完全一致）；
本地模型训练完成后通过 config.json 的 judge 段或构造参数切换，无需改调用方。

判别器实现：
- JevPairJudge        —— Jev 对判（teacher，基准与终审）
- EmbeddingPairJudge  —— embedding 余弦（L1 微调前=通用基线；微调后=本地快速通道）
- RerankerPairJudge   —— cross-encoder 打分（L2 蒸馏学生，bge-reranker 系）
- CascadeJudge        —— 级联：本地快速通道 + 灰带升 Jev 终审

事件对表示（EventPair 的 a/b）统一用 frame dict：
  {"frame": 框架文本, "type": 事件类型, "time": "起~止", "evidence": [引文...]}
——与 resolve._llm_judge 的 state 结构同源，保证训练/评测/生产三态同分布。
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from intel import config as cfg, llm

RULES = ("同事件判定规则：原子事件=特定参与方+特定对象+特定时间+一次具体动作；同一动作被不同"
         "媒体报道（含转载、中英文）=同一事件；同一产品先后两次不同调价/发布=不同事件；并购"
         "宣布与交割=不同事件；旧闻被重新报道=同一事件；官方更正金额/日期=同一事件；主题相近"
         "但动作、对象或时间不同=不同事件；只按给出的信息判断，不按常识补充。")


def frame_text(f) -> str:
    """frame dict → 判别器输入文本（Embedding/Reranker 用的单串表示）。
    兼容已是文本的输入（benchmark 存储格式）。"""
    if isinstance(f, str):
        return f
    ev = "；".join((f.get("evidence") or [])[:1])
    parts = [f.get("frame", "")]
    if f.get("type"):
        parts.append(f"[{f['type']}]")
    if f.get("time"):
        parts.append(f"({f['time']})")
    if ev:
        parts.append(f"证据“{ev}”")
    return " ".join(p for p in parts if p)


class JevPairJudge:
    """Jev 对判（teacher）。批量接口按 60 对/请求组批（32k token 预算内）。"""

    name = "jev"
    needs = ["openrouter"]

    def __init__(self, batch: int = 36, stage: str = "pair.judge"):
        self.batch = batch
        self.stage = stage

    def judge_batch(self, pairs: list[tuple[dict, dict]]) -> list[float]:
        """返回每对的同事件概率（0-1，noul）。逐批调用，全部走 llm 缓存可复现。

        批级容错：单批网络失败重试后仍失败时，该批记 0.5 中性分并继续
        （万级对全量打分不因单批抖动毁整轮；中性分进图后边权≈0 不影响分组）。
        """
        out: list[float] = []
        for i in range(0, len(pairs), self.batch):
            chunk = pairs[i:i + self.batch]
            qs, meta = {}, {}
            for j, (a, b) in enumerate(chunk):
                qid = f"p{j}"
                qs[qid] = {
                    "type": "noul",
                    "instructions": (f"按 `判定规则` 判断 `cases.{qid}.事件甲` 与 "
                                     f"`cases.{qid}.事件乙` 是否为同一个现实发生的原子事件。"),
                    "criteria": {"true": "同一事件（相同参与方/对象+时间+同一动作，含不同媒体报道）",
                                 "false": "不同事件（动作/对象/时间不同，或仅主题相近）"}}
                meta[qid] = {"事件甲": a, "事件乙": b}
            try:
                answers = llm.decide({"判定规则": RULES, "cases": meta}, qs, stage=self.stage)
            except llm.LLMError:
                out.extend([0.5] * len(chunk))
                continue
            for j in range(len(chunk)):
                out.append(float((answers.get(f"p{j}") or {}).get("noul", 0.0)))
        return out


class EmbeddingPairJudge:
    """embedding 余弦判别。model 参数支持三种形态：
    - None                  → config 里的现役 embedding（当前 qwen3-embedding-8b，vLLM）
    - "path/to/adapter"     → L1 微调后的 LoRA adapter（挂回基座，需部署侧加载）
    - "model@dim"           → 显式指定模型 id 与维度（实验用）
    微调前它就是"通用表示基线"——benchmark 上与 Jev 的差距即 L1 要缩小的目标。
    """

    name = "embedding"

    def __init__(self, model: str | None = None, stage: str = "pair.embed"):
        self.model = model
        self.stage = stage

    def judge_batch(self, pairs: list[tuple[dict, dict]]) -> list[float]:
        import numpy as np
        texts = [frame_text(x) for pair in pairs for x in pair]
        mat = llm.embed(texts, stage=self.stage)  # 走向量缓存，重放零成本
        a = mat[0::2]
        b = mat[1::2]
        return [float(x) for x in (a * b).sum(axis=1)]


class RerankerPairJudge:
    """cross-encoder 学生判别器（L2 蒸馏产物）。框架阶段可加载未微调的
    bge-reranker-v2-m3 作为"零训练基线"；微调后加载 adapter 版本。

    模型文件本地缓存于 ml/models/（HF_HOME 指向该处），支持：
    - "base"                → BAAI/bge-reranker-v2-m3 原始权重（零训练基线）
    - "path/to/checkpoint"  → 蒸馏后的 checkpoint
    推理用 transformers；生产化后可换 vLLM/ONNX（见 ml/README）。
    """

    name = "reranker"

    def __init__(self, model: str = "base", device: str | None = None):
        self.model_spec = model
        self.device = device
        self._m = None
        self._tok = None

    def _load(self):
        if self._m is not None:
            return
        import torch  # noqa: F401 环境探测
        from transformers import AutoTokenizer, AutoConfig
        spec = "BAAI/bge-reranker-v2-m3" if self.model_spec == "base" else self.model_spec
        home = Path(__file__).parent / "models"
        home.mkdir(exist_ok=True)
        os.environ.setdefault("HF_HOME", str(home))
        self._tok = AutoTokenizer.from_pretrained(spec)
        cfg = AutoConfig.from_pretrained(spec)
        is_causal = cfg.model_type in ("qwen2", "qwen3", "llama", "mistral")
        if is_causal:
            from transformers import AutoModelForCausalLM
            self._m = AutoModelForCausalLM.from_pretrained(spec, torch_dtype="auto").cuda()
            self._is_causal = True
            self._pfx = ("<|im_start|>system\nJudge whether the Document meets the requirements "
                        "based on the Query. Only give me the judgment and do not output any other "
                        "words or explanations. The judgment should be yes or no.<|im_end|>\n"
                        "<|im_start|>user\nQuery: ")
            self._sfx = "\nDocument: "
            self._sfx2 = "\nJudgment: <|im_end|>\n<|im_start|>assistant\n"
            self._yes_id = self._tok("yes", add_special_tokens=False)["input_ids"][0]
            self._no_id = self._tok("no", add_special_tokens=False)["input_ids"][0]
            self._tok.padding_side = "left"
        else:
            from transformers import AutoModelForSequenceClassification
            self._m = AutoModelForSequenceClassification.from_pretrained(spec, torch_dtype="auto").cuda()
            self._is_causal = False
        self._m.eval()
        # 原生置信度：温度缩放后校准（训练侧随 checkpoint 保存 calibration.json）
        cal = Path(spec) / "calibration.json"
        self._temperature = json.loads(cal.read_text())["temperature"] if cal.exists() else 1.0

    def judge_batch(self, pairs: list[tuple[dict, dict]]) -> list[float]:
        import numpy as np
        import torch
        self._load()
        scores: list[float] = []
        if self._is_causal:
            B = 8
            with torch.no_grad():
                for i in range(0, len(pairs), B):
                    chunk = pairs[i:i+B]
                    texts = [f"{self._pfx}{frame_text(a)}{self._sfx}{frame_text(b)}{self._sfx2}"
                             for a, b in chunk]
                    inp = self._tok(texts, padding=True, truncation=True,
                                    max_length=512, return_tensors="pt",
                                    add_special_tokens=False)
                    inp = {k: v.to(self._m.device) for k, v in inp.items()}
                    la = self._m(**inp).logits[:, -1, :].float()
                    two = torch.stack([la[:, self._no_id], la[:, self._yes_id]], dim=-1)
                    probs = torch.softmax(two / self._temperature, dim=-1)[:, 1]
                    scores += [float(x) for x in probs]
            return scores
        with torch.no_grad():
            for a, b in pairs:
                ta, tb = frame_text(a), frame_text(b)
                inp = self._tok(ta, tb, return_tensors="pt", truncation=True,
                                max_length=512)
                logits = self._m(**inp).logits.squeeze()
                # sigmoid/logits 先过温度 T（校准），输出才是原生概率而非过自信分
                if logits.numel() == 1:
                    p = torch.sigmoid(logits / self._temperature).item()
                else:
                    p = float(torch.softmax(logits / self._temperature, dim=-1)[1])
                scores.append(p)
        _ = np  # noqa
        return scores


class CascadeJudge:
    """级联判别：本地快速通道处理置信区外，灰带升 teacher 终审。

    阈值语义：|p_local - 0.5| > band（如 band=0.35 即 p<0.15 或 p>0.85 直判），
    灰带全部送 teacher。评测时输出 escalation_rate 供风险-覆盖曲线调参。
    """

    def __init__(self, local: EmbeddingPairJudge | RerankerPairJudge,
                 teacher: JevPairJudge, band: float = 0.35):
        self.local = local
        self.teacher = teacher
        self.band = band
        self.last_escalation_rate = None

    def judge_batch(self, pairs: list[tuple[dict, dict]]) -> list[float]:
        import numpy as np
        local_scores = self.local.judge_batch(pairs)
        idx_gray = [i for i, s in enumerate(local_scores) if abs(s - 0.5) <= self.band]
        if idx_gray:
            gray_pairs = [pairs[i] for i in idx_gray]
            teacher_scores = self.teacher.judge_batch(gray_pairs)
            for i, s in zip(idx_gray, teacher_scores):
                local_scores[i] = s
        self.last_escalation_rate = len(idx_gray) / max(len(pairs), 1)
        _ = np  # noqa
        return local_scores


class LocalEmbeddingJudge:
    """本地加载 L1 训练产物评测用：transformers + (可选)LoRA adapter 直接编码算余弦。
    与生产 EmbeddingPairJudge（vLLM 服务）互为镜像；仅评测/离线场景使用。"""

    name = "embedding_local"

    def __init__(self, base: str, adapter: str | None = None, batch: int = 32):
        self.base = base
        self.adapter = adapter
        self.batch = batch
        self._m = None
        self._tok = None

    def _load(self):
        if self._m is not None:
            return
        import torch
        from transformers import AutoModel, AutoTokenizer
        from peft import PeftModel
        self._tok = AutoTokenizer.from_pretrained(self.base, padding_side="left")
        self._m = AutoModel.from_pretrained(self.base, torch_dtype=torch.bfloat16).cuda().eval()
        if self.adapter:
            self._m = PeftModel.from_pretrained(self._m, self.adapter)
            self._m = self._m.merge_and_unload().eval()

    def encode(self, texts):
        import numpy as np
        import torch
        self._load()
        out = []
        with torch.no_grad():
          for i in range(0, len(texts), self.batch):
            chunk = texts[i:i + self.batch]
            inp = self._tok(chunk, padding=True, truncation=True, max_length=512,
                            return_tensors="pt").to("cuda")
            h = self._m(**inp).last_hidden_state
            # last-token pooling（Qwen3-Embedding 红线：取右侧最后一个有效 token）
            left_pad = (inp["attention_mask"][:, -1] == 1)
            idx = inp["attention_mask"].sum(dim=1) - 1
            vecs = h[torch.arange(h.size(0)), idx]
            vecs = torch.nn.functional.normalize(vecs, dim=-1)
            out.append(vecs.float().cpu().numpy())
        return np.concatenate(out)

    def judge_batch(self, pairs):
        import numpy as np
        texts = [frame_text(x) for pair in pairs for x in pair]
        mat = self.encode(texts)
        a, b = mat[0::2], mat[1::2]
        return [float(x) for x in (a * b).sum(axis=1)]


class EnsembleJudge:
    """v8+v10a 双模型集成（logit 空间加权平均 + 集成温度）。

    依据 v10 集成扫描实测（2026-10-03）：两模型学到互补特征——
    v8 强于 gray 层（同实体+同类型+时间重叠的难例）、v10a 强于 pos 层
    （措辞多样的真同事件）——logit 平均同时拿到两者长处且 ECE 更好。
    最优配置 w=0.4, T_ensemble=0.5（三门禁全过区间 0.1-0.9 的稳健中点）。
    生产用法：config.judge.provider = "ensemble"。
    """

    name = "ensemble"

    def __init__(self, models: dict | None = None,
                 weights: dict | None = None,
                 temperature: float = 0.5,
                 tta: bool = True):   # 对称性 TTA：score(A,B)与score(B,A)平均（ACL 2022 证据）
        self.model_paths = models or {
            "v8": "ml/models/l2_v8",
            "v10a": "ml/models/l2_v10a",
        }
        self.weights = weights or {"v8": 0.4, "v10a": 0.6}
        self.temperature = temperature
        self.tta = tta
        self._models = {}

    def _load(self):
        if self._models:
            return
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        for name, path in self.model_paths.items():
            tok = AutoTokenizer.from_pretrained(path)
            model = AutoModelForSequenceClassification.from_pretrained(
                path, torch_dtype=torch.bfloat16).cuda().eval()
            cal = {}
            cal_path = Path(path) / "calibration.json"
            if cal_path.exists():
                cal = json.loads(cal_path.read_text())
            self._models[name] = (tok, model, cal.get("temperature", 1.0))

    def judge_batch(self, pairs):
        import numpy as np
        import torch
        self._load()
        n = len(pairs)
        mixed = np.zeros(n)
        for name, (tok, model, t_cal) in self._models.items():
            scores = []
            B = 32
            with torch.no_grad():
                for i in range(0, n, B):
                    chunk = pairs[i:i+B]
                    inp = tok([frame_text(a) if isinstance(a, dict) else a
                               for a, _ in chunk],
                              [frame_text(b) if isinstance(b, dict) else b
                               for _, b in chunk],
                              padding=True, truncation=True, max_length=512,
                              return_tensors="pt")
                    inp = {k: v.cuda() for k, v in inp.items()}
                    logits = model(**inp).logits.squeeze(-1).float()
                    if self.tta:
                        inp_r = tok([frame_text(b) if isinstance(b, dict) else b
                                     for _, b in chunk],
                                    [frame_text(a) if isinstance(a, dict) else a
                                     for a, _ in chunk],
                                    padding=True, truncation=True, max_length=512,
                                    return_tensors="pt")
                        inp_r = {k: v.cuda() for k, v in inp_r.items()}
                        logits_r = model(**inp_r).logits.squeeze(-1).float()
                        logits = (logits + logits_r) / 2
                    scores += [float(x) for x in torch.sigmoid(logits / t_cal)]
            w = self.weights[name]
            mixed += w * np.log(np.array(scores) + 1e-7)
        return [float(1 / (1 + np.exp(-l / self.temperature))) for l in mixed]


def make_judge(provider: str | None = None, **kw):
    """按 config.judge.provider 或显式参数构造判别器（生产/评测共用入口）。"""
    p = provider or cfg.load_config().get("judge", {}).get("provider", "jev")
    if p == "jev":
        return JevPairJudge(**kw)
    if p == "embedding":
        return EmbeddingPairJudge(**kw)
    if p == "reranker":
        return RerankerPairJudge(**kw)
    if p == "ensemble":
        return EnsembleJudge(**kw)
    if p == "cascade":
        conf = cfg.load_config().get("judge", {})
        local_p = conf.get("cascade_local", "embedding")
        local = (EmbeddingPairJudge(**kw) if local_p == "embedding"
                 else RerankerPairJudge(model=conf.get("cascade_model", "base")))
        return CascadeJudge(local, JevPairJudge(), band=conf.get("cascade_band", 0.35))
    raise ValueError(f"未知 judge provider: {p}")
