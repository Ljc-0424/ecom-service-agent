"""知识向量存储：支持 Embedding 检索与本地词袋回退。

索引缓存记录 backend/model；配置变更时自动重建，避免不同向量空间混用。
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

import httpx

from config.settings import settings


def _tokenize(text: str) -> list[str]:
    """中英文混合切词：英文按单词，中文按二元组。

    Embedding 后端不直接用切词结果（模型吃原文），但 title/section 的
    切词会作为元数据参与词袋回退与检索展示。

    【为什么中文用二元组（bigram）】
      中文没有空格可分词，「退货政策」按单字拆语义太弱；
      按相邻两字拆成「退货 / 货政 / 政策」，保住大部分含义且无需分词库。
    """
    text = text.lower()
    tokens: list[str] = []
    tokens.extend(re.findall(r"[a-z0-9]+", text))
    for seg in re.findall(r"[一-鿿]+", text):
        if len(seg) == 1:
            tokens.append(seg)
        else:
            tokens.extend(seg[i : i + 2] for i in range(len(seg) - 1))
            tokens.extend(seg)  # 完整词也保留，提高短查询召回
    return tokens


def _normalize(vec: list[float]) -> list[float]:
    """L2 归一化：除以自身长度，变成「单位向量」。

    归一化后两个向量的点积 = 余弦相似度（只比方向、比长短更公平）。
    """
    norm = math.sqrt(sum(x * x for x in vec))
    return [x / norm for x in vec] if norm > 0 else vec


@dataclass
class ChunkRecord:
    """知识分块：向量库里存的一条记录。

    字段：
      chunk_id —— 块唯一编号，如 "退货政策-3"（文件名-第几块）
      title    —— 所属文档标题
      source   —— 来源文件名（检索结果展示「出处」用）
      section  —— 所属小节标题（这块内容上面最近的 # 标题）
      content  —— 这一块的正文
      tokens   —— 切词结果（词袋回退后端用）
      vector   —— 本块的向量（后端产出；检索时与查询向量算点积）
    """

    chunk_id: str
    title: str
    source: str
    section: str
    content: str
    tokens: list[str] = field(default_factory=list)
    vector: list[float] = field(default_factory=list)


@dataclass
class ChunkHit:
    """一条检索命中：分块内容 + 相似度得分（search 的返回元素）。"""

    chunk_id: str  # 块编号
    title: str  # 所属文档标题
    source: str  # 来源文件名（检索结果展示「出处」用）
    section: str  # 所属小节标题
    content: str  # 知识片段正文
    score: float  # 相似度得分（越大越相关）


class VectorStore:
    """本地向量库：Embedding 后端（默认）+ 词袋回退，JSON 缓存。

    【内部数据】
      chunks  —— 所有知识分块（vector 字段即该块的向量）
      _vocab  —— 词袋后端的词表：token → 向量下标（embedding 后端不使用）
    """

    _default: Optional["VectorStore"] = None

    def __init__(self, persist_dir: Optional[Path] = None) -> None:
        """初始化实例。

        【参数】
          persist_dir —— 缓存目录；不传则用全局配置 settings.vector_path
        """
        self.persist_dir = Path(persist_dir) if persist_dir else settings.vector_path
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.chunks: list[ChunkRecord] = []
        self._vocab: dict[str, int] = {}  # 词袋后端专用：token → 向量下标
        self.backend = self._detect_backend()
        self._loaded = False

    @staticmethod
    def _detect_backend() -> str:
        """按配置决定后端：配了 Embedding Key 就用 embedding，否则词袋回退。"""
        return "embedding" if (settings.embedding_api_key and settings.embedding_model) else "bow"

    @classmethod
    def get_default(cls) -> "VectorStore":
        """获取全局向量库单例；第一次调用时加载或构建索引。

        【@classmethod + cls】类方法不依赖实例即可调用，
        专门用来管理「全局只该有一份」的东西（同 SessionStore.get_default）。
        """
        if cls._default is None:
            cls._default = VectorStore()
            cls._default.load_or_build()
        return cls._default

    @classmethod
    def set_default(cls, store: "VectorStore") -> None:
        """替换全局向量库（测试用：换成指向临时目录的实例）。"""
        cls._default = store

    # ---- 向量化 ----

    def _embed_api(self, texts: list[str]) -> list[list[float]]:
        """调用 OpenAI 兼容 /embeddings 接口批量向量化。

        【trust_env=False 的原因】
          httpx 默认读取系统代理；本机代理会把到国内 API 的 TLS 连接掐断
          （实测 SSL EOF），而硅基流动是直连可达的国内服务，所以明确直连。
        【重试】网络抖动重试 2 次（20s 后重试一次，再失败抛出让上层兜底）。
        """
        url = settings.embedding_base_url.rstrip("/") + "/embeddings"
        headers = {"Authorization": f"Bearer {settings.embedding_api_key}"}
        vectors: list[list[float]] = []
        batch_size = 16
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            for attempt in (1, 2):
                try:
                    resp = httpx.post(
                        url,
                        headers=headers,
                        json={"model": settings.embedding_model, "input": batch},
                        timeout=60,
                        trust_env=False,
                    )
                    resp.raise_for_status()
                    data = sorted(resp.json()["data"], key=lambda d: d["index"])
                    vectors.extend([_normalize(d["embedding"]) for d in data])
                    break
                except Exception:
                    if attempt == 2:
                        raise
        return vectors

    def _bow_vectors(self, chunks: list[ChunkRecord]) -> list[list[float]]:
        """词袋回退：按全部块的 token 建词表，每块编码为归一化词频向量。

        【词表怎么建】第一次遇到的 token 编号为「当前词表大小」：
        vocab[t] = len(vocab)，相当于自增编号；向量第 vocab[t] 位代表 token t。
        """
        vocab: dict[str, int] = {}
        for ch in chunks:
            for t in ch.tokens:
                if t not in vocab:
                    vocab[t] = len(vocab)
        self._vocab = vocab
        vectors = []
        for ch in chunks:
            vec = [0.0] * max(len(vocab), 1)
            for t in ch.tokens:
                vec[vocab[t]] += 1.0
            vectors.append(_normalize(vec))
        return vectors

    def _embed_query_bow(self, query: str) -> list[float]:
        """词袋后端的查询向量化（复用建库时的词表）。"""
        vec = [0.0] * max(len(self._vocab), 1)
        for t in _tokenize(query):
            if t in self._vocab:
                vec[self._vocab[t]] += 1.0
        return _normalize(vec)

    def _embed_query(self, query: str) -> list[float]:
        """查询向量化：与建库后端保持一致（混用两个向量空间没有意义）。"""
        if self.backend == "embedding":
            return self._embed_api([query])[0]
        return self._embed_query_bow(query)

    # ---- 构建 ----

    def build_from_markdown_dir(self, md_dir: Path) -> int:
        """从 Markdown 目录全量构建知识分块与向量（写 JSON 缓存）。

        【流程】
          递归找出所有 .md → 按标题切块 → 向量化（按后端）→ 写缓存

        【返回】
          切出的分块总数（目录不存在返回 0）
        """
        chunks: list[ChunkRecord] = []
        md_dir = Path(md_dir)
        if not md_dir.exists():
            return 0
        for path in sorted(md_dir.glob("**/*.md")):
            # glob("**/*.md")：** 递归进所有子目录；sorted 保证顺序稳定
            chunks.extend(self._chunk_markdown(path))
        self.chunks = chunks
        self._vectorize()
        self._persist()
        self._loaded = True
        return len(chunks)

    def _vectorize(self) -> None:
        """按当前后端给全部分块产出向量。"""
        if self.backend == "embedding":
            texts = [f"{c.title} {c.section} {c.content}" for c in self.chunks]
            vectors = self._embed_api(texts)
        else:
            vectors = self._bow_vectors(self.chunks)
        for chunk, vec in zip(self.chunks, vectors):
            chunk.vector = vec

    def _chunk_markdown(self, path: Path) -> list[ChunkRecord]:
        """按标题把一个 Markdown 文件切成检索分块。

        【为什么按标题切】整篇文档做检索单元粒度太粗——命中了也返回
        一大篇，模型抓不住重点。按小节切，每块语义集中。
        """
        text = path.read_text(encoding="utf-8")
        lines = text.splitlines()
        title = path.stem  # 文件名去扩展名当文档标题
        section = ""
        buffer: list[str] = []
        chunks: list[ChunkRecord] = []
        idx = 0

        def flush() -> None:
            """把 buffer 里攒的内容打包成一块（nonlocal 声明改外层的 idx）。"""
            nonlocal idx
            content = "\n".join(buffer).strip()
            if not content:
                return
            idx += 1
            chunks.append(
                ChunkRecord(
                    chunk_id=f"{path.stem}-{idx}",
                    title=title,
                    source=path.name,
                    section=section or title,
                    content=content,
                    tokens=_tokenize(content + " " + title + " " + section),
                )
            )

        for line in lines:
            if line.startswith("#"):
                flush()  # 遇到新标题：先结算上一块
                buffer = []
                section = line.lstrip("#").strip()
                if not title or title == path.stem:
                    title = section or path.stem
            else:
                buffer.append(line)
        flush()  # 最后一块后面没有标题了，别忘结算
        return chunks

    def _persist(self) -> None:
        """将分块 + 向量 + 后端元数据持久化到本地 JSON。"""
        data = {
            "backend": self.backend,
            "model": settings.embedding_model if self.backend == "embedding" else "bow",
            "chunks": [
                {
                    "chunk_id": c.chunk_id,
                    "title": c.title,
                    "source": c.source,
                    "section": c.section,
                    "content": c.content,
                    "tokens": c.tokens,
                    "vector": c.vector,
                }
                for c in self.chunks
            ],
        }
        path = self.persist_dir / "chunks.json"
        path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    def load_or_build(self) -> None:
        """加载向量缓存；缓存不存在或**后端/模型与当前配置不一致**时自动重建。

        【为什么自动重建】
          缓存记录了构建时的 backend/model。比如之前用词袋建过索引、
          现在配了 Embedding——旧向量和新查询不在一个空间里，混用必错。
          与其让人记得删缓存，不如让代码自己发现并重建（自愈）。
        """
        path = self.persist_dir / "chunks.json"
        want_backend, want_model = self.backend, (
            settings.embedding_model if self.backend == "embedding" else "bow"
        )
        if path.exists():
            raw = json.loads(path.read_text(encoding="utf-8"))
            if raw.get("backend") == want_backend and raw.get("model") == want_model:
                self.chunks = [
                    ChunkRecord(
                        chunk_id=c["chunk_id"],
                        title=c["title"],
                        source=c["source"],
                        section=c["section"],
                        content=c["content"],
                        tokens=c.get("tokens") or [],
                        vector=c.get("vector") or [],
                    )
                    for c in raw.get("chunks", [])
                ]
                if self.backend == "bow":
                    self._bow_vectors(self.chunks)  # 重建词表（查询向量化要用）
                self._loaded = True
                return
        self.build_from_markdown_dir(settings.knowledge_path)

    # ---- 检索 ----

    def search(self, query: str, top_k: int = 3) -> list[ChunkHit]:
        """向量相似度检索，返回最相关的 Top-K 知识分块。

        【流程】
          1. 没加载过索引 → 先 load_or_build（懒加载）
          2. 查询按同一后端向量化并归一化
          3. 与每块向量点积（归一化后点积 = 余弦相似度）
          4. 按得分从高到低取前 top_k 个
        """
        if not self._loaded:
            self.load_or_build()
        if not self.chunks:
            return []
        q_vec = _normalize(self._embed_query(query))
        scored: list[tuple[float, ChunkRecord]] = []
        for chunk in self.chunks:
            # 向量均已归一化，点积即余弦相似度。
            if not chunk.vector:
                continue
            score = sum(a * b for a, b in zip(q_vec, chunk.vector))
            scored.append((score, chunk))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            ChunkHit(
                chunk_id=c.chunk_id,
                title=c.title,
                source=c.source,
                section=c.section,
                content=c.content,
                score=score,
            )
            for score, c in scored[:top_k]
        ]
