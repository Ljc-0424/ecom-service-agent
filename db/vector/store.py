"""轻量向量库：Markdown 分块 + 词袋向量 + 余弦相似度。

【这个文件在干什么】
  把 knowledge/ 目录的企业知识变成「能被搜到」的数据，分四步：
    1. 切块   _chunk_markdown —— 每篇 Markdown 按标题切成小段
    2. 切词   _tokenize       —— 每段拆成一个个词（中文按相邻两字组合）
    3. 向量化 _embed_tokens   —— 每段变成一串数字（词袋向量）
    4. 检索   search          —— 用户问题也变成向量，找「方向最接近」的几段

【谁在用】
  service/rag/rag_service.py → agent/tools/knowledge_tools.py：
  客服答不了业务事实时，LLM 调 search_knowledge_base 工具走到这里查知识。

第一版不引入外部 Embedding 服务，保证离线可运行、可替换。
后续可把 embed() 换成真实 embedding 模型，接口保持不变。

【语法速查】
  @dataclass                  —— 数据类：自动生成 __init__ 等样板代码
  field(default_factory=list) —— 默认值是「新建的空列表」；
                                 不能直接写 =[]，否则所有实例共用同一个列表
  @classmethod                —— 类方法：第一个参数是类本身（习惯写 cls）
  nonlocal                    —— 在闭包里声明「我要修改外层函数的变量」
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

from config.settings import settings


def _tokenize(text: str) -> list[str]:
    """中英文混合切词：英文按单词，中文按二元组。

    【为什么要切词】
      计算机没法直接比较「两段话像不像」。要先拆成最小单位（token），
      统计每个 token 出现的次数，变成向量后才能算相似度。

    【为什么中文用二元组（bigram）】
      中文没有空格可分词，「退货政策」按单字拆成「退 / 货 / 政 / 策」
      单字含义太弱；按相邻两字拆成「退货 / 货政 / 政策」，保住了大部分含义，
      而且不需要引入分词库。

    【返回】
      token 列表，如「退货政策 + refund」→ ["退货", "货政", "政策", "refund", ...]
    """
    text = text.lower()  # 统一转小写：Refund 和 refund 算同一个词
    tokens: list[str] = []
    # 英文/数字：正则 [a-z0-9]+ 匹配连续的小写字母或数字串
    tokens.extend(re.findall(r"[a-z0-9]+", text))
    # 中文连续段：[一-鿿] 覆盖常用汉字的 Unicode 范围
    for seg in re.findall(r"[一-鿿]+", text):
        if len(seg) == 1:
            tokens.append(seg)  # 只有一个字，没得拆
        else:
            # seg[i : i + 2] 切片：从位置 i 取 2 个字符
            # i 依次取 0,1,2,...（range(len(seg)-1)），得到所有相邻两字组合
            tokens.extend(seg[i : i + 2] for i in range(len(seg) - 1))
            tokens.extend(seg)  # 完整词本身也保留，提高短查询的召回
    return tokens


@dataclass
class ChunkRecord:
    """知识分块：向量库里存的一条记录（一篇文档被切成多块）。

    字段：
      chunk_id —— 块唯一编号，如 "退货政策-3"（文件名-第几块）
      title    —— 所属文档标题（取文件名）
      source   —— 来源文件名，检索结果里展示「出处」用
      section  —— 所属小节标题（这块内容上面最近的一个 # 标题）
      content  —— 这一块的正文
      tokens   —— 切词结果，检索时用；default_factory 保证每个实例
                  各有一个自己的空列表
    """

    chunk_id: str  # 块编号（如「退货政策-3」：文件名-第几块）
    title: str  # 所属文档标题
    source: str  # 来源文件名（检索结果展示「出处」用）
    section: str  # 所属小节标题（这块内容上面最近的 # 标题）
    content: str  # 这一块的正文
    tokens: list[str] = field(default_factory=list)  # 切词结果，检索时用


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
    """本地文件向量库（JSON 持久化）。

    【内部三份数据（检索的核心结构）】
      chunks  —— 所有知识分块的列表（下标就是块的编号）
      _vocab  —— 词表：token → 向量里的位置，如 {"退货": 0, "物流": 1}
      _matrix —— 向量矩阵：第 i 行就是 chunks[i] 的向量
    """

    _default: Optional["VectorStore"] = None

    def __init__(self, persist_dir: Optional[Path] = None) -> None:
        """初始化实例。

        【参数】
          persist_dir —— 缓存目录；不传则用全局配置 settings.vector_path
        """
        self.persist_dir = Path(persist_dir) if persist_dir else settings.vector_path
        self.persist_dir.mkdir(parents=True, exist_ok=True)  # 目录不存在则建
        self.chunks: list[ChunkRecord] = []
        self._vocab: dict[str, int] = {}      # 词表：token → 向量下标
        self._matrix: list[list[float]] = []  # 每行 = 一个分块的向量
        self._loaded = False                  # 懒加载标记：还没加载/构建过索引

    @classmethod
    def get_default(cls) -> "VectorStore":
        """获取全局向量库单例；第一次调用时加载或构建索引。

        【@classmethod + cls 语法】
          普通方法第一个参数是实例 self；类方法第一个参数是类本身 cls。
          它不依赖任何实例就能调用，专门用来管理「全局只该有一份」的东西。
        """
        if cls._default is None:
            cls._default = VectorStore()
            cls._default.load_or_build()
        return cls._default

    @classmethod
    def set_default(cls, store: "VectorStore") -> None:
        """替换全局向量库（测试用：换成指向临时目录的实例）。"""
        cls._default = store

    # ---- 构建 ----

    def build_from_markdown_dir(self, md_dir: Path) -> int:
        """从 Markdown 目录全量构建知识分块与向量矩阵。

        【流程】
          递归找出所有 .md → 每个文件切块 → 建向量矩阵 → 写 JSON 缓存

        【返回】
          切出的分块总数（目录不存在返回 0）
        """
        chunks: list[ChunkRecord] = []
        md_dir = Path(md_dir)
        if not md_dir.exists():
            return 0
        for path in sorted(md_dir.glob("**/*.md")):
            # glob("**/*.md")：** 表示递归进入所有子目录
            # sorted：保证处理顺序稳定，缓存内容不会因文件顺序不同而变化
            chunks.extend(self._chunk_markdown(path))
        self.chunks = chunks
        self._build_matrix()
        self._persist()
        self._loaded = True
        return len(chunks)

    def _chunk_markdown(self, path: Path) -> list[ChunkRecord]:
        """按标题把一个 Markdown 文件切成检索分块。

        【为什么按标题切】
          一整篇文档做检索单元粒度太粗——命中了也返回一大篇，
          模型抓不住重点。按小节切，每块语义集中，检索结果更准。

        【切分规则】
          从头攒内容到 buffer；每遇到一行 # 开头（标题），
          就把攒的内容 flush 成一块，该标题记为下一块的 section。
        """
        text = path.read_text(encoding="utf-8")
        lines = text.splitlines()  # 按行拆成列表
        title = path.stem          # 文件名去掉 .md 后缀，当文档标题
        section = ""               # 当前所在小节
        buffer: list[str] = []     # 当前块攒的行
        chunks: list[ChunkRecord] = []
        idx = 0                    # 块编号，用于拼 chunk_id

        def flush() -> None:
            """把 buffer 里攒的内容打包成一块。

            【nonlocal 语法】
              flush 里要修改外层函数的 idx。Python 默认「函数内赋值 =
              新建局部变量」，nonlocal 声明「idx 用的是外层那个」，否则报错。
            """
            content = "\n".join(buffer).strip()  # 攒的行拼回一段文字
            if not content:
                return  # 空块（比如两个标题连着）跳过
            idx += 1
            # 把正文、文档标题、小节名一起切词：让「标题词」也参与检索
            tokens = _tokenize(content + " " + title + " " + section)
            chunks.append(
                ChunkRecord(
                    chunk_id=f"{path.stem}-{idx}",
                    title=title,
                    source=path.name,
                    section=section or title,
                    content=content,
                    tokens=tokens,
                )
            )

        for line in lines:
            if line.startswith("#"):
                flush()  # 遇到新标题：先结算上一块
                buffer = []
                # lstrip("#")：去掉行首所有 # 号，再去掉首尾空白得到小节名
                section = line.lstrip("#").strip()
                if not title or title == path.stem:
                    title = section or path.stem  # 文档第一个标题兼作文档名
            else:
                buffer.append(line)
        flush()  # 别忘了最后一块（后面没有标题了）
        return chunks

    def _build_matrix(self) -> None:
        """根据分块 token 构建词表与向量矩阵。

        【词表怎么建】
          扫描所有分块的 token，第一次遇到某个 token 时给它编号——
          vocab[t] = len(vocab)：赋值前 len(vocab) 正好等于已收录词数，
          相当于「用当前大小自增编号」。之后向量里第 vocab[t] 位就代表 token t。
        """
        vocab: dict[str, int] = {}
        for ch in self.chunks:
            for t in ch.tokens:
                if t not in vocab:
                    vocab[t] = len(vocab)
        self._vocab = vocab
        # 列表推导式：对每个分块算一遍词袋向量，得到「一行一块」的矩阵
        self._matrix = [self._embed_tokens(ch.tokens) for ch in self.chunks]

    def _embed_tokens(self, tokens: Iterable[str]) -> list[float]:
        """将 token 列表编码为 L2 归一化词袋向量。

        【词袋（bag of words）】
          不关心词的先后顺序，只统计每个词出现几次 → 一串数字。
          例：词表 {"退货":0, "物流":1, ...}，文本含 2 次「退货」
          → 向量 [2, 0, ...]。

        【L2 归一化】
          把向量除以自身的长度（各分量平方和开根号），变成「单位长度」。
          好处：归一化后两向量的点积 = 余弦相似度，只比较方向、不受长短影响，
          长文本不会因为「词多」而天然得分高。
        """
        vec = [0.0] * max(len(self._vocab), 1)  # 全 0 向量，长度 = 词表大小
        for t in tokens:
            if t in self._vocab:
                vec[self._vocab[t]] += 1.0  # 该词出现一次，对应位置 +1
        # L2 归一化
        norm = math.sqrt(sum(x * x for x in vec))  # 各分量平方和开根号
        if norm > 0:
            vec = [x / norm for x in vec]  # 每个分量除以长度
        return vec

    def _persist(self) -> None:
        """将分块数据持久化到本地 JSON（检索缓存）。"""
        data = {
            "chunks": [
                {
                    "chunk_id": c.chunk_id,
                    "title": c.title,
                    "source": c.source,
                    "section": c.section,
                    "content": c.content,
                    "tokens": c.tokens,
                }
                for c in self.chunks
            ]
        }
        path = self.persist_dir / "chunks.json"
        # ensure_ascii=False：中文直接写成汉字，文件可读；否则会是 \uXXXX
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def load_or_build(self) -> None:
        """加载已有的向量缓存；缓存不存在则从知识库重建。

        【为什么缓存】
          全量重建要重读所有 Markdown 并重新切词，慢；有缓存直接读 JSON。
          注意：改了知识库内容后要删掉 chunks.json（或调用
          build_from_markdown_dir）才会生效。
        """
        path = self.persist_dir / "chunks.json"
        if path.exists():
            raw = json.loads(path.read_text(encoding="utf-8"))
            # **c：把 {"chunk_id": ..., ...} 摊开成关键字参数构造 ChunkRecord
            self.chunks = [ChunkRecord(**c) for c in raw.get("chunks", [])]
            self._build_matrix()
            self._loaded = True
            return
        self.build_from_markdown_dir(settings.knowledge_path)

    # ---- 检索 ----

    def search(self, query: str, top_k: int = 3) -> list[ChunkHit]:
        """向量相似度检索，返回最相关的 Top-K 知识分块。

        【流程】
          1. 没加载过索引 → 先 load_or_build（懒加载：第一次搜索才建）
          2. 用户问题切词 → 也变成向量
          3. 和每个分块的向量算点积（归一化后点积 = 余弦相似度）
          4. 按得分从高到低取前 top_k 个

        【参数】
          top_k —— 返回几条。给 LLM 的参考片段数量，太多会淹没重点。
        """
        if not self._loaded:
            self.load_or_build()
        if not self.chunks:
            return []
        q_tokens = _tokenize(query)
        q_vec = self._embed_tokens(q_tokens)
        scored: list[tuple[float, int]] = []  # (得分, 分块下标)
        for i, vec in enumerate(self._matrix):
            # zip 把两个向量按位置配对；对应位相乘再求和 = 点积 = 相似度
            score = sum(a * b for a, b in zip(q_vec, vec))
            if score > 0:
                scored.append((score, i))
        scored.sort(key=lambda x: x[0], reverse=True)
        # lambda x: x[0]：按每项的第 0 个元素（得分）排；reverse=True 从大到小
        hits: list[ChunkHit] = []
        for score, i in scored[:top_k]:  # 切片取前 top_k 个
            c = self.chunks[i]
            hits.append(
                ChunkHit(
                    chunk_id=c.chunk_id,
                    title=c.title,
                    source=c.source,
                    section=c.section,
                    content=c.content,
                    score=score,
                )
            )
        return hits
