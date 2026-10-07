"""全局配置：从环境变量 / .env 读取，禁止把密钥写进代码。

【这个文件在干什么】
  定义 Settings 类：读取 LLM_API_KEY、数据库路径等配置。
  其它模块写 from config.settings import settings 来使用。

【谁会用到这些配置】
  agent/llm/client.py         —— 用 llm_api_key / llm_base_url / llm_model 连真实大模型
  db/business/database.py     —— 用 sqlite_path 找到 .db 文件（建库、读写都靠它）
  db/vector/store.py          —— 用 vector_path 存检索缓存，用 knowledge_path 读知识库
  run_server.py / api/main.py —— 用 api_host / api_port 启动 Web 服务

【为什么字段名和环境变量名长得不一样】
  类里写 llm_api_key（Python 变量习惯小写），
  .env 里写 LLM_API_KEY=xxx（环境变量习惯大写），
  两者靠 alias="LLM_API_KEY" 对上：pydantic 读 .env / 环境变量时按大写找，
  代码里用 settings.llm_api_key 访问。

【语法速查】
  BaseSettings + SettingsConfigDict —— pydantic-settings：自动读环境变量和 .env
  Field(default=..., alias="LLM_API_KEY") —— 字段默认值 + 对应的环境变量名
  @property —— 方法当属性用：settings.sqlite_path 不用写括号
  Path       —— 路径对象（比字符串路径好用，支持 / 拼接、判断是否存在等）
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# __file__ 是「当前这个文件」的磁盘路径；.parent 取上一级目录
# resolve() 把它变成规范的绝对路径；连取两次 .parent：config/settings.py → config → 项目根
# 为什么需要它：下面那些 ./data 相对路径都要「相对项目根」，见 sqlite_path 的说明
PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """应用配置。"""

    model_config = SettingsConfigDict(
        # 告诉 pydantic-settings：去项目根目录找 .env 文件自动读取
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        # .env 里写了 Settings 没定义的变量时不要报错，直接忽略
        extra="ignore",
    )

    # ---- LLM（被 agent/llm/client.py 的 create_llm() 使用）----
    # 密钥：.env 里配 LLM_API_KEY=sk-xxx；不配置则调用真实模型时会直接报错
    llm_api_key: str = Field(default="", alias="LLM_API_KEY")
    # 大模型接口地址：OpenAI 兼容格式，换国内中转/私有部署只改这一处
    llm_base_url: str = Field(default="https://api.openai.com/v1", alias="LLM_BASE_URL")
    # 模型名：告诉服务端用哪个模型
    llm_model: str = Field(default="gpt-4o-mini", alias="LLM_MODEL")

    # ---- 存储 ----
    # 业务数据库「连接串」。sqlite:///文件路径 是数据库工具的通用写法，
    # 本项目自己解析它（见 sqlite_path）；默认存在 data/ecom.db
    database_url: str = Field(default="sqlite:///./data/ecom.db", alias="DATABASE_URL")
    # 向量检索缓存的目录：db/vector/store.py 把分块索引存在里面的 chunks.json
    vector_db_path: str = Field(default="./data/vector_store", alias="VECTOR_DB_PATH")
    # 企业知识库目录：里面放 Markdown 文档，启动时从这里建检索索引（RAG 的知识来源）
    knowledge_dir: str = Field(default="./knowledge", alias="KNOWLEDGE_DIR")

    # ---- Embedding（企业知识检索向量化）----
    # 配置后知识检索使用 Embedding 模型（OpenAI 兼容 /embeddings 接口）；
    # 不配置则回落词频向量（离线可用）。推荐硅基流动 BAAI/bge-m3（免费、中文优化）
    embedding_api_key: str = Field(default="", alias="EMBEDDING_API_KEY")
    embedding_base_url: str = Field(default="https://api.siliconflow.cn/v1", alias="EMBEDDING_BASE_URL")
    embedding_model: str = Field(default="BAAI/bge-m3", alias="EMBEDDING_MODEL")

    # ---- 服务（被 run_server.py 启动时使用）----
    # 监听地址：0.0.0.0 表示本机所有网卡都能访问（本机 + 局域网）
    api_host: str = Field(default="0.0.0.0", alias="API_HOST")
    # 端口：浏览器访问 http://localhost:8000
    api_port: int = Field(default=8000, alias="API_PORT")

    @property
    def sqlite_path(self) -> Path:
        """把 DATABASE_URL（连接串）翻译成磁盘上的 .db 文件路径。

        【为什么需要这个函数】
          原因一：格式翻译。DATABASE_URL 用的是 sqlite:///./data/ecom.db 这种
          「连接串」格式——它是 SQLAlchemy 等数据库库的行业惯例，好处是以后
          换 MySQL/PostgreSQL 时配置字段不用改。但 V1 直接用 Python 标准库
          sqlite3，sqlite3.connect() 只认文件路径、不认连接串，
          所以中间要做一次「把前缀剥掉」的翻译。

          原因二：把相对路径固定到项目根。./data/ecom.db 是相对路径，
          「相对谁」取决于你在哪个目录运行启动命令——从项目根运行没问题，
          换个目录运行就会把数据库建到别的位置。拼上项目根变成绝对路径后，
          无论从哪里启动，数据库永远是同一个文件。

        【结果被谁用】
          db/business/database.py 的 BusinessDatabase：
          用这个路径创建 .db 文件、建表，之后所有业务数据的读写都连它。

        【流程】
          sqlite:///./data/ecom.db
            → 去掉 sqlite:/// 前缀      得到 ./data/ecom.db
            → 发现是相对路径
            → 拼上项目根               得到 <项目根>/data/ecom.db（绝对路径）
        """
        url = self.database_url
        # 三元表达式：以 sqlite:/// 开头 → 用切片剥掉前缀；
        # url[len("sqlite:///"):] 表示「从前缀长度那个位置截取到末尾」
        # 不以它开头（配置的本来就是路径）→ 原样使用
        raw = url[len("sqlite:///") :] if url.startswith("sqlite:///") else url
        path = Path(raw)
        # is_absolute()：是否已经是盘符/根开头的绝对路径，是就不需要再拼
        # Path / Path：Path 对象用 / 拼接路径，自动处理斜杠方向
        return path if path.is_absolute() else PROJECT_ROOT / path

    @property
    def vector_path(self) -> Path:
        """向量检索缓存目录的绝对路径。

        【为什么需要】
          和 sqlite_path 同理：配置里的 ./data/vector_store 是相对路径，
          拼上项目根，保证从任何目录启动都定位到同一个缓存目录。

        【结果被谁用】
          db/vector/store.py 的 VectorStore：把知识分块索引存在该目录的
          chunks.json 里；下次启动直接加载，不用重新解析全部 Markdown。
        """
        path = Path(self.vector_db_path)
        return path if path.is_absolute() else PROJECT_ROOT / path

    @property
    def knowledge_path(self) -> Path:
        """知识库目录的绝对路径。

        【结果被谁用】
          db/vector/store.py 的 load_or_build / api/main.py 的启动钩子：
          读取该目录下所有 .md 文件，切块建索引（客服能答的「政策、规则」
          都来自这里）。想让客服多懂点，就往这个目录加 Markdown。
        """
        path = Path(self.knowledge_dir)
        return path if path.is_absolute() else PROJECT_ROOT / path

    def ensure_dirs(self) -> None:
        """启动前确保 data 目录存在，避免写文件时报错。

        【为什么需要】
          Python 写文件不会自动创建多级目录，目录不存在会直接抛
          FileNotFoundError。数据库 .db 文件、chunks.json 都写在 data/ 下，
          所以趁服务还没开始干活，先把目录建好。

        【谁在调用】
          run_server.py 的 main() 和 api/main.py 的 on_startup()（启动钩子）。

        【语法】
          mkdir(parents=True, exist_ok=True)
          parents=True  —— 父目录不存在就连父目录一起建
          exist_ok=True —— 目录已存在不报错（服务重复启动也没事）
        """
        # sqlite_path 是文件路径，.parent 取它所在的目录（即 data/）
        self.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        self.vector_path.mkdir(parents=True, exist_ok=True)


# 模块被 import 时执行一次，得到全局唯一的配置对象（单例）：
# 全项目任何地方拿到的 settings 都是同一份
settings = Settings()
