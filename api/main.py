"""FastAPI 入口。

架构：
  Frontend → FastAPI
               ├─ /api/chat + /api/orders  → Agent / 业务查询
               └─ /api/mock/*              → Service → Business DB

Mock API 不是 Agent Tool，但两者读写同一套模拟业务数据。
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from api import chat, mock, orders
from config.settings import PROJECT_ROOT, settings
from db.business.database import get_business_db
from db.business.seed import seed_demo_data
from db.vector.store import VectorStore

app = FastAPI(
    title="电商客服 Agent 评测业务交互模拟系统",
    version="0.1.0",
    description="Agent Core + 模拟电商业务系统 + Mock Console + 评测预留",
)

# CORS 中间件：解决浏览器「跨域」限制。
# 浏览器安全策略默认禁止「页面 A 端口」的 JS 请求「服务 B 端口」
# （前端开发服务器和 8000 端口不同源）。这个中间件在响应里加「允许跨域」头放行；
# V1 用 ["*"] 全放开，正式部署应改成具体的前端域名。
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# include_router：把各文件里定义的接口挂到主应用上
# prefix 是统一前缀，如 mock.py 里的 /orders 实际路径是 /api/mock/orders
app.include_router(chat.router, prefix="/api", tags=["agent"])
app.include_router(orders.router, prefix="/api", tags=["orders"])
app.include_router(mock.router, prefix="/api/mock", tags=["mock"])


@app.on_event("startup")
def on_startup() -> None:
    """启动钩子：初始化目录、种子数据与知识向量库。

    【@app.on_event("startup") 语法】
      FastAPI 的启动钩子：服务开始接收请求「之前」自动执行一次，
      适合做初始化（建目录、灌数据、建索引这类一次性工作）。
    """
    settings.ensure_dirs()
    db = get_business_db()
    # 首次启动灌入演示数据（库里已有商品说明不是第一次，跳过）
    if db.list_products() == []:
        seed_demo_data(db)
    # 构建/加载知识向量：有缓存读缓存，没缓存从 knowledge/ 全量重建
    store = VectorStore.get_default()
    store.load_or_build()
    if not store.chunks:
        store.build_from_markdown_dir(settings.knowledge_path)


@app.get("/api/health")
def health() -> dict:
    """健康检查接口：返回 ok 表示服务活着（部署/监控常用）。"""
    return {"status": "ok", "service": "ecom-service-agent"}


# 静态前端：把 frontend/ 目录挂到 /static 路径，浏览器能直接下载里面的文件
_frontend_dir = PROJECT_ROOT / "frontend"
if _frontend_dir.exists():
    app.mount("/static", StaticFiles(directory=str(_frontend_dir)), name="static")

    @app.get("/")
    def index() -> FileResponse:
        """返回前端首页 HTML：访问 http://localhost:8000/ 看到的就是它。"""
        return FileResponse(str(_frontend_dir / "index.html"))
