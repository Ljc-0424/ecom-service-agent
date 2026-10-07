"""一键启动脚本：python run_server.py

【运行它会发生什么】
  1. ensure_dirs() —— 建好 data/ 目录（数据库和向量缓存要写在这里）
  2. uvicorn.run() —— 启动 Web 服务器，加载 api/main.py 里定义的 app
  3. 浏览器访问 http://localhost:8000 即可打开客服界面

【语法速查】
  if __name__ == "__main__": —— 「直接运行本文件」时才为真；
                                被别的文件 import 时不执行，避免 import 副作用。
  uvicorn —— 把 FastAPI 应用跑成 HTTP 服务的服务器程序
             （FastAPI 只定义接口，真正监听端口的是 uvicorn）
"""

from __future__ import annotations

import uvicorn

from config.settings import settings


def main() -> None:
    """启动 FastAPI 服务。"""
    # 先建目录再启动：不然首次运行时数据库/向量缓存写不进去
    settings.ensure_dirs()
    uvicorn.run(
        # 「模块路径:应用变量名」—— 告诉 uvicorn 去哪找 FastAPI 实例
        "api.main:app",
        host=settings.api_host,  # 来自 .env 的 API_HOST，默认 0.0.0.0
        port=settings.api_port,  # 来自 .env 的 API_PORT，默认 8000
        reload=False,            # True = 改代码自动重启（开发模式）；V1 关闭
    )


if __name__ == "__main__":
    main()
