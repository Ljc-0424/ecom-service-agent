"""config 包：对外只暴露 settings 这一个对象。

其它模块统一写 from config.settings import settings；
如果以后要加配置相关的东西，也从这个包出去，保持入口唯一。
"""

from config.settings import settings

__all__ = ["settings"]
# __all__：约定 from config import * 时只导出列表里列出的名字
