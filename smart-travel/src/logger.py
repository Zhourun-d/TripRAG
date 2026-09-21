"""
日志模块。

职责：
1. 提供统一的 logger 获取接口
2. 同时输出到控制台和文件
3. 按模块分文件，方便排查

为什么不用 print：
- print 没有级别（INFO/ERROR）
- print 不能同时输出到文件
- print 无法按模块过滤
"""

import logging
import sys
from pathlib import Path

from src.config import LOG_DIR


# 日志格式：时间 [级别] 模块名 - 消息
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s - %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def get_logger(name: str) -> logging.Logger:
    """
    获取一个 logger。

    Args:
        name: logger 名称，通常用 __name__ 或模块名，如 "crawl"

    Returns:
        配置好的 logger

    注意：
        - 同一个 name 多次调用返回同一个 logger，不会重复加 handler
        - 日志同时输出到控制台和 logs/{name}.log
    """
    logger = logging.getLogger(name)

    # 避免重复添加 handler（多次调用 get_logger 时）
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    logger.propagate = False  # 不向 root logger 传播，避免重复输出

    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

    # ---- 控制台 handler ----
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # ---- 文件 handler ----
    log_file = LOG_DIR / f"{name}.log"
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger