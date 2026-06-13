# Crypto Freqtrade

这个目录存放历史加密货币/Freqtrade 相关内容，用来和当前 A 股研究主线隔离。

## 当前保存内容

- `strategies/*.py`：历史 Freqtrade 策略源码。

## 不保存内容

以下内容属于本地运行产物，已由根目录 `.gitignore` 忽略：

- Freqtrade 下载的 K 线数据。
- 回测、hyperopt、日志、图表输出。
- 本地配置 JSON 和交易所相关配置。

需要继续跑这些策略时，在目标机器上单独安装和配置 Freqtrade，再把 `strategies/` 下的策略源码放入对应运行环境。
