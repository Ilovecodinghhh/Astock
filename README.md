# Astock

这是一个基于 baostock A 股数据的量化研究项目，包含本地行情下载、因子生成、组合回测、策略扫描和稳健性验证工具。

## 仓库里会保存什么

- `scripts/`：数据下载、因子生成、回测、扫描、稳健性验证脚本。
- `tests/`：核心研究工具的回归测试。
- `user_data/astock_pools/*.txt`：小体积股票池定义。
- `user_data/strategies/*.py`：保留作参考的 Freqtrade 策略源码。
- `docs/`：项目说明和协作文档。

大型行情数据、生成特征、回测输出、运行缓存和本地配置文件不会进 Git。换电脑协作前先看 [docs/data-management.md](docs/data-management.md)。

## 环境安装

```bash
python -m venv .venv
source .venv/Scripts/activate
python -m pip install -r requirements.txt
```

Linux/macOS 使用 `source .venv/bin/activate` 激活虚拟环境。

## 验证

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider --basetemp=.tmp_pytest_run/local -q
```

## 常用 A 股数据初始化

```bash
python scripts/astock_metadata.py --date 2026-05-29 --output user_data/astock_metadata_2026-05-29.feather
python scripts/astock_index.py --code sh.000300 --start-date 2020-01-01 --end-date 2026-05-31 --output-dir user_data/astock_indexes
python scripts/astock_batch_download.py --pool-file user_data/astock_pools/baostock_wide_500.txt --start-date 2020-01-01 --end-date 2026-05-31 --output-dir user_data/astock_baostock_wide_500 --per-code-timeout 90
```

这些输出都是本地数据产物，已经被 `.gitignore` 忽略。
