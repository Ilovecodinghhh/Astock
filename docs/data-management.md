# 数据管理与跨电脑协作

这个仓库只适合保存源码、测试、策略定义、小体积股票池和必要说明。大型数据、机器相关产物和本地配置不要直接放进 Git。

## 不建议提交的数据

下面这些路径已经被 `.gitignore` 忽略：

- `user_data/astock_baostock_*/*.feather`
  - baostock 下载的 A 股日线、成交额、估值等缓存。
  - 文件多、总体体积大，而且可以按股票池和日期区间重新生成。
- `user_data/astock_indexes/*.feather`
  - 指数行情，例如 `sh.000300-d.feather`。
  - 用 baostock 在本机重建即可。
- `user_data/astock_metadata*.feather`
  - 行业、上市日期、退市日期、状态等元数据。
  - 和查询日期有关，建议用命令重建。
- `user_data/astock_fundamentals*.feather`
  - 季度财务数据。
  - 可能下载较慢，但仍属于生成数据，不适合进 Git。
- `user_data/data/`、`user_data/cross_sectional_features/`
  - Freqtrade 下载的币圈行情和生成特征。
  - 体积较大，且和本机运行环境强相关。
- `crypto_freqtrade/user_data/`、`crypto_freqtrade/backtest_results/`、`crypto_freqtrade/hyperopt_results/`、`crypto_freqtrade/logs/`、`crypto_freqtrade/plot/`
  - 加密货币/Freqtrade 的本地运行数据、回测输出、参数搜索结果、日志和图表。
  - 这些内容和机器环境、下载区间、交易所配置强相关，不适合放进 Git。
- `scan_results/`、`user_data/backtest_results/`、`user_data/hyperopt_results/`、`user_data/logs/`、`user_data/plot/`
  - 回测、参数搜索、日志和图表输出。
  - 需要讨论结果时，优先把少量精选结论整理进文档，而不是提交原始输出目录。
- `user_data/config*.json`、`.env`、`.env.*`
  - 本地运行配置。
  - 可能包含交易所配置、路径、凭据或其他私密信息。
- `.tmp*`、`.pytest_cache/`、`__pycache__/`、`*.pyc`
  - 测试、编译、运行缓存。

除非非常明确，否则不要用 `git add -f` 强行加入这些文件。即使单个文件没有超过 GitHub 限制，大量行情文件也会让仓库膨胀，后续克隆、拉取、回滚都会变慢。

## 建议提交的内容

- `scripts/` 下的源码。
- `tests/` 下的测试。
- `user_data/astock_pools/` 下的小体积股票池文本。
- `crypto_freqtrade/strategies/` 下的历史加密货币 Freqtrade 策略源码。
- `docs/` 下的说明文档。
- 经过筛选、用于解释决策的小型结果摘要。

## 新电脑初始化

先拉代码并安装核心依赖：

```bash
git clone https://github.com/Ilovecodinghhh/Astock.git
cd Astock
python -m venv .venv
source .venv/Scripts/activate
python -m pip install -r requirements.txt
```

Linux/macOS 使用 `source .venv/bin/activate`。

然后按需要重建本地数据。

### wide_500 A 股日线缓存

```bash
python scripts/astock_batch_download.py \
  --pool-file user_data/astock_pools/baostock_wide_500.txt \
  --start-date 2020-01-01 \
  --end-date 2026-05-31 \
  --output-dir user_data/astock_baostock_wide_500 \
  --per-code-timeout 90
```

### 股票元数据

```bash
python scripts/astock_metadata.py \
  --date 2026-05-29 \
  --output user_data/astock_metadata_2026-05-29.feather
```

### 指数择时数据

```bash
python scripts/astock_index.py \
  --code sh.000300 \
  --start-date 2020-01-01 \
  --end-date 2026-05-31 \
  --output-dir user_data/astock_indexes
```

### 财务数据

只有依赖财务字段的策略才需要下载：

```bash
python scripts/astock_fundamentals.py \
  --pool-file user_data/astock_pools/baostock_wide_500.txt \
  --start-year 2020 \
  --end-year 2026 \
  --output user_data/astock_fundamentals_2020_2026.feather \
  --sleep 0.05
```

## 无法放进 Git 的数据怎么协作

跨电脑协作时，优先用下面几种方式：

- 能公开重建的数据，直接用脚本重新下载。
- 大型产物放到 Git 之外的地方，例如 NAS、云盘、对象存储或 GitHub Release 附件。
- Git 里只保存 manifest，不保存原始数据。manifest 至少写清楚数据来源、日期区间、股票池、生成命令、输出路径；如果需要精确复现，再加 checksum。

示例：

```text
artifact: user_data/astock_baostock_wide_500
source: baostock
pool: user_data/astock_pools/baostock_wide_500.txt
range: 2020-01-01 to 2026-05-31
command: python scripts/astock_batch_download.py --pool-file user_data/astock_pools/baostock_wide_500.txt --start-date 2020-01-01 --end-date 2026-05-31 --output-dir user_data/astock_baostock_wide_500 --per-code-timeout 90
```

## Freqtrade 相关说明

当前 Git 里只保留 `crypto_freqtrade/strategies/*.py` 策略源码。Freqtrade 的运行目录、下载行情、hyperopt 结果、日志和配置 JSON 都是本地产物。需要继续跑 Freqtrade 时，在每台机器上单独安装和配置 Freqtrade。
