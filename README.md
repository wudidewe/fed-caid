# FedCAID

FedCAID provides reproducible local and federated intrusion-detection experiments
for CAN and network flows. It uses a shared TabM workflow, dataset-specific feature
and label schemas, and matched controls for local adaptation, donor pooling and
federated aggregation.

## Reference experiment

This source matches the Python files recorded for run
`run_20261117_114937_guhefcl1` on 17 November 2026. The four dataset primary
comparison uses grouped development validation and five seeds from 42 to 46.
The CAN quantization, later day CIC-UNSW-NB15, and compact adapter transfer
diagnostics report their own retrospective test partitions. Independent Hulk
confirmation and cross-host LAN measurement require separate declared inputs.

Run `python main.py --smoke` to check a clean CPU installation, or
`python main.py --full-review` after preparing all four public provider datasets.
The full run writes `final_plan.json`, `final_complete.json`, and
`report/SUMMARY.md` under its own output directory.

## 一键运行

1. 下载并解压源码，在 PyCharm 打开包含 [main.py](main.py) 的目录。
2. 选用 Python 3.11 解释器，并在该解释器中安装 [requirements.txt](requirements.txt)。
3. 打开 `main.py`，点击底部 `if __name__ == "__main__"` 旁的绿色三角。运行参数留空。

`src` 是项目的 Python 包。PyCharm 打开项目根目录即可解析 `from src...` 导入。

无参数启动会检查项目下的 `dataset/`。未放入任何所需 CSV 时，程序运行合成数据 CPU 检查，
适用于刚下载的源码，回执位于 `outputs/fedcaid_demo/run_*/benchmark/complete.json`。
只放入部分输入时，程序列出缺失的文件。全部输入就绪后，同一次点击
启动全部本地实验，从源数据创建新的运行目录并训练。正式实验的文件来源和结构见
[数据集准备](dataset/README.md)。完整运行通常需要数小时，结束后打开控制台给出的
`outputs/fedcaid_review/run_*/report/SUMMARY.md`。

完整运行从原始数据训练 NB15 参考 TabM
和难正常样本加权候选，按同一数据划分、五个种子和训练预算比较二元误报、漏报及分类表现。
难样本权重仅取决于 TRAIN。子类型对照使用**同一轮**参考检测器的告警和分数。供体 CAL
用精确分数断点选出一个改判阈值，接收端 CAL 只检查是否接受或回退。VAL 用于评价。

Hulk 的五种子匹配对照也重新训练。完整运行启动时预检 `external/hulk_manifest.json` 和
`external/lan_manifest.json`。清单须在点击前准备好。独立确认使用本轮新训练的冻结包。
配置缺失时，相应阶段显示 `pending`，本地实验继续。实验报告使用本轮产物，历史 CAN
量化、规模和异质性结果留在各自的归档中。

四数据集主对照统一用 TabM、分组 TRAIN/CAL/VAL 和原始 K−1 适配器协议。
论文分别引用四数据集主对照和网络流量的受控缺类共享头实验。主对照给出基础、
本地适配、完整联邦及全权重 FedAvg 的匹配指标和配对差。分组 VAL 来自原有采集，
本阶段只审计保留 TEST 的数据角色。外部采集确认由独立验证阶段单独提供。

本轮还重新执行两套流量数据的规模与多 α 对照、原始 K−1 缺类迁移分数诊断、两套 CAN 的
标准与精细量化对照、NB15 按时间留出，以及同机 TCP 回环。已有来源的 TEST 评分标为
回顾性评价。回环以合成模型更新测试 3、5、10 个客户端和两种维度，共 180 轮。

| 输出 | 内容 |
| --- | --- |
| `run_*/final_plan.json` | 启动预检、声明文件哈希、入口及模块源码哈希 |
| `run_*/final_complete.json` | 本轮阶段执行、待输入和失败状态 |
| `run_*/iov_diagnosis/` | 逐客户端工作点、逐类指标、配对差和固定告警验收 |
| `run_*/nb15_pair/` | 两套 TabM 模型及五种子对照判定，见 `binary_pair_status.json` |
| `run_*/targeted/` | 本轮 NB15 子类型对照 |
| `run_*/nb15_low_far/` | 参考 TabM、冻结骨干与完整微调的匹配工作点、逐类结果和验收回执 |
| `run_*/nb15_threshold/` | 同一轮参考检测器的三个 CAL 阈值规则、工作点和验收回执 |
| `run_*/donor_check/` | 供体 CAL 可行阈值、接收端检查和回退、匹配指标和配对差 |
| `run_*/hulk_attribution/` | 本轮 Hulk 基础、本地和联邦匹配结果 |
| `run_*/benchmark_matrix/` | 四数据集五种子主对照、逐类结果及 CAL 分组回执 |
| `run_*/scale_matrix/` | 固定总量、固定每客户端量和多 α 的本轮规模结果 |
| `run_*/can_quantization/` | 两套 CAN 的分组与冻结角色量化对照 |
| `run_*/nb15_temporal/` | 清除跨角色精确输入后的 NB15 同源时间留出 |
| `run_*/kminus1_transfer/` | 原始 K−1 的缺类迁移和分数级配对 |
| `run_*/loopback/` | 本轮 180 轮 TCP 回环、应用层字节与聚合一致性 |
| `run_*/report/SUMMARY.md` | 实验覆盖、科学判定、NB15 工作点、Hulk 同告警主表和配对区间 |

每轮保留固定种子以便比较，运行回执记录本轮代码哈希和产物来源。需要调用专项时，在同一
入口指定 `--iov-review`、`--nb15-low-far` 或 `--nb15-threshold`。需要恢复兼容的
中断运行时，使用 `--resume RUN`，其中 `RUN` 为 `outputs/fedcaid_review/` 下的
运行目录名。需要重建已有报告时，使用 `--refresh-report <运行目录>`。派生报告写入
`outputs/fedcaid_review_reports/`，并记录原运行来源与哈希。

`--full-review` 要求提供真实数据。`--local-evidence` 核验本地已完成的四数据集、
IoV、时间留出与单机回环归档，汇总写入 `outputs/fedcaid_local_evidence/review_*/report/`。

## Installation and verification

Install the pinned dependencies in a Python 3.11 environment. PyCharm can install
them directly from `requirements.txt`. For a fresh CPU checkout, the same script
accepts `--smoke` and exercises the model pipeline on synthetic data. This mode
needs no real datasets or prior outputs. With no input files, the no-argument
launch also runs this check. The Windows and Linux CI workflow checks the source
entry and verifies that the installed `src` package imports outside the checkout.

```text
python -m pip install -r requirements.txt
python main.py
```

`--preflight-only` checks the optional Hulk and LAN inputs without opening
capture contents, loading old model results or connecting to endpoints. `--hulk-manifest` and `--lan-manifest`
can point to alternative declarations.

## Evidence and scope

- [Dataset preparation](dataset/README.md) describes the original sources, identities and feature schemas.
- [Hulk external preflight](docs/external_preflight.md) describes independent acquisition, seal and confirmation.
- [Protocol validation](docs/protocol_validation.md) describes collector setup and the measured security and timing scope.
- [Validation and reproduction](docs/review_readiness.md) describes what was measured and lists external inputs needed for further validation.
- [Software comparison](docs/software_comparison.md) distinguishes the simulator, framework and matched detector comparisons.

The software is available under the [Apache-2.0 licence](LICENSE). Third-party
notices and upstream licences are stored under `src/vendor`. Dataset terms are
set by their respective providers.
