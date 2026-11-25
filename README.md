# 组合优化候选方案选择实验

在同一组起点和相同的评分次数下，比较三种选择方式：由模型选择、按符号回归预测选择、随机选择。实验覆盖三个完整的公开题集，共 175 题。每组最终交付一个可行方案、它的真实得分以及每轮的选择记录。

[实测结果](results/README.md) · [逐题完整记录](results/per_run.csv) · [公开领先成绩与比较口径](results/references.md)

### 流水车间的独立复验

[复验成绩](results/confirmation/README.md)包含 Taillard 全部 120 题的新种子试验，以及 OR-Library 全部 31 道新题。沿用下方的安装与 API 配置后，可运行以下命令。第二条是仅针对 OR-Library 的补充对照，双方同时看到全部历史观测。

```sh
python -m selection_bench.confirm --workers 4
python -m selection_bench.confirm --all-history --workers 4
python -m selection_bench.analyse_confirmation
```

只复核已发布的实验时，运行 `python -m selection_bench.analyse_confirmation --replay-csv results/confirmation/runs.csv`。程序从实际选项编号重建排程、计算工期和配对统计，无需 API Key。

## 题集与评分

| 题集 | 题数 | 一个方案是什么 | 真实得分，均为越小越好 | 公开参照 |
|---|---:|---|---|---|
| 无容量仓库选址（UFL） | 15 | 选择至少一个仓库开放 | 开仓费用加上所有顾客的最低分配费用 | [OR-Library 题目及最优值](https://people.brunel.ac.uk/~mastjjb/jeb/orlib/uncapinfo.html) |
| 图上设施选址（p-median） | 40 | 从图中选固定数量的节点作为设施 | 所有节点到最近设施的最短距离之和 | [OR-Library 定义](https://people.brunel.ac.uk/~mastjjb/jeb/orlib/pmedinfo.html) · [逐题最优值](https://people.brunel.ac.uk/~mastjjb/jeb/orlib/files/pmedopt.txt) |
| 置换流水车间调度 | 120 | 给出全部工件的加工顺序，各台机器沿用该顺序 | 最后一件工件在最后一台机器上的完成时间 | [原始题目生成器](https://people.brunel.ac.uk/~mastjjb/jeb/orlib/flowshopinfo.html) · [公开排程与上界](https://zenodo.org/records/24342515) · [最优性记录](https://zenodo.org/records/4542886) |

UFL 和 p-median 每题均有已证最优值。流水车间逐题区分已证最优值和公开可行排程取得的最好已知上界。相对差距为 `100 × (本次最佳得分 − 参考值) / 参考值`。对上界得到 0% 表示追平参考排程，对已证最优值得到 0% 表示达到最优。不同题集的费用、距离与时间单位分别汇总。

## 一次实验怎样进行

1. 每题使用随机种子 11、29、47。对同一题和同一种子，六种策略共享 12 个不同的初始可行方案及其真实得分。
2. 此后进行 12 轮。每轮从迄今最好的方案出发，生成 8 个不同且尚未评分的合法邻居。例如，p-median 在每个邻居中换掉一个设施，流水车间交换或移动一个工件。
3. 策略选一个候选，程序计算它的真实得分并加入历史。下一轮重新生成 8 个选项。策略选了不同的方案后，后续候选也可能不同。
4. 每条轨迹实际评分 24 次。全套为 175 题 × 3 种子 × 6 策略，共 3,150 条轨迹。每题的公开参考值仅用于事后评分。

模型看到的是任务说明、带列名的方案特征、迄今最好及最近两条已测记录（去重），以及 8 个候选的特征。特征和得分使用 12 个初始方案的均值和标准差固定换算。模型回答一个候选编号 0 至 7。符号回归（SR）每轮使用已有的全部已测记录重新拟合；`x0` 起依次对应提示中的特征列。

| 命令中的策略名 | 如何选择下一项 | 使用模型 API |
|---|---|---|
| `random` | 在 8 个候选中均匀随机选一项 | 否 |
| `deepseek` | 模型根据结构特征和已有观测选择 | 是 |
| `deepseek-operon` | 模型同时收到 Operon 的 8 个预测值和拟合公式 | 是 |
| `deepseek-gplearn` | 模型同时收到 gplearn 的 8 个预测值和拟合公式 | 是 |
| `greedy-operon` | 直接选 Operon 预测得分最低的一项 | 否 |
| `greedy-gplearn` | 直接选 gplearn 预测得分最低的一项 | 否 |

模型使用 SR 的两组都同时增加预测和公式，因此这六组能检验整套 SR 信息的作用；要单独估计展示公式的作用，还需要相同 SR 后端的仅预测组。公开最优值用于衡量结果离目标多远。论文中其他算法的耗时或评分次数需另行匹配后才能比较效率。

## 安装与准备数据

以下命令均在项目根目录执行。本地验证使用 64 位 CPython 3.13。使用其他平台时，可先查看 pyoperon 为该 Python 版本提供的系统与架构组合。[pyoperon 安装包](https://pypi.org/project/pyoperon/0.6.1/#files)

Windows PowerShell：

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r selector-requirements.txt
```

macOS / Linux：

```sh
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r selector-requirements.txt
```

下载并校验 3 套公开数据。已校验的文件会在后续运行中复用，默认位置为 `.cache/`。

```sh
python -m selection_bench.run --prepare-only
```

要核验已发布的 3,150 条轨迹，准备好数据后可直接运行 `python -m selection_bench.verify --csv results/per_run.csv`。该命令使用逐轮选择编号重建实际解并复算目标值。

先在一题上运行三种本地策略。运行完可以打开 `outputs/readme-check/pmedian/pmed1/11/` 中的三个 JSON 文件查看解、分数和各轮选择。

```sh
python -m selection_bench.run --suite pmedian --case pmed1 --seed 11 --policy random greedy-operon greedy-gplearn --out outputs/readme-check
```

## 接入模型与开展完整实验

三个模型策略共用一个 Chat Completions 客户端。环境变量 `DEEPSEEK_BASE_URL` 填接口前缀，程序会追加 `/chat/completions`；`DEEPSEEK_MODEL` 填服务商的确切模型 ID。接口需要支持关闭思考、JSON 对象输出和温度 0。API Key 在本机终端输入。

Windows PowerShell：

```powershell
$env:DEEPSEEK_BASE_URL = Read-Host 'API base URL'
$env:DEEPSEEK_MODEL = Read-Host 'Model ID'
$env:DEEPSEEK_API_KEY = [System.Net.NetworkCredential]::new('', (Read-Host 'API key' -AsSecureString)).Password
```

macOS / Linux（Bash 或 Zsh）：

```sh
read -r DEEPSEEK_BASE_URL
read -r DEEPSEEK_MODEL
read -rs DEEPSEEK_API_KEY
export DEEPSEEK_BASE_URL DEEPSEEK_MODEL DEEPSEEK_API_KEY
```

先运行单题模型示例：

```sh
python -m selection_bench.run --suite pmedian --case pmed1 --seed 11 --policy deepseek --out outputs/model-check
```

需要全部题目、种子和策略时运行：

```sh
python -m selection_bench.run --out outputs/selection
```

`--suite`、`--case`、`--seed`、`--policy` 可限定范围；`--workers` 控制并发轨迹数，默认 1。中断后以相同参数及模型重新运行可复用已完成的轨迹和已保存的选点。更换模型或代码时使用新的 `--out` 目录，并保存实际模型 ID 及依赖版本，避免把不同配置混在同一汇总中。

## 汇总与离线核验

完成整套运行后，先生成成绩表，再从保存的选点编号重新生成候选并复算每一步的真实得分。核验阶段使用本地题目缓存，无需 API 或重新拟合 SR。

```sh
python -m selection_bench.summarize --raw outputs/selection --out outputs/selection-summary
python -m selection_bench.verify --csv outputs/selection-summary/per_run.csv --cache .cache --traces outputs/selection
```

| 输出文件 | 内容 |
|---|---|
| `per_run.csv` | 每题、每种子、每策略的解、得分、选择编号及逐轮得分 |
| `summary.csv` | 各题集每种子的平均差距、三个种子间的样本标准差和参考值命中数 |
| `paired.csv` | 同题同种子的两种策略谁赢、打平或输，以及平均差距相差多少个百分点 |

`python -m selection_bench.progress` 从发布的逐轮记录计算进展曲线与配对区间。`python -m selection_bench.rank_options --csv results/per_run.csv` 会在实验结束后，单独计算每个已选方案在当轮八个候选中的真实排名。

只汇总一个完整题集时，在汇总命令追加 `--suite pmedian`（也可用 `ufl` 或 `flowshop`），在核验命令追加 `--partial`。仅有单题试跑时，直接查看对应的轨迹 JSON。已有完整 `per_run.csv` 时，去掉 `--traces` 即可从 CSV 独立复算；加上该参数可再与原轨迹核对。离线复算检验选择对应的方案与分数；重新调用模型可能返回不同编号。

## 复算专用算法

这些算法可以利用完整的题目结构，帮助衡量最终解与传统算法的差距。下列命令可独立运行。

```sh
python -m selection_bench.ufl_solver
python -m pip install kmedoids==0.5.5 permutation-flowshop==1.0.3
python -m selection_bench.pam
python -m selection_bench.neh
```

此前的连续优化脚本可按[对应版本的使用说明](https://github.com/yxue301603/bbob-api/blob/99c6d698493dc0bec0af778347f3d6ed9ca03c9f/README.md)运行。
