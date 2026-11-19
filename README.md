# BBOB 公式选点 API 实验

本目录可以单独复制运行。实验采用五维 COCO BBOB，在相同起点和查询预算下比较历史观测、原始符号回归（SR）、Kriging 残差修正、LCF 变量组合和固定规则。模型选点通过供应商的 Chat Completions 接口直接发送 HTTP 请求，身份凭据来自环境变量。每轮为一次独立请求。

## COCO 标准评测与公开算法对照

`coco_run.py` 将每次真实函数调用交给 COCO 官方记录器，一次运行评测一种方法。结果 ZIP 可直接交给 `cocopp` 生成达标率曲线、达到目标所需的查询次数和 HTML 图表。原有 `run.py` 保留五维配对实验及其自定义指标。

先按下文安装依赖，激活虚拟环境，再安装后处理工具。以下命令适用于 Windows 和 macOS。

```sh
python -m pip install cocopp==2.9.3
python coco_run.py --arm raw_rule --functions 1 15 --dimensions 2 5 --instances 1 2 --budget 25 --out outputs/check
python -m cocopp --no-interactive -o outputs/plots outputs/check/raw_rule.zip 2009/NEWUOA_ros_noiseless
```

以上命令评测固定规则，共 8 个问题，最多 200 次真实查询，模型调用为零。最后一条命令会下载公开基线并生成对比图。公开基线覆盖更多问题，小样本的汇总曲线可能涉及不同的题目范围。要在严格相同的范围上比较，请下载下方的 NEWUOA 数据包到当前目录，再运行：

```sh
python coco_compare.py outputs/check/raw_rule.zip NEWUOA_ros_noiseless.tgz --budget 25 --target 1
```

输出按函数、维度和实例列出两边的命中次数和运行次数。命中表示在前 25 次真实查询内，最优目标值与已知最优值之差达到 1 或更小。重复运行先在同一问题内求成功率，各问题再等权平均。上述配置的实测值为固定规则 25%，NEWUOA 50%。[NEWUOA 官方数据包](https://raw.githubusercontent.com/numbbo/data-archive/gh-pages/data-archive/bbob/2009/NEWUOA_ros_noiseless.tgz)

**运行模型对照。** 按下文设置 `BBO_API_KEY`、`BBO_MODEL` 和服务地址。以下两组使用相同的起点、候选生成规则和预算；公式组额外看到 SR 表达式。每组最多 40 次模型调用。短名称用于图表标注，实际模型记录在输出目录的 `config.json` 中。

```sh
python coco_run.py --arm raw_numeric --name SR-numeric --functions 1 15 --dimensions 2 5 --instances 1 2 --budget 25 --out outputs/numeric
python coco_run.py --arm raw_formula --name SR-formula --functions 1 15 --dimensions 2 5 --instances 1 2 --budget 25 --out outputs/formula
python -m cocopp --no-interactive -o outputs/sr-plots outputs/numeric/SR-numeric.zip outputs/formula/SR-formula.zip
```

**规划全套评测。** `--full` 使用固定版本 COCO 的默认套件，共 24 个函数、6 种维度（2、3、5、10、20、40）、每组 15 个实例。预算包含 20 个初始点，其后每次从新生成的 20 个候选点中选一个进行真实查询。可以用 `--budget` 设每题固定次数，也可以用 `--budget-per-dim` 设每题预算为维度的倍数。先运行下列命令查看调用上限，移除 `--plan` 后才会执行。高维度和长历史也会增加单次模型请求的长度。

```sh
python coco_run.py --full --arm raw_formula --budget 70 --out outputs/full-formula --plan
```

同配置、同目录重跑可恢复。脚本利用已保存的选点重建完整 COCO 日志，已经保存的模型决策会直接复用。ZIP 仅在所选问题全部结束后生成。发布时请保留实际模型 ID、运行配置、代码版本和预算。COCO 接受与论文或预印本关联的完整数据集申请，按每个算法变体分别打包并向官方数据仓库提交收录请求。[官方发布说明](https://coco-platform.org/howto-publish.html)

## 安装

### macOS 快速开始（默认终端 zsh）

建议使用原生 CPython 3.13。Apple Silicon 有对应的 COCO 预编译包。Intel Mac 安装 COCO 2.8.3 时需要从源码编译，请先安装 Xcode Command Line Tools（`xcode-select --install`）。当前已在 Windows 完成端到端测试，macOS 已核对依赖发行文件，尚待真机验证。[COCO 发行文件](https://pypi.org/project/coco-experiment/2.8.3/#files)

在解压后的 `bbob-api` 目录打开终端，先运行本地自检：

```zsh
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python tests/test_workflow.py
```

自检通过后，在同一终端输入 API Key 和服务商提供的模型 ID，运行 4 次模型选点的小实验。默认使用 OpenAI API；其他兼容服务先设置 `export BBO_BASE_URL='你的服务商 API 根地址'`。

```zsh
read -rs 'BBO_API_KEY?API Key: '; echo
export BBO_API_KEY
read 'BBO_MODEL?Model ID: '
export BBO_MODEL
python run.py --functions 15 --instances 18 --arms k_numeric k_formula --rounds 2 --out outputs/pilot
```

下文的实验选项也适用于 macOS。激活虚拟环境后，用 `python run.py` 调用即可。

### Windows

本机验证环境为 Windows、Python 3.13.7。以下 PowerShell 命令在本目录执行。

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## 选择服务与模型

将地址设为你信任的供应商的 API 根地址，模型名使用该供应商为你的 Key 开放的实际模型 ID。默认地址为 OpenAI 公共 API 的 `https://api.openai.com/v1`。兼容服务需要支持 `POST /chat/completions`、system/user 消息和文本 JSON 回答。模型需返回形如 `{"pick":3}` 的候选编号。

```powershell
$env:BBO_BASE_URL = 'https://api.openai.com/v1'
$env:BBO_MODEL = Read-Host 'Model ID'
$secret = Read-Host 'API Key' -AsSecureString
$env:BBO_API_KEY = [System.Net.NetworkCredential]::new('', $secret).Password
```

Key 仅供当前进程及其子进程读取。也可用 `--key-env OPENAI_API_KEY` 指定已有的环境变量。模型通过 `--model` 或 `BBO_MODEL` 选择。对支持推理强度设置的模型，可以加 `--effort low`。供应商的模型版本、采样与服务设置会影响最终分数。

## 运行

**预览完整预算**。此命令无需密钥即可执行。

```powershell
.\.venv\Scripts\python.exe run.py --plan
```

**小规模实跑**。在 F15 的一个实例上比较 Kriging 数值组和公式组，各走 2 轮，共请求模型 4 次。

```powershell
.\.venv\Scripts\python.exe run.py --functions 15 --instances 18 --arms k_numeric k_formula --rounds 2 --out outputs/pilot
```

**完整实验**。默认测试 F8、F9、F15、F19，各取实例 18、19、20。每条路径共享同一任务的初始设计，先测 20 个点，每轮新生成 20 个候选点，选 1 个实测，再测 50 轮。完整运行包含 120 条路径、4,200 次模型选点，按路径累计 8,400 次函数评估。用量和费用取决于模型和服务商。

```powershell
.\.venv\Scripts\python.exe run.py --out outputs/full
```

| 选项 | 对应方法 |
|---|---|
| history | 模型根据历史观测和候选点坐标选点 |
| raw_numeric / raw_formula | 原 SR 预测值 / 预测值加公式 |
| k_numeric / k_formula | SR＋Kriging 预测值 / 预测值加公式 |
| l_numeric / l_formula | SR＋LCF＋Kriging 预测值 / 预测值加公式 |
| raw_rule / k_ei / l_ei | 原 SR 固定规则 / Kriging 预期改进规则 / LCF＋Kriging 预期改进规则，全部在本地计算 |

## 结果与续跑

每完成一步就保存一次。网络异常、额度不足或返回无效编号时，程序停止。修复后重复相同命令，从上一个保存点继续。一次请求已发送但响应未成功保存时，重跑该步可能产生额外 API 用量。若改换模型、函数、预算或代码，请用新的输出目录。

* `config.json` 保存模型、端点、实验参数、运行环境和代码指纹。
* 每条路径的 JSON 保存初始观测、候选点、拟合结果、模型请求消息、返回编号、供应商报告的模型名与 token 用量，以及所选点的真实目标值。
* `results.csv` 给出逐任务、逐方法的最终目标值、差距缩小比例、查询次数、成功调用数和 token 数。`summary.json` 按方法汇总。差距缩小比例＝（初始最好值－最终最好值）／（初始最好值－已知最优值）×100%。各任务等权平均。
* Token 来自已保存成功决策的服务端 usage 字段。缺失值在 CSV 中留空、在 JSON 中记为 null。固定规则组用量为零。初始评估按路径累计。

在本机运行离线接口测试：

```powershell
.\.venv\Scripts\python.exe tests/test_workflow.py
```

测试使用临时本地 HTTP 服务和合成密钥，覆盖十种设置、小规模完整流程、HTTP 429 中断、续跑和用量汇总。
