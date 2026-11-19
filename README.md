# BBOB 公式选点 API 实验

本目录可以单独复制运行。实验采用五维 COCO BBOB，在相同起点和查询预算下比较历史观测、原始符号回归（SR）、Kriging 残差修正、LCF 变量组合和固定规则。模型选点通过供应商的 Chat Completions 接口直接发送 HTTP 请求，身份凭据来自环境变量。每轮为一次独立请求。

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
