[README.md](https://github.com/user-attachments/files/27707460/README.md)
# 大语言模型本地部署与中文语义理解横向对比

本项目为《人工智能导论》第三次作业实践。项目基于阿里云 ModelScope（魔搭）社区提供的免费 CPU 资源，完成了 **通义千问 (Qwen-7B-Chat)** 与 **智谱 (ChatGLM3-6B)** 两款百亿参数开源大语言模型的本地非量化部署与推理测试。

同时，本项目针对中文自然语言处理中的语用学、句法倒装、多层逻辑嵌套等复杂场景设计了“压力测试”，并对两款模型的语义理解能力和“幻觉”退化模式进行了深度的横向对比分析。

------

## ⚠️ 注意事项与平台限制

- 资源释放机制**：ModelScope 平台的 CPU 资源如果 **1小时未使用会自动释放，且配置的运行环境与临时文件会被清空 。建议及时保存代码脚本。  
- 存储限制**：由于云端存储空间有限，**一次最好只下载并运行一个大模型，否则极易导致存储不足报错 。  

------

##      **测试用例：**

- 请说出以下两句话区别在哪里？ 1、冬天：能穿多少穿多少 2、夏天：能穿多少穿多少
- 请说出以下两句话区别在哪里？单身狗产生的原因有两个，一是谁都看不上，二是谁都看不上
- 他知道我知道你知道他不知道吗？ 这句话里，到底谁不知道
- 明明明明明白白白喜欢他，可她就是不说。 这句话里，明明和白白谁喜欢谁？
- 领导：你这是什么意思？ 小明：没什么意思。意思意思。 领导：你这就不够意思了。 小明：小意思，小意思。领导：你这人真有意思。 小明：其实也没有别的意思。 领导：那我就不好意思了。 小明：是我不好意思。请问：以上“意思”分别是什么意思。

------



## 一、 平台搭建与环境配置

本项目基于 ModelScope 的 DSW (Data Science Workshop) Notebook 环境运行，选择基础的 **Ubuntu CPU 环境** 。  



### 1. 启动终端

进入 DSW 实例后，点击 `Terminal` 图标，打开终端命令行环境 。我们选择直接在 `root` 目录下进行环境配置 。  



### 2. 安装基础依赖库

由于是 CPU 推理部署，首先需要安装 CPU 版本的 PyTorch 及相关视觉库 ：  

Bash

```
pip install torch==2.3.0+cpu torchvision==0.18.0+cpu --index-url https://download.pytorch.org/whl/cpu
```

检查 pip 联网并更新基础打包工具 ：  

Bash

```
pip install -U pip setuptools wheel
```

安装核心推理依赖（注意锁定版本号以兼容框架） ：  

Bash

```
pip install \
"intel-extension-for-transformers==1.4.2" \
"neural-compressor==2.5" \
"transformers==4.33.3" \
"modelscope==1.9.5" \
"pydantic==1.10.13" \
"sentencepiece" \
"tiktoken" \
"einops" \
"transformers_stream_generator" \
"uvicorn" \
"fastapi" \
"yacs" \
"setuptools_scm"
```

安装对话框架及辅助工具 ：  

Bash

```
pip install fschat --use-pep517
pip install tqdm huggingface-hub
```

------

## 二、 模型下载与推理实例

### 1. 模型下载

进入平台的数据挂载目录，通过 Git 将模型仓库克隆到本地 ：  

Bash

```
cd /mnt/data

# 下载通义千问模型
git clone https://www.modelscope.cn/qwen/Qwen-7B-Chat.git

# 或者下载智谱模型
# git clone https://www.modelscope.cn/ZhipuAI/chatglm3-6b.git
```

### 2. 编写推理脚本

切换到工作目录 `/mnt/workspace` ，创建并编写推理脚本 `run_qwen_cpu.py` ：  

Python

```
from transformers import TextStreamer, AutoTokenizer, AutoModelForCausalLM

# 配置本地模型路径
model_name = "/mnt/data/Qwen-7B-Chat" 
# 设置测试 Prompt
prompt = "请说出以下两句话区别在哪里？ 1、冬天：能穿多少穿多少 2、夏天：能穿多少穿多少"

# 加载 Tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True
)

# 加载模型
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    torch_dtype="auto" # 自动选择 float32/float16
).eval()

# 编码输入
inputs = tokenizer(prompt, return_tensors="pt").input_ids

# 设定流式输出并生成
streamer = TextStreamer(tokenizer)
outputs = model.generate(inputs, streamer=streamer, max_new_tokens=300)
```

### 3. 运行推理

在终端执行脚本 ：  

Bash

```
python run_qwen_cpu.py
```
