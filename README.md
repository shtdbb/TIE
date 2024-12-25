# TIE 模型项目文档


## 概述

TIE (Temporal Information Extraction) 模型是一个用于带时间信息的实体和关系抽取的深度学习模型。该模型基于ERNIE（Enhanced Representation through kNowledge Integration）架构，能够从文本中提取三元组的结构化时间信息，如开始时间和结束时间。项目中包含了模型定义、分词器扩展以及示例代码。


## 目录结构

```README.md
project/
├── model/
│   ├── tie/
│   │   ├── __init__.py
│   │   └── modeling_tie.py
│   └── ernie3_base/
│       ├── __init__.py
│       └── modeling_ernie.py
├── demo.py
├── script/
│   └── add_special_token.py
└── README.md
```


## 依赖库

- `transformers`
- `torch`


## 模型定义

### `model/tie/modeling_tie.py`
该文件定义了TIE模型及其分词器。


#### 类 `TIE`

- **描述**: TIE模型，继承自`torch.nn.Module`。

- **参数**:

  - `encoder_config_path`: ERNIE编码器配置文件路径。


- **方法**:
  - `__init__`: 初始化模型，加载ERNIE编码器配置并构建编码器、开始和结束分类器。

  
  - `forward`: 前向传播函数，接收输入并返回开始和结束时间的logits。


#### 类 `TIETokenizer`

- **描述**: 扩展自`BertTokenizer`，用于处理TIE模型的特殊token。

- **方法**:

  - `_get_extended_attention_mask`: 生成扩展的注意力掩码。
  
  - `split_chunk`: 将文本分割成指定大小的块。
  
  - `get_inputs`: 根据任务类型生成输入，包括头实体抽取、关系抽取和时间抽取。


### `model/ernie3_base/modeling_ernie.py`

该文件定义了ERNIE模型及其各种变体，如`ErnieModel`、`ErnieForMaskedLM`等。TIE模型使用其中的`ErnieModel`作为编码器。


## 示例代码

### `demo.py`

该文件展示了如何使用TIE模型进行时间信息抽取。

- **步骤**:

  1. 加载预训练的ERNIE模型和TIE分词器。
  2. 准备输入文本并进行分词。
  3. 将分词后的输入传递给模型并获取输出。
  4. 打印模型输出的logits。


### `script/add_special_token.py`

该脚本用于向ERNIE分词器添加特殊token，以便TIE模型能够正确处理这些token。


- **步骤**:

  1. 加载预训练的ERNIE分词器。

  2. 添加特殊token，如`[Y]`、`[M]`、`[D]`、`[U]`、`<memory>`和`</memory>`。

  3. 保存修改后的分词器。


## 使用方法

1. **安装依赖库**:
   ```bash
   pip install transformers torch
   ```

2. **添加特殊token**:
   ```bash
   python script/add_special_token.py
   ```

3. **运行示例代码**:
   ```bash
   python demo.py
   ```

## 贡献

欢迎对项目进行贡献，包括但不限于代码优化、文档改进和新功能开发。


## 许可证

本项目采用 [Apache License 2.0](LICENSE) 许可证。