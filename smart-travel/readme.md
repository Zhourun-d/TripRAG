# 智能旅行规划系统 - RAG 知识库

基于 RAG 的智能旅行规划系统，通过高德 API 抓取景点数据，
用通义千问自动打标签，构建可检索的向量知识库，并建立评估体系。

## 项目结构

```
smart-travel-rag/
├── config/                      # 配置
│   ├── cities.yaml              # 城市抓取配置
│   └── schema.yaml              # POI 字段规范
├── data/
│   ├── raw/                     # 高德原始数据
│   ├── processed/               # 清洗 + 打标签后数据
│   ├── knowledge_base/          # 最终知识库（jsonl）
│   └── eval/
│       ├── {city}_eval.json     # 评估集
│       └── reports/             # 评估报告
├── src/
│   ├── config.py                # 配置加载
│   ├── logger.py                # 日志
│   ├── crawl/                   # 高德抓取
│   ├── clean/                   # 清洗 + 两层过滤
│   ├── tag/                     # LLM 打标签
│   ├── build/                   # 知识库构建
│   ├── retrieve/                # 检索接口
│   └── eval/                    # 评估
├── chroma_db/                   # 向量库（每城市一个 collection）
├── logs/                        # 日志
├── .env                         # API Key
└── requirements.txt
```

## 核心设计

### 1. 配置驱动

所有城市配置放在 `config/cities.yaml`，换城市只改配置，不改代码。
POI 字段规范放在 `config/schema.yaml`，加字段只改配置，不改代码。

### 2. 五步流水线

```
crawl → clean → tag → build → eval
```

每一步都是独立模块，输入输出为文件，支持单步重跑。

### 3. 两层过滤

清洗阶段采用两层过滤：

- **硬过滤**：地铁站、停车场等，命中即干掉，不可豁免
- **软过滤**：学校、政府机关等，命中后看名字是否含豁免关键词

### 4. 检索去重

检索结果做同名去重（按名字前 3 字分组），同一主景点最多返回 2 条，
避免“黄鹤楼红墙”“黄鹤楼故址”霸榜。

### 5. 评估体系

用三个指标量化检索效果：

- **Hit Rate@K**：期望结果有多少被检索到
- **MRR**：第一个期望结果的排名倒数
- **Noise Rate**：不该出现的混入 top-K 的比例

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 配置环境变量

在项目根目录建 `.env`：

```
AMAP_API_KEY=你的高德key
DASHSCOPE_API_KEY=你的dashscope key
```

### 跑完整流水线（以武汉为例）

```bash
python -m src.crawl.amap_crawler --city wuhan
python -m src.clean.poi_cleaner --city wuhan
python -m src.tag.poi_tagger --city wuhan
python -m src.build.kb_builder --city wuhan
python -m src.eval.evaluator --city wuhan
```

### 单独测试检索

```bash
python -m src.retrieve.retriever --city wuhan \
  --query "武汉 历史人文 古建筑" --top_k 10
```

## 当前状态

- 支持城市：武汉（重庆、南京待跑）
- 知识库规模：250 条 POI
- 检索指标：Hit Rate 0.500, MRR 0.458, Noise Rate 0.000

## 技术栈

- **数据源**：高德地图 API
- **LLM**：通义千问（qwen-turbo / qwen-plus）
- **Embedding**：DashScope text-embedding-v2
- **向量库**：Chroma
- **框架**：LangChain

## 待办

- [ ] 优化 combo 类 query（多路召回）
- [ ] 优化 landmark 类 query（地理邻近检索）
- [ ] 扩展到重庆、南京
- [ ] 智能问答助手
- [ ] 行程生成
- [ ] 遗传算法多目标优化