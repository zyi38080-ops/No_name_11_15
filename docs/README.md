# 文档索引

| 文档 | 内容 | 适合谁看 |
|---|---|---|
| [competition-report.md](competition-report.md) | 比赛项目报告：需求、创新、技术、实施、成效、展望、答辩脚本、评分对照 | 评委、答辩 |
| [product.md](product.md) | 产品定位、目标问题、非目标、用户原则、会话路径 | 产品/设计 |
| [architecture.md](architecture.md) | 处理链路、模块职责、状态机、存储与降级 | 工程 |
| [safety.md](safety.md) | 安全、隐私、伦理规范、禁止回复、规则语境 | 安全/合规 |
| [evaluation.md](evaluation.md) | 四层评测、60 场景、透明基线、在线评测、当前结果 | 评测/答辩 |
| [references.md](references.md) | 需求与设计依据的参考文献（GB/T 7714） | 评委/研究 |
| [model-card.md](model-card.md) | 系统用途、边界、架构、评测、安全、局限 | 评委/部署者 |
| [data-statement.md](data-statement.md) | 训练数据、知识库、场景、运行时数据、伦理 | 评委/合规 |
| [research/](research/) | 匿名问卷、半结构访谈提纲、知情同意模板 | 研究/伦理 |
| [acceptance.md](acceptance.md) | 第五阶段验收清单与本地复核结果 | 工程 |
| [phase-4-validation.md](phase-4-validation.md) | 第四阶段工程验证范围 | 工程 |
| [branding.md](branding.md) | Agent 名称与构建期配置 | 前端/部署 |
| [fixes/](fixes/) | 历史修复说明 | 工程 |

## 快速命令

```powershell
# 一条命令验证（需先激活 Python 环境并安装前后端依赖）
powershell -ExecutionPolicy Bypass -File scripts/verify.ps1
```

```powershell
cd backend
python -m noname.evaluation
python -m noname.benchmark
```

> 自动评测只验证工程与对话规则，不代表临床疗效；在线结果受模型、供应商、网络和采样随机性影响。
