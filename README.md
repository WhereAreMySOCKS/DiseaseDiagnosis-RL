## 1. 简介

- 基于医疗知识图谱和强化学习方法，完成疾病诊断任务。

- 尝试了基于TransE，TransH，TransR的知识图谱表示学习方法。

- 强化学习方法使用了经典的Critic-Actor算法


## 2. 数据集

- 使用了两个数据集，其中一个公开数据集来自**[这里](http://www.openkg.cn/dataset/disease-information)**：


## 3. 执行流程简介

- 运行train_transe_model.py, 学习知识图谱表示，学习结束后将每个实体和关系的表示保存。
- 运行train_agent.py, 加载训练好的表示向量，训练强化学习智能体。
- 运行test_agnent.py, 输出测试结果。

- 参数解释：
-- '--dataset': 选择Aier或Medical数据集
-- '--embedding_type': 知识图谱表示算法，可选择TransE，TransH，TransR
-- '--enhanced_type': 增强表示算法，可选择none，embedding，w2v

-- '--max_acts': 强化学习候选项最大值
-- '--max_path_len': 生成路径长度
-- '--path_limit': 是否对路径生成进行限制



 
## 4. 代码结构与简要说明

```undefined
./repo_template               # 项目文件夹名称
|-- data                      # 数据集一
|-- medical_data              # 数据集二
|-- data_processor.py         # 针对数据集一的数据预处理文件
|-- data_utils.py             # Dataset类文件
|-- draw.py                   # 绘制曲线图
|-- kg_env.py                 # 生成强化学习的环境
|-- knowledge_graph.py        # 根据Dataset生成知识图谱
|-- medical_processor.py      # 针对数据集二的预处理文件
|-- test_agent.py             # 测试强化学习智能体
|-- train_test.py             # 训练强化学习智能体
|-- train_transe_model.py     # 知识图谱表示学习
|-- transe_model.py           # 知识图谱模型
|-- utils.py                  # 工具文件
|-- README.md                 # 中文用户手册

```
