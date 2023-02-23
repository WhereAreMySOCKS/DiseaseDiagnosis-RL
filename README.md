## 1. 简介

- 基于医疗知识图谱和强化学习方法，完成疾病诊断任务.。

- 尝试了基于TransE，TransH，TransR的知识图谱表示学习方法。

- 强化学习方法使用了经典的Critic-Actor算法


## 2. 数据集

- 使用了两个数据集，其中一个公开数据集来自**[这里](http://www.openkg.cn/dataset/disease-information)**：
 
## 3. 代码结构与简要说明

### 3.1 代码结构

- 代码目录结构

```undefined
./repo_template               # 项目文件夹名称，可以修改为自己的文件夹名称
|-- config                    # 配置类文件夹
|   ├── competition.json      # 项目配置信息文件
|-- dataset                   # 数据集类文件夹
|   ├── dataset.py            # 数据集代码文件
|-- log                       # 日志类文件夹
|   ├── train.log             # 训练日志文件
|-- model                     # 模型类文件夹
|   ├── full_regression.pkl   # 训练好的模型文件
|-- preprocess                # 预处理类文件夹
|   ├── preprocess.py         # 数据预处理代码文件
|-- tools                     # 工具类文件夹
|   ├── train.py              # 训练代码文件
|   ├── eval.py               # 验证代码文件
|   ├── averaging_model.py    # 定义模型文件
|-- main.py                   # 项目主文件
|-- README.md                 # 中文用户手册
|-- LICENSE                   # LICENSE文件
```

### 5.2 代码简要说明

```undefined
./dataset/dataset.py       # 数据集代码文件
|-- class Dataset          # 数据集类
|   ├── get_feature        # 类主函数，返回可用于训练的数据集
|   ├── train_val_split    # 划分train&val数据集
|   ├── data_normalize     # 对特征进行归一化
|   ├── get_label_scaler   # 获得预测标签的归一化器

./tools/train.py           # 模型训练
|-- class Train            # 训练
|   ├── manual_seed        # 设置随机种子
|   ├── regression         # 回归函数

./tools/eval.py            # 模型验证
|-- class Eval             # 验证类
|   ├── evaluation         # 验证函数

./tools/averaging_model.py # 模型类
|-- class AveragingModels  # 模型融合类
|   ├── fit                # 拟合数据
|   ├── predict            # 预测数据
|   ├── save               # 保存模型
|   ├── load               # 加载模型
|-- class OptionalModel    # 可设置对标签是否进行log变换
|   ├── fit                # 拟合数据
|   ├── predict            # 预测数据
|-- class OptionalNnModels # 深度神经网络模型类
|   ├── fit                # 拟合数据
|   ├── predict            # 预测数据
|-- build_nn               # 构建深度神经网络模型

```
