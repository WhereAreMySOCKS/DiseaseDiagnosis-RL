# DiseaseDiagnosis-RL
本项目使用强化学习方法，基于医疗知识图谱进行疾病推理。

1. 数据预处理 ： python data_processor.py
2. 知识图谱表示学习： python train_transe_model.py --dataset='Aier'
3. 训练强化学习模块： python train_agent.py --dataset='Aier'
4. 测试结果 ： python test_agent.py
