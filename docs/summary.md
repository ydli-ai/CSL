## 文本摘要任务

将 BERT 模型转换为 UNILM 进行 fine-tuning 实现文本生成。

### 运行方法

1. 下载并解压 BERT 预训练模型及训练数据 

[chinese_L-12_H-768_A-12.zip](https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip)

abstract_cs.zip [百度网盘](https://pan.baidu.com/s/1nYM1rMHLNW0o7gGwV_dEiA)  (提取码: s967)
2. 安装环境

依赖 bert4keras v0.3.6 如果已经安装其他版本 bert4keras 请先执行 pip uninstall bert4keras 卸载。

执行 `pip install git+https://www.github.com/bojone/bert4keras.git@v0.3.6`

3. 修改代码中相关路径并运行

[代码](../examples/summary_baseline.py)