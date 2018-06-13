# 复现《A Decomposable Attention Model for Natural Language Inference》中的模型完成问答任务

## 准备

#### 下载词向量文件[glove](../download.sh)。

```
cd ..
bash download.sh
```

#### 预处理wiki数据

```
cd ..
python preprocess_wiki.py
```

## 运行

```
bash run.sh
```
