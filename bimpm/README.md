# 复现《Bilateral Multi-Perspective Matching for Natural Language Sentences》中的模型完成问答任务

此代码依照原作者的[开源代码](https://github.com/zhiguowang/BiMPM)复现并且作了简化（方便读者阅读）。不同点包括

* 无字向量（char embedding）
* 对输入的Q和A经过了上下文编码后再进入highway编码（原代码直接进入highway编码）
* 数据输入处理方式不同（原代码没给出wikiQA的处理方法）

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
