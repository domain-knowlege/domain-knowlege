目前实验分为三个部分：

* 图像旋转(或affine)的恢复，主实验
* NLP实验，句子乱序的恢复
* 公式恢复，基于CFG

## 图像旋转
图像旋转的实验有三个数据集：cifar10，cifar100，tinyimagenet，
分别使用如下的模型：
* cifar10: DenseNet(depth=100)
    * Generalized-ODIN
        * 原正确率：94.61%
        * 旋转正确率 / 恢复正确率
            * 49.96% -> 75.98% (4-way)
            * 37.39% -> 78.75%
            * 39.40% -> 77.02%
    * softmax / Mahalanobis
        * 原正确率：95.19%
        * 旋转正确率：52.55%
        * softmax恢复正确率：65.30%
        * mahalanobis恢复正确率：60.16%
    * 重训练
        * Best accuracy:  88.63
    * fine-tuning, 1000张样本, 40epochs, lr=0.01
        * Best accuracy:  65.75
    * ETN
        * INFO:root:Test loss = 1.53e+00 INFO:root:Test error = 0.2191
        * 准确率：78.09%
* cifar100: ResNet34
    * Generalized-ODIN
        * 原正确率：74.78%
        * 旋转正确率 / 恢复正确率
            * 33.99% -> 48.35% (4-way)
            * 34.00% -> 49.83% (4-way)
            * 24.93% -> 51.72% (3-way)
    * softmax / Mahalanobis
        * 原正确率：77.11%
        * 旋转正确率：31.60%
        * softmax恢复正确率: 34.15%
        * mahalanobis恢复正确率：46.57%
    * 重训练
        * Best accuracy:  64.08
    * fine-tuning, 1000张样本, 40epochs, lr=0.01
        * Best accuracy:  34.99
    * ETN
        * INFO:root:Test loss = 4.55e+00 INFO:root:Test error = 0.5505
        * 准确率：44.95%
* TinyImagenet: WRN(depth=40)
    * Generalized-ODIN
        * 原正确率：59.94%
        * 旋转正确率 / 恢复正确率
            * 22.99% -> 43.05%
            * 22.79% -> 41.39%
    * softmax / Mahalanobis
        * 原正确率：63.67%
        * 旋转正确率：34.08%
        * softmax恢复正确率：36.12%
        * mahalanobis恢复正确率：38.30%
    * 重训练：
        * Best accuracy:  54.3
    * fine-tuning, 1000张样本, 40epochs, lr=0.01
        * Best accuracy:  27.06
    * ETN
        * INFO:root:Test loss = 3.26e+00 INFO:root:Test error = 0.6689
        * 准确率：33.11%

## 句子恢复
实验了两个场景，机器翻译和语句分类。其中机器翻译任务比较难，即使原句也难以达到翻译效果；打乱后的句子很难恢复，恢复的效果也不容易量化，因此没有被采用。
语句分类与图像旋转一样都是分类任务，比较容易量化。

### 语句分类
数据集TREC，包含了50个种类的问题。

* Maha
    * 原样本：test acc: 0.77 	| test loss: 1.1363135948181153
    * 打乱后：test acc: 0.462 	| test loss: 2.41964111328125
    * 恢复后：test acc: 0.63 	| test loss: 2.5580032043457033
* GMM
    * 原样本：test acc: 0.77 	| test loss: 1.1363135948181153
    * 打乱后：test acc: 0.43 	| test loss: 2.509507598876953
    * 恢复后：test acc: 0.644 	| test loss: 2.6578923950195312
* Confidence
    * 原样本：test acc: 0.77 	| test loss: 1.1363135948181153
    * 打乱后：test acc: 0.428 	| test loss: 2.811613525390625
    * 恢复后：test acc: 0.66 	| test loss: 2.8735596618652344

## 公式恢复
使用Hand Written Formula数据集，模型是两层conv+两层fc的LeNet网络。
每条数据包含若干图片(1-9,+-×÷)，模型试图识别这些图片并计算正确的结果。我们在正常识别的基础上，利用语法知识对结果进行修正，得到一个满足语法的概率最大的结果，以此提高识别和计算正确率。
测试集包含2000个样本。逐字符的识别正确率可达99%以上(就是mnist+四则运算符)，降低了原模型的性能。

* 原始结果：语法错误率2.90%，结果正确率83.65%，逐字符正确率96.79%
* 恢复结果：语法错误率0.00%，结果正确率85.15%，逐字符正确率97.11%
* baseline:
    * NGS: 语法错误率0.00%，结果正确率97.80%，逐字符正确率99.61% (mode: BS, num_epochs: 10, lr: 0.001)
