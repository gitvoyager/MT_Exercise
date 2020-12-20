# MT_Exercise
本报告分别实现了基于RNN和LSTM的神经网络语言模型，通过在同一语言模型数据集PTB上进行训练，并计算了不同训练次数的困惑度进行比较。

实验fork了Tae Hwan Jung (Jeff Jung) @graykode等人的项目（https://github.com/graykode/nlp-tutorial ），引用了该项目中基于RNN和LSTM实现的语言模型，在此基础上修改了batch的生成部分和训练的次数，增加了句子预处理部分和计算困惑度的函数，而神经网络模型部分并未作修改。
1、数据集
实验所使用的语言模型数据集是PTB数据集。
2、数据预处理部分
借鉴n-gram语言模型的思想，为了减少计算量，实验将模型简化为一个“四元组”的句子结构，即给定前三个单词序列，通过模型预测出第四个单词。
3、batch生成部分
由于训练集中的四元组句子数量比较大，故每次选取数量大小为1000的四元组句子生成one-hot张量进行训练或验证。
4、计算困惑度（PPL）部分
利用交叉熵损失计算困惑度，具体计算公式见报告。
5、训练与验证
实验将分别利用基于RNN和LSTM的语言模型，在PTB数据集的训练集上进行训练，一共训练六轮，每轮将句子打乱后重新训练。然后在验证集上计算困惑度，将每一轮训练后得到的困惑度值进行比较，评估两个语言模型的效果好坏。

实验结果表明，LSTM语言模型要优于RNN语言模型；模型的困惑度在训练开始时显著下降，随着训练次数的增加，下降变得不明显。
