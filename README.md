# 任务介绍

基于文本图像的多模态图像识别

## 环境
tqdm==4.62.3
torchvision==0.9.1+cu111
numpy==1.19.5
torch==1.8.1+cu111
transformers==4.15.0
Pillow==9.2.0
scikit_learn==1.1.1


## 训数据预处理
对train数据进行shuffle后按8:2比例切分成训练集和验证集，图片和文本分别处理


## 模型介绍
本实验采用基于word2vec+LSTM+ResNet50的多模态模型，对于文本部分采用glove预训练的英文word2vec词向量加上bilstm模型进行特征抽取，对于图像部分采用Resnet50模型进行特征抽取，将两种模态特征进行concat操作后输入MLP进行结果分类

## 训练和预测
超参数在`utils/config.py`处设置，训练结束后效果最好的模型保存在`saved_dict`文件夹下，生成的预测文件在`data/实验五数据/test_with_label.txt`路径下
> python run.py 

### 模型保存
模型保存在saved_dict文件夹，训练过程和模型构造log文件夹

