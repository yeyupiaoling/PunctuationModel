# 中文标点符号模型

基于Ernie开发的中文标点符号模型，默认使用的预训练模型为`ernie-3.0-medium-zh`，该模型可以用于语音识别结果添加标点符号，使用案例[PPASR](https://github.com/yeyupiaoling/PPASR)。


# 安装环境
 1. 安装PaddlePaddle的GPU版本，命令如下，如果已经安装过了，请忽略。
```shell
conda install paddlepaddle-gpu==2.3.2 cudatoolkit=10.2 --channel https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/Paddle/
```

2. 安装PaddleNLP工具，命令如下。
```shell
python -m pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/
```

# 准备数据

项目提供了一个较小的数据集可以直接使用。如果想自定义数据集，可以参考这个数据集的格式进行制作。注意在制作标点符号列表`punc_vocab`时，不需要加上空格，项目默认会加上空格的。这里提供一个制作方式，首先可以在[这里](https://www.qishuta.info/)下载小说的TXT文件，将它们存放在`dataset/files`目录下，然后执行`clear_data.py`程序清洗和分割数据集，注意这个清洗并不是万能的，要更加自己下载的数据，修改清洗方式。

如何想训练更多的标点符号，可以在`punc_vocab`增加，`punc_vocab`默认只有`，。？`三个标点符号，注意在制作标点符号列表`punc_vocab`时，不需要加上空格，项目默认会加上空格的。

```
├── dataset
  ├── dev.txt
  ├── punc_vocab
  ├── test.txt
  └── train.txt
```

# 训练

准备好数据集之后，就可以执行`train.py`开始训练，如果是自定义数据集，在开始训练之前，要注意修改类别数量参数`num_classes`，执行命令如下，第一次训练时会下载ernie预训练模型，所以需要联网。
```shell
# 单机单卡
python train.py
# 单价多卡
python -m paddle.distributed.launch --devices=0,1 train.py
```

训练输出的日志：
```
-----------  Configuration Arguments -----------
batch_size: 32
dev_data_path: dataset/dev.txt
learning_rate: 1e-05
model_path: models/checkpoint
num_classes: 4
num_epoch: 20
num_workers: 8
pretrained_token: ernie-3.0-medium-zh
punc_path: dataset/punc_vocab
train_data_path: dataset/train.txt
------------------------------------------------
[2022-09-14 17:11:48.482046 INFO   ] train:train:39 - 正在预处理数据集，时间比较长，请耐心等待...
[2022-09-14 17:11:48,482] [    INFO] - Already cached /home/test/.paddlenlp/models/ernie-3.0-medium-zh/ernie_3.0_medium_zh_vocab.txt
[2022-09-14 17:11:48,507] [    INFO] - tokenizer config file saved in /home/test/.paddlenlp/models/ernie-3.0-medium-zh/tokenizer_config.json
[2022-09-14 17:11:48,508] [    INFO] - Special tokens file saved in /home/test/.paddlenlp/models/ernie-3.0-medium-zh/special_tokens_map.json
100%|█████████████████████████████████████████████████████████████| 4328594/4328594 [05:42<00:00, 12645.37it/s]
[2022-09-14 17:17:31,589] [    INFO] - Already cached /home/test/.paddlenlp/models/ernie-3.0-medium-zh/ernie_3.0_medium_zh_vocab.txt
[2022-09-14 17:17:31,610] [    INFO] - tokenizer config file saved in /home/test/.paddlenlp/models/ernie-3.0-medium-zh/tokenizer_config.json
[2022-09-14 17:17:31,610] [    INFO] - Special tokens file saved in /home/test/.paddlenlp/models/ernie-3.0-medium-zh/special_tokens_map.json
100%|██████████████████████████████████████████████████████████████| 33741/33741 [00:02<00:00, 12532.24it/s]
[2022-09-14 17:17:34.309391 INFO   ] train:train:58 - 预处理数据集完成！
[2022-09-14 17:17:34,309] [    INFO] - Already cached /home/test/.paddlenlp/models/ernie-3.0-medium-zh/ernie_3.0_medium_zh.pdparams
W0914 17:17:34.310540 10320 device_context.cc:447] Please NOTE: device: 0, GPU Compute Capability: 7.5, Driver API Version: 11.6, Runtime API Version: 10.2
W0914 17:17:34.313140 10320 device_context.cc:465] device: 0, cuDNN Version: 7.6.
[2022-09-14 17:17:37.758967 INFO   ] train:train:90 - Train epoch: [1/20], batch: [0/1283], loss: 2.05675, f1_score: 0.02082, learning rate: 0.00001000, eta: 2:18:40
[2022-09-14 17:17:54.295418 INFO   ] train:train:90 - Train epoch: [1/20], batch: [100/1283], loss: 0.12979, f1_score: 0.33040, learning rate: 0.00000990, eta: 1:11:06
[2022-09-14 17:18:10.936073 INFO   ] train:train:90 - Train epoch: [1/20], batch: [200/1283], loss: 0.13771, f1_score: 0.37442, learning rate: 0.00000980, eta: 1:10:43
[2022-09-14 17:18:27.706051 INFO   ] train:train:90 - Train epoch: [1/20], batch: [300/1283], loss: 0.10602, f1_score: 0.47096, learning rate: 0.00000970, eta: 1:10:35
[2022-09-14 17:18:44.545404 INFO   ] train:train:90 - Train epoch: [1/20], batch: [400/1283], loss: 0.12836, f1_score: 0.55652, learning rate: 0.00000961, eta: 1:10:27
[2022-09-14 17:19:01.434206 INFO   ] train:train:90 - Train epoch: [1/20], batch: [500/1283], loss: 0.11024, f1_score: 0.51312, learning rate: 0.00000951, eta: 1:10:18
```


# 评估

训练结束之后，可以进行评估模型，观察模型的收敛情况，执行命令如下。
```shell
python eval.py
```

输出的日志信息：
```
-----------  Configuration Arguments -----------
batch_size: 32
model_path: models/checkpoint
num_classes: 4
num_workers: 8
pretrained_token: ernie-3.0-medium-zh
punc_path: dataset/punc_vocab
test_data_path: dataset/test.txt
------------------------------------------------
[2022-09-14 19:17:54.851788 INFO   ] eval:evaluate:32 - 正在预处理数据集，时间比较长，请耐心等待...
[2022-09-14 19:17:54,851] [    INFO] - Already cached /home/test/.paddlenlp/models/ernie-3.0-medium-zh/ernie_3.0_medium_zh_vocab.txt
[2022-09-14 19:17:54,877] [    INFO] - tokenizer config file saved in /home/test/.paddlenlp/models/ernie-3.0-medium-zh/tokenizer_config.json
[2022-09-14 19:17:54,877] [    INFO] - Special tokens file saved in /home/test/.paddlenlp/models/ernie-3.0-medium-zh/special_tokens_map.json
100%|████████████████████████████████████████████████████████████████████████████████████| 43468/43468 [00:03<00:00, 12605.40it/s]
[2022-09-14 19:17:58.336113 INFO   ] eval:evaluate:43 - 预处理数据集完成！
[2022-09-14 19:17:58,336] [    INFO] - Already cached /home/test/.paddlenlp/models/ernie-3.0-medium-zh/ernie_3.0_medium_zh.pdparams
W0914 19:17:58.337256 11985 device_context.cc:447] Please NOTE: device: 0, GPU Compute Capability: 7.5, Driver API Version: 11.6, Runtime API Version: 10.2
W0914 19:17:58.339792 11985 device_context.cc:465] device: 0, cuDNN Version: 7.6.
[2022-09-14 19:18:02.054659 INFO   ] eval:evaluate:63 - Batch: [0/13], loss: 0.08727, f1_score: 0.78612
[2022-09-14 19:18:02.775505 INFO   ] eval:evaluate:65 - Avg eval, loss: 0.12825, f1_score: 0.70011
```

# 导出预测模型

要执行模型之前，需要导出预测模型方能使用，执行下面命令导出预测模型，导出的模型文件默认会保存在`models/pun_models`，[PPASR](https://github.com/yeyupiaoling/PPASR)就需要把这整个文件夹复制到`models`目录下。
```shell
python export_model.py
```

输出的日志信息：
```
-----------  Configuration Arguments -----------
infer_model_path: models/pun_models
model_path: models/checkpoint
num_classes: 4
pretrained_token: ernie-3.0-medium-zh
punc_path: dataset/punc_vocab
------------------------------------------------
[2022-09-14 19:20:42,188] [    INFO] - Already cached /home/test/.paddlenlp/models/ernie-3.0-medium-zh/ernie_3.0_medium_zh.pdparams
W0914 19:20:42.189301 12045 device_context.cc:447] Please NOTE: device: 0, GPU Compute Capability: 7.5, Driver API Version: 11.6, Runtime API Version: 10.2
W0914 19:20:42.192952 12045 device_context.cc:465] device: 0, cuDNN Version: 7.6.
[2022-09-14 19:20:49.433919 INFO   ] export_model:main:43 - 模型导出成功，保存在：models/pun_models
```


# 给文本添加标点符号

使用导出的预测模型为文本添加标点符号，也可以下载博主提供的模型[三个标点符号](https://download.csdn.net/download/qq_33200967/86539773)和[五个标点符号](https://download.csdn.net/download/qq_33200967/75664996)，解压到`dataset`目录下，通过`text`参数指定中文文本，实现添加标点符号，这可以应用在语音识别结果上面，具体可以参考[PPASR](https://github.com/yeyupiaoling/PPASR)语音识别项目。
```shell
python infer.py --text=近几年不但我用书给女儿儿压岁也劝说亲朋不要给女儿压岁钱而改送压岁书
```

输出日志信息：
```
-----------  Configuration Arguments -----------
infer_model_path: models/pun_models
text: 近几年不但我用书给女儿儿压岁也劝说亲朋不要给女儿压岁钱而改送压岁书
------------------------------------------------
[2022-09-14 19:23:48,566] [    INFO] - Already cached /home/test/.paddlenlp/models/ernie-3.0-medium-zh/ernie_3.0_medium_zh_vocab.txt
[2022-09-14 19:23:48,590] [    INFO] - tokenizer config file saved in /home/test/.paddlenlp/models/ernie-3.0-medium-zh/tokenizer_config.json
[2022-09-14 19:23:48,591] [    INFO] - Special tokens file saved in /home/test/.paddlenlp/models/ernie-3.0-medium-zh/special_tokens_map.json
[2022-09-14 19:23:49.960468 INFO   ] predictor:__init__:60 - 标点符号模型加载成功。
近几年，不但我用书给女儿儿压岁，也劝说亲朋不要给女儿压岁钱而改送压岁书。
```