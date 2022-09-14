import argparse
import functools
import os
import time
from datetime import timedelta

import paddle
from paddle import nn
from paddle.io import DataLoader
from paddle.optimizer import Adam
from paddle.optimizer.lr import ExponentialDecay
from sklearn.metrics import f1_score

from utils.reader import PuncDatasetFromErnieTokenizer
from utils.model import ErnieLinear
from utils.utils import add_arguments, print_arguments
from utils.logger import setup_logger

logger = setup_logger(__name__)


parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('num_classes',      int,    4,                        '字符分类大小，标点符号数量加1，因为开头还有空格')
add_arg('batch_size',       int,    32,                       '训练的批量大小')
add_arg('num_workers',      int,    8,                        '读取数据的线程数量')
add_arg('num_epoch',        int,    20,                       '训练的轮数')
add_arg('learning_rate',    float,  1.0e-5,                   '初始学习率的大小')
add_arg('train_data_path',  str,    'dataset/train.txt',      '训练数据的数据文件路径')
add_arg('dev_data_path',    str,    'dataset/dev.txt',        '测试数据的数据文件路径')
add_arg('punc_path',        str,    'dataset/punc_vocab',     '标点符号字典路径')
add_arg('model_path',       str,    'models/checkpoint',      '保存检查点的目录')
add_arg('pretrained_token', str,    'ernie-3.0-medium-zh',
        '使用的ERNIE模型权重，具体查看：https://paddlenlp.readthedocs.io/zh/latest/model_zoo/transformers/ERNIE/contents.html#ernie')
args = parser.parse_args()
print_arguments(args)


def train():
    logger.info('正在预处理数据集，时间比较长，请耐心等待...')
    train_dataset = PuncDatasetFromErnieTokenizer(data_path=args.train_data_path,
                                                  punc_path=args.punc_path,
                                                  pretrained_token=args.pretrained_token,
                                                  seq_len=100)
    dev_dataset = PuncDatasetFromErnieTokenizer(data_path=args.dev_data_path,
                                                punc_path=args.punc_path,
                                                pretrained_token=args.pretrained_token,
                                                seq_len=100)
    train_loader = DataLoader(train_dataset,
                              shuffle=True,
                              num_workers=args.num_workers,
                              batch_size=args.batch_size)

    dev_loader = DataLoader(dev_dataset,
                            batch_size=args.batch_size,
                            shuffle=False,
                            drop_last=False,
                            num_workers=args.num_workers)
    logger.info('预处理数据集完成！')

    model = ErnieLinear(pretrained_token=args.pretrained_token, num_classes=args.num_classes)
    criterion = nn.CrossEntropyLoss()

    scheduler = ExponentialDecay(learning_rate=args.learning_rate, gamma=0.9999)
    optimizer = Adam(learning_rate=scheduler,
                     parameters=model.parameters(),
                     weight_decay=paddle.regularizer.L2Decay(1.0e-6))

    train_times = []
    sum_batch = len(train_loader) * args.num_epoch
    for epoch in range(args.num_epoch):
        epoch += 1
        start = time.time()
        for batch_id, (inputs, labels) in enumerate(train_loader()):
            labels = paddle.reshape(labels, shape=[-1])
            y, logit = model(inputs)
            pred = paddle.argmax(logit, axis=1)
            loss = criterion(y, labels)
            optimizer.clear_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            F1_score = f1_score(labels.numpy().tolist(), pred.numpy().tolist(), average="macro")
            train_times.append((time.time() - start) * 1000)
            # 多卡训练只使用一个进程打印
            if batch_id % 100 == 0:
                eta_sec = (sum(train_times) / len(train_times)) * (sum_batch - (epoch - 1) * len(train_loader) - batch_id)
                eta_str = str(timedelta(seconds=int(eta_sec / 1000)))
                logger.info(
                    'Train epoch: [{}/{}], batch: [{}/{}], loss: {:.5f}, f1_score: {:.5f}, learning rate: {:>.8f}, eta: {}'.format(
                        epoch, args.num_epoch, batch_id, len(train_loader), loss.numpy()[0], F1_score, scheduler.get_lr(), eta_str))
            start = time.time()
        model.eval()
        eval_loss = []
        eval_f1_score = []
        for batch_id, (inputs, labels) in enumerate(dev_loader()):
            labels = paddle.reshape(labels, shape=[-1])
            y, logit = model(inputs)
            pred = paddle.argmax(logit, axis=1)
            loss = criterion(y, labels)
            eval_loss.append(loss.numpy()[0])
            F1_score = f1_score(labels.numpy().tolist(), pred.numpy().tolist(), average="macro")
            eval_f1_score.append(F1_score)
            if batch_id % 100 == 0:
                logger.info('Batch: [{}/{}], loss: {:.5f}, f1_score: {:.5f}'.format(
                    batch_id, len(dev_loader), loss.numpy()[0], F1_score))
        logger.info('Avg eval, loss: {:.5f}, f1_score: {:.5f}'.format(
            sum(eval_loss) / len(eval_loss), sum(eval_f1_score) / len(eval_f1_score)))
        model.train()
        os.makedirs(args.model_path, exist_ok=True)
        paddle.save(model.state_dict(), os.path.join(args.model_path, 'model.pdparams'))
        paddle.save(optimizer.state_dict(), os.path.join(args.model_path, 'optimizer.pdopt'))
        logger.info(f'模型保存在：{args.model_path}')


if __name__ == "__main__":
    train()
