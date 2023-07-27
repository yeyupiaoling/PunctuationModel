import argparse
import functools
import os

import paddle
from paddle import nn
from paddle.io import DataLoader
from sklearn.metrics import f1_score

from utils.logger import setup_logger
from utils.model import ErnieLinear
from utils.reader import PuncDatasetFromErnieTokenizer, collate_fn
from utils.utils import add_arguments, print_arguments

logger = setup_logger(__name__)


parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('batch_size',       int,    32,                       '评估的批量大小')
add_arg('max_seq_len',      int,    200,                      '评估数据的最大长度')
add_arg('num_workers',      int,    8,                        '读取数据的线程数量')
add_arg('test_data_path',   str,    'dataset/test.txt',       '测试数据的数据文件路径')
add_arg('punc_path',        str,    'dataset/punc_vocab',     '标点符号字典路径')
add_arg('model_path',       str,    'models/best_checkpoint', '加载检查点的目录')
add_arg('pretrained_token', str,    'ernie-3.0-medium-zh',    '使用的ERNIE模型权重')
args = parser.parse_args()
print_arguments(args)


def evaluate():
    logger.info('正在预处理数据集，时间比较长，请耐心等待...')
    test_dataset = PuncDatasetFromErnieTokenizer(data_path=args.test_data_path,
                                                 punc_path=args.punc_path,
                                                 pretrained_token=args.pretrained_token,
                                                 max_seq_len=args.max_seq_len)

    test_loader = DataLoader(test_dataset,
                             batch_size=args.batch_size,
                             shuffle=False,
                             drop_last=False,
                             collate_fn=collate_fn,
                             num_workers=args.num_workers)
    logger.info('预处理数据集完成！')

    # num_classes为字符分类大小
    model = ErnieLinear(pretrained_token=args.pretrained_token, num_classes=len(test_dataset.punc2id))
    criterion = nn.CrossEntropyLoss()
    model_dict = paddle.load(os.path.join(args.model_path, 'model.pdparams'))
    model.set_state_dict(model_dict)

    model.eval()
    eval_loss = []
    eval_f1_score = []
    for batch_id, (inputs, labels) in enumerate(test_loader()):
        labels = paddle.reshape(labels, shape=[-1])
        y, logit = model(inputs)
        pred = paddle.argmax(logit, axis=1)
        loss = criterion(y, labels)
        eval_loss.append(float(loss))
        F1_score = f1_score(labels.numpy().tolist(), pred.numpy().tolist(), average="macro")
        eval_f1_score.append(F1_score)
        if batch_id % 100 == 0:
            logger.info('Batch: [{}/{}], loss: {:.5f}, f1_score: {:.5f}'.format(
                batch_id, len(test_loader), float(loss), F1_score))
    logger.info('Avg eval, loss: {:.5f}, f1_score: {:.5f}'.format(
        sum(eval_loss) / len(eval_loss), sum(eval_f1_score) / len(eval_f1_score)))


if __name__ == "__main__":
    evaluate()
