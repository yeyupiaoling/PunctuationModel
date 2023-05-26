import argparse
import functools
import os
import shutil

import paddle
from paddle.static import InputSpec

from utils.logger import setup_logger
from utils.model import ErnieLinearExport
from utils.utils import add_arguments, print_arguments

logger = setup_logger(__name__)


parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('punc_path',        str,    'dataset/punc_vocab',     '标点符号字典路径')
add_arg('model_path',       str,    'models/best_checkpoint', '加载检查点的目录')
add_arg('infer_model_path', str,    'models/pun_models',      '保存的预测的目录')
add_arg('pretrained_token', str,    'ernie-3.0-medium-zh',    '使用的ERNIE模型权重')
args = parser.parse_args()
print_arguments(args)


def main():
    os.makedirs(args.infer_model_path, exist_ok=True)
    with open(args.punc_path, 'r', encoding='utf-8') as f1, \
            open(os.path.join(args.infer_model_path, 'vocab.txt'), 'w', encoding='utf-8') as f2:
        lines = f1.readlines()
        lines = [line.replace('\n', '') for line in lines]
        # num_classes为字符分类大小，标点符号数量加1，因为开头还有空格
        num_classes = len(lines) + 1
        f2.write(' \n')
        for line in lines:
            f2.write(f'{line}\n')
    model = ErnieLinearExport(pretrained_token=args.pretrained_token, num_classes=num_classes)
    model_dict = paddle.load(os.path.join(args.model_path, 'model.pdparams'))
    model.set_state_dict(model_dict)

    input_spec = [InputSpec(shape=(-1, -1), dtype=paddle.int64), InputSpec(shape=(-1, -1), dtype=paddle.int64)]
    paddle.jit.save(layer=model, path=os.path.join(args.infer_model_path, 'model'), input_spec=input_spec)
    with open(os.path.join(args.infer_model_path, 'info.json'), 'w', encoding='utf-8') as f:
        f.write(str({'pretrained_token': args.pretrained_token}).replace("'", '"'))
    logger.info(f'模型导出成功，保存在：{args.infer_model_path}')


if __name__ == "__main__":
    main()
