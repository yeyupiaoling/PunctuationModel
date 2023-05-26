import argparse
import functools

from utils.predictor import PunctuationExecutor
from utils.utils import add_arguments, print_arguments

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('text',             str,    '近几年不但我用书给女儿压岁也劝说亲朋不要给女儿压岁钱而改送压岁书',      '需要加标点符号的文本')
add_arg('infer_model_path', str,    'models/pun_models',      '预测的目录')
args = parser.parse_args()
print_arguments(args)


if __name__ == '__main__':
    pun_executor = PunctuationExecutor(model_dir=args.infer_model_path, use_gpu=True)
    result = pun_executor(args.text)
    print(result)
