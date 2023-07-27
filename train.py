import argparse
import functools
import os
import time
from datetime import timedelta
from paddle.distributed import fleet
import paddle
from paddle import nn
from paddle.io import DataLoader
from paddle.optimizer import Adam
from paddle.optimizer.lr import CosineAnnealingDecay
from sklearn.metrics import f1_score
from visualdl import LogWriter

from utils.reader import PuncDatasetFromErnieTokenizer, collate_fn
from utils.model import ErnieLinear
from utils.sampler import CustomBatchSampler, CustomDistributedBatchSampler
from utils.utils import add_arguments, print_arguments
from utils.logger import setup_logger

logger = setup_logger(__name__)


parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('batch_size',       int,    32,                       '训练的批量大小')
add_arg('max_seq_len',      int,    200,                      '训练数据的最大长度')
add_arg('num_workers',      int,    8,                        '读取数据的线程数量')
add_arg('num_epoch',        int,    30,                       '训练的轮数')
add_arg('learning_rate',    float,  1.0e-5,                   '初始学习率的大小')
add_arg('train_data_path',  str,    'dataset/train.txt',      '训练数据的数据文件路径')
add_arg('dev_data_path',    str,    'dataset/dev.txt',        '测试数据的数据文件路径')
add_arg('punc_path',        str,    'dataset/punc_vocab',     '标点符号字典路径')
add_arg('model_path',       str,    'models/',                '保存检查点的目录')
add_arg('resume_model',     str,    None,                     '恢复训练模型文件夹')
add_arg('pretrained_token', str,    'ernie-3.0-medium-zh',
        '使用的ERNIE模型权重，具体查看：https://paddlenlp.readthedocs.io/zh/latest/model_zoo/transformers/ERNIE/contents.html#ernie')
args = parser.parse_args()
print_arguments(args)


def train():
    paddle.set_device("gpu")
    # 获取有多少张显卡训练
    nranks = paddle.distributed.get_world_size()
    local_rank = paddle.distributed.get_rank()
    writer = None
    if local_rank == 0:
        # 日志记录器
        writer = LogWriter(logdir='log')
    # 支持多卡训练
    if nranks > 1:
        # 选择设置分布式策略
        strategy = fleet.DistributedStrategy()
        fleet.init(is_collective=True, strategy=strategy)

    train_dataset = PuncDatasetFromErnieTokenizer(data_path=args.train_data_path,
                                                  punc_path=args.punc_path,
                                                  pretrained_token=args.pretrained_token,
                                                  max_seq_len=args.max_seq_len)
    dev_dataset = PuncDatasetFromErnieTokenizer(data_path=args.dev_data_path,
                                                punc_path=args.punc_path,
                                                pretrained_token=args.pretrained_token,
                                                max_seq_len=args.max_seq_len)
    # 支持多卡训练
    if nranks > 1:
        train_batch_sampler = CustomDistributedBatchSampler(train_dataset,
                                                            batch_size=args.batch_size,
                                                            drop_last=True,
                                                            shuffle=True)
    else:
        train_batch_sampler = CustomBatchSampler(train_dataset,
                                                 batch_size=args.batch_size,
                                                 drop_last=True,
                                                 shuffle=True)
    train_loader = DataLoader(train_dataset,
                              collate_fn=collate_fn,
                              batch_sampler=train_batch_sampler,
                              num_workers=args.num_workers)

    dev_loader = DataLoader(dev_dataset,
                            batch_size=args.batch_size,
                            shuffle=False,
                            collate_fn=collate_fn,
                            drop_last=False,
                            num_workers=args.num_workers)
    logger.info('预处理数据集完成！')
    # num_classes为字符分类大小
    model = ErnieLinear(pretrained_token=args.pretrained_token, num_classes=len(train_dataset.punc2id))
    criterion = nn.CrossEntropyLoss()
    # 支持多卡训练
    if nranks > 1:
        model = fleet.distributed_model(model)

    scheduler = CosineAnnealingDecay(learning_rate=args.learning_rate, T_max=args.num_epoch)
    optimizer = Adam(learning_rate=scheduler,
                     parameters=model.parameters(),
                     weight_decay=paddle.regularizer.L2Decay(1.0e-5))
    # 支持多卡训练
    if nranks > 1:
        optimizer = fleet.distributed_optimizer(optimizer)
    # 恢复训练
    last_epoch = 0
    if args.resume_model:
        assert os.path.exists(os.path.join(args.resume_model, 'model.pdparams')), "模型参数文件不存在！"
        assert os.path.exists(os.path.join(args.resume_model, 'optimizer.pdopt')), "优化方法参数文件不存在！"
        model.set_state_dict(paddle.load(os.path.join(args.resume_model, 'model.pdparams')))
        opt_state = paddle.load(os.path.join(args.resume_model, 'optimizer.pdopt'))
        last_epoch = opt_state['LR_Scheduler']['last_epoch']
        optimizer.set_state_dict(opt_state)

    best_loss = 1e3
    train_step, test_step = 0, 0
    train_times = []
    sum_batch = len(train_loader) * args.num_epoch
    for epoch in range(last_epoch, args.num_epoch):
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
            F1_score = f1_score(labels.numpy().tolist(), pred.numpy().tolist(), average="macro")
            train_times.append((time.time() - start) * 1000)
            # 多卡训练只使用一个进程打印
            if batch_id % 100 == 0:
                eta_sec = (sum(train_times) / len(train_times)) * (sum_batch - (epoch - 1) * len(train_loader) - batch_id)
                eta_str = str(timedelta(seconds=int(eta_sec / 1000)))
                logger.info(
                    'Train epoch: [{}/{}], batch: [{}/{}], loss: {:.5f}, f1_score: {:.5f}, learning rate: {:>.8f}, eta: {}'.format(
                        epoch, args.num_epoch, batch_id, len(train_loader), float(loss), F1_score, scheduler.get_lr(), eta_str))
                if local_rank == 0:
                    writer.add_scalar('Train/Loss', float(loss), train_step)
                    writer.add_scalar('Train/F1_Score', F1_score, train_step)
                train_step += 1
            start = time.time()
        if local_rank == 0:
            writer.add_scalar('Train/LearnRate', scheduler.get_lr(), epoch)
        scheduler.step()
        model.eval()
        eval_loss = []
        eval_f1_score = []
        for batch_id, (inputs, labels) in enumerate(dev_loader()):
            labels = paddle.reshape(labels, shape=[-1])
            y, logit = model(inputs)
            pred = paddle.argmax(logit, axis=1)
            loss = criterion(y, labels)
            eval_loss.append(float(loss))
            F1_score = f1_score(labels.numpy().tolist(), pred.numpy().tolist(), average="macro")
            eval_f1_score.append(F1_score)
            if batch_id % 100 == 0:
                logger.info('Batch: [{}/{}], loss: {:.5f}, f1_score: {:.5f}'.format(
                    batch_id, len(dev_loader), float(loss), F1_score))
        eval_loss1 = sum(eval_loss) / len(eval_loss)
        eval_f1_score1 = sum(eval_f1_score) / len(eval_f1_score)
        if eval_loss1 < best_loss:
            best_loss = eval_loss1
            # 保存最优模型
            if local_rank == 0:
                save_dir = os.path.join(args.model_path, "best_checkpoint")
                os.makedirs(save_dir, exist_ok=True)
                paddle.save(model.state_dict(), os.path.join(save_dir, 'model.pdparams'))
                paddle.save(optimizer.state_dict(), os.path.join(save_dir, 'optimizer.pdopt'))
                logger.info(f'模型保存在：{save_dir}')
        logger.info('Avg eval, loss: {:.5f}, f1_score: {:.5f} best loss: {:.5f}'.
                    format(eval_loss1, eval_f1_score1, best_loss))
        model.train()
        if local_rank == 0:
            writer.add_scalar('Test/Loss', eval_loss1, test_step)
            writer.add_scalar('Test/F1_Score', eval_f1_score1, test_step)
            save_dir = os.path.join(args.model_path, "checkpoint")
            os.makedirs(save_dir, exist_ok=True)
            paddle.save(model.state_dict(), os.path.join(save_dir, 'model.pdparams'))
            paddle.save(optimizer.state_dict(), os.path.join(save_dir, 'optimizer.pdopt'))
            logger.info(f'模型保存在：{save_dir}')
            test_step += 1


if __name__ == "__main__":
    train()
