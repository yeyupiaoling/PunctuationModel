import os
import random


def is_chinese(text):
    for c in text:
        if u'\u4e00' <= c <= u'\u9fa5':
            return True
    return False


def clear_text(src_dir, clear_file):
    os.makedirs(os.path.dirname(clear_file), exist_ok=True)
    results = []
    for f in os.listdir(src_dir):
        with open(os.path.join(src_dir, f), 'r', encoding='utf-8') as fp:
            lines = fp.readlines()
            for line in lines:
                line = line.replace('\n', '')
                line = line.replace('１', '')
                line = line.replace(' ', '')
                line = line.replace('!', '！')
                line = line.replace('.', '。')
                line = line.replace('?', '？')
                line = line.replace('“”', '')
                line = line.replace('’’', '')
                line = line.replace('-->>', '')
                line = line.replace('。。。', '。')
                line = line.replace('啊……', '啊！')
                line = line.replace('？！', '？')
                line = line.replace('！！！', '！')
                line = line.strip()
                if line.startswith('201') and line.endswith('发表'):
                    continue
                if line.startswith('（') and line.endswith('）'):
                    line = line[1:-1]
                if line.startswith('“') and line.endswith('”') and line.count("“") == 1:
                    line = line[1:-1]
                if line.count("“") == 1:
                    line = line.replace('“', '')
                if line.count("”") == 1:
                    line = line.replace('”', '')
                if line.count("”") == 1:
                    line = line.replace('”', '')
                if line.endswith('……'):
                    line = line.replace('……', '')
                    line = line + '。'
                if line.endswith('．．．．'):
                    line = line.replace('．．．．', '')
                    line = line + '。'
                if line.endswith('...'):
                    line = line.replace('...', '。')
                if line.endswith('…'):
                    line = line.replace('…', '')
                    line = line + '。'
                if line.endswith('(未完待续。)'):
                    line = line.replace('(未完待续。)', '')
                if len(line) < 3: continue
                if "兄弟（下）" in line: continue
                if "兄弟（上）" in line: continue
                if line[0] == "第" and "章" in line: continue
                if len(line) == 0 or not is_chinese(line): continue
                results.append(f'{line}')
    # 写在同一个文件上
    with open(clear_file, 'w', encoding='utf-8') as fp1:
        results = sorted(results, key=lambda x: len(x))
        for line in results:
            fp1.write(f'{line}\n')


def create_list(clear_file, save_dir, num_test=10000):
    with open(clear_file, 'r', encoding='utf-8') as fp:
        lines = fp.readlines()
    random.shuffle(lines)
    dev_text = lines[:num_test]
    test_text = lines[num_test:num_test + num_test]
    train_text = lines[num_test + num_test:]
    with open(os.path.join(save_dir, 'dev.txt'), 'w', encoding='utf-8') as fp1:
        for line in dev_text:
            line = line.replace('\n', '')
            fp1.write(f'{" ".join(line)} \n')
    with open(os.path.join(save_dir, 'test.txt'), 'w', encoding='utf-8') as fp1:
        for line in test_text:
            line = line.replace('\n', '')
            fp1.write(f'{" ".join(line)} \n')
    with open(os.path.join(save_dir, 'train.txt'), 'w', encoding='utf-8') as fp1:
        for line in train_text:
            line = line.replace('\n', '')
            fp1.write(f'{" ".join(line)} \n')


if __name__ == '__main__':
    clear_text(src_dir="dataset/files", clear_file='dataset/data.txt')
    create_list(clear_file='dataset/data.txt', save_dir='dataset')
