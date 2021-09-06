import re
import sys
import random


def export_for_robust():
    # generate noisy zh
    f_zh = open(sys.argv[2])
    fw_zh = open(sys.argv[3], 'w')
    fw_pos = open(sys.argv[4], 'w')

    for line in f_zh:
        line = line.strip()
        tokens = line.split()

        sps = re.findall(r'<[DropPro|DropPunc|Wrong].*?>', line)
        idx = 0
        i = 0
        temp = []
        noise = []
        for token in tokens:
            if idx >= len(sps):
                temp.append(token)
                i += 1
            elif token == sps[idx]:
                idx += 1
                ts = token[1:-1].split('-')
                if ts[0] == 'DropPro':
                    # process droppro
                    raw = ts[1][1:-1]
                    if random.random() < 0.7:
                        temp.append(raw)
                        i += 1
                    else:
                        # drop 50%
                        noise.append(f'{token}|{i}')
                elif ts[0] == 'DropPunc':
                    # process droppunc
                    raw = ts[1][1:-1]
                    if random.random() < 0.7:
                        temp.append(raw)
                        i += 1
                    else:
                        # drop 50%
                        noise.append(f'{token}|{i}')
                elif ts[0] == 'Wrong':
                    # do random before, so keep all wrong words here
                    raw, new, pos = ts[1][1:-1], ts[2][1:-1], ts[3][1:-1]
                    new_chars = new.split('|')
                    for new_char in new_chars:
                        temp.append(new_char)
                    noise.append(f'{token}|{i+int(pos)}')
                    i += len(new_chars)
                else:
                    print('error!!!!!')
                    exit(0)

            else:
                temp.append(token)
                i += 1
            
        fw_zh.write(' '.join(temp) + '\n')
        fw_pos.write(' '.join(noise) + '\n')


def work(s):
    if s == 'noise':
        export_for_robust()

if __name__ == '__main__':
    work(sys.argv[1])