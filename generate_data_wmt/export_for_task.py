import re
import sys


'''
<InsertRandom-[with]>
<Delete-[&apos;]>
<InsertRepeat-[a|a|a]>
'''
def export_for_robust():
    # generate noisy zh
    f_zh = open(sys.argv[2])
    fw_zh = open(sys.argv[3], 'w')
    fw_pos = open(sys.argv[4], 'w')

    for line in f_zh:
        line = line.strip()
        tokens = line.split()

        sps = re.findall(r'<[Delete|InsertRepeat|InsertRandom].*?>', line)
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
                if ts[0] == 'Delete':
                    raw = ts[1][1:-1]
                    noise.append(f'{token}|{i}')
                elif ts[0] == 'InsertRepeat':
                    raw = ts[1][1:-1]
                    raw_tokens = raw.split('|')
                    temp.extend(raw_tokens)
                    noise.append(f'{token}|{i}to{i+len(raw_tokens)}')
                    i += len(raw_tokens)
                elif ts[0] == 'InsertRandom':
                    raw = ts[1][1:-1]
                    temp.append(raw)
                    noise.append(f'{token}|{i}')
                    i += 1
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