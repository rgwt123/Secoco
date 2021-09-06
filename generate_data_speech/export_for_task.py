import re
import sys


'''
<DropPunc-[。]>
<ReplacePunc-[，|。]>
<InsertSpoken-[其实]>
'''
def export_for_robust():
    # generate noisy zh
    f_zh = open(sys.argv[2])
    fw_zh = open(sys.argv[3], 'w')
    fw_pos = open(sys.argv[4], 'w')

    for line in f_zh:
        line = line.strip()
        tokens = line.split()

        sps = re.findall(r'<[DropPunc|ReplacePunc|InsertSpoken].*?>', line)
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
                if ts[0] == 'DropPunc':
                    raw = ts[1][1:-1]
                    noise.append(f'{token}|{i}')
                elif ts[0] == 'ReplacePunc':
                    raw = ts[1][1:-1]
                    _, raw_new = raw.split('|')
                    temp.append(raw_new)
                    noise.append(f'{token}|{i}')
                    i += 1
                elif ts[0] == 'InsertSpoken':
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
    f_zh.close()
    fw_pos.close()
    fw_zh.close()


def work(s):
    if s == 'noise':
        export_for_robust()


if __name__ == '__main__':
    work(sys.argv[1])