import jieba
import random
import re
import fastBPE

zh_bpe = fastBPE.fastBPE('synthetic/bpe.zh')

allowed_en = set("my i me we us you your yourself they their them he him his she her".split())
allowed_zh = set("我的 我 我们 我们的 你 你们 你自己 他们 他们的 他 他的 她 她的".split())
f_zh = open('train/dev.zh')
# names = ['train.dp_3.src', 'train.dp_3.trg', 'train.dq.src', 'train.dq.trg', 'train.dpq.src', 'train.dpq.trg']
names = ['dev.dp_5.src', 'dev.dp_5.trg', 'dev.dq.src', 'dev.dq.trg', 'dev.dpq.src', 'dev.dpq.trg']

for name in names:
    locals()[name] = open('synthetic/dp5/'+name, 'w')
    bpe_name = name.replace('src', 'bpe.src').replace('trg', 'bpe.trg')
    locals()[bpe_name] = open('synthetic/dp5/'+bpe_name, 'w')


def rep_punc(matched):
    return matched.group()[0].strip()

for zh in f_zh:
    zh = zh.strip()
    zh = zh.replace(',', '，').replace('?', '？')
    zh = re.sub(r'[。，？]\s+', rep_punc, zh)

    zh = list(jieba.cut(zh))
    dq_src = ' '.join([word for word in zh if len(word.strip())>0])
    dq_trg = ' '.join([word if word != ' ' else '，' for word in zh])
    temp = []
    for word in zh:
        if word in allowed_zh:
            if random.random() > 0.5:
                temp.append(word)
        else:
            if len(word.strip())>0:
                temp.append(word)
    dp_src = ' '.join(temp)

    dp_trg = dq_src
    dpq_src = dp_src
    dpq_trg = dq_trg

    towrite = [dp_src, dp_trg, dq_src, dq_trg, dpq_src, dpq_trg]
    bpe_towrite = [zh_bpe.apply([s])[0] for s in towrite]

    for i, name in enumerate(names):
        bpe_name = name.replace('src', 'bpe.src').replace('trg', 'bpe.trg')
        locals()[name].write(towrite[i]+'\n')
        locals()[bpe_name].write(bpe_towrite[i]+'\n')