import sys
import random

import fastBPE

from confusion import Confusion

SUBS = set("我的 我 我们 我们的 你 你们 你自己 他们 他们的 他 他的 她 她的".split())
PUNCS = set("， 。 ？".split())
confusion = Confusion()
bpe = fastBPE.fastBPE('chat_translation_dialogue/exp_data/bpe.zh')

f = open(sys.argv[1])

for zh in f:
    zh = zh.strip()
    # zh_tokens = [token for token in jieba.cut(zh) if len(token.strip())>0]
    zh_tokens = [token for token in zh.split() if len(token.strip())>0]

    # pro
    temp = []
    for token in zh_tokens:
        # pros
        if token in SUBS:
            symbol = f'<DropPro-[{token}]>'
            temp.append(symbol)
        # puncs
        elif token in PUNCS:
            symbol = f'<DropPunc-[{token}]>'
            temp.append(symbol)
        # generate some wrong words
        elif token == '<SEP>':
            temp.append(token)
        elif random.random() < 0.01 and not token.endswith('@@'):
            rep = confusion.changeword_28(token)
            i = 0
            if rep != token:
                rep_bpe = bpe.apply([rep])[0]
                # 他们 -》 他@@ 门
                rep_bpes = rep_bpe.split()
                if len(rep_bpes) > 1:
                    rep_bpes_len = len(rep_bpes)
                    j = 0
                    for i in range(rep_bpes_len):
                        rep_bpe_char = rep_bpes[i].replace('@@', '')
                        rep_bpe_char_len = len(rep_bpe_char)
                        if rep_bpe_char != token[j:j+rep_bpe_char_len]:
                            break
                        j += rep_bpe_char_len

                symbol = f"<Wrong-[{token}]-[{'|'.join(rep_bpes)}]-[{i}]>"
            else:
                symbol = token
            temp.append(symbol)
        else:
            temp.append(token)

    sys.stdout.write(f"{' '.join(temp)}\n")

f.close()