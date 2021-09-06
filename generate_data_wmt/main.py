import sys
import random

import fastBPE


bpe = fastBPE.fastBPE('exp_data_dialogue/bpe.zh')

f = open(sys.argv[1])
choices = ['the', ',', '.', 'of', 'and', 'to', 'in', 'a', 'is', 'that',
    'for', 'on', 'with', 'be', 'are', 'The', 'I', 'as', 'this', 'it', 'we']

for src in f:
    src = src.strip()
    src_tokens = [token for token in src.split() if len(token.strip())>0]

    # pro
    temp = []
    prev_bpe = False
    present_bpe = False
    for token in src_tokens:
        # ignore bpe words
        if token.endswith('@@'):
            present_bpe = True
        else:
            present_bpe = False 

        # random delete
        if random.random() < 0.02 and not present_bpe and not prev_bpe and '-' not in token:
            symbol = f'<Delete-[{token}]>'
            temp.append(symbol)
        # random insert
        elif random.random() < 0.02 and not present_bpe and not prev_bpe:
            # repeat 1-3 times
            if random.random() <= 0.7 and '-' not in token:
                temp.append(token)
                t = random.randint(1, 3)
                t_tokens = [token]*t
                symbol = f"<InsertRepeat-[{'|'.join(t_tokens)}]>"
            # random insert
            else:
                temp.append(token)
                symbol = f'<InsertRandom-[{random.choice(choices)}]>'
            temp.append(symbol)
        else:
            temp.append(token)
        
        if token.endswith('@@'):
            prev_bpe = True
        else:
            prev_bpe = False 

    sys.stdout.write(f"{' '.join(temp)}\n")

f.close()
