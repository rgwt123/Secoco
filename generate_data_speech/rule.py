import sys
import random


f = open(sys.argv[1])
choices = ['这个','那个','就是','然后','其实','的话']

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
        # punctunation
        if token in ['，', '。'] and random.random() < 0.3:
            # random drop
            if random.random() < 0.6:
                symbol = f'<DropPunc-[{token}]>'
                temp.append(symbol)
            # replace
            else:
                if token == '，':
                    rep = '。'
                else:
                    rep = '，'
                symbol = f'<ReplacePunc-[{token}|{rep}]>'
                temp.append(symbol)
        # spoken words
        elif random.random() < 0.02 and not present_bpe and not prev_bpe:
            temp.append(token)
            symbol = f"<InsertSpoken-[{random.choice(choices)}]>"
            temp.append(symbol)
        else:
            temp.append(token)
        
        if token.endswith('@@'):
            prev_bpe = True
        else:
            prev_bpe = False 

    sys.stdout.write(f"{' '.join(temp)}\n")

f.close()