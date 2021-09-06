'''
just do bpe using fastBPE
avoid <SEP-[]> and so on with format <x-[y](-[])>
'''

import sys
import re
import fastBPE

f = open(sys.argv[1])
bpefile = sys.argv[2]

bpe = fastBPE.fastBPE(bpefile)

for line in f:
    line = line.strip()

    # res = ['<SEP-[,]>', '<DropPro-[他]>', '<DropPunc-[,]>', '<Wrong-[幸]-[辛]>', '<SEP-[,]>', '<SEP-[,]>']
    sps = re.findall(r'<[SEP|DropPro|DropPunc|Wrong].*?>', line)
    segs = re.split(r'<[SEP|DropPro|DropPunc|Wrong].*?>', line)

    assert len(segs) == len(sps)+1

    segs_bpe = []
    for i in range(len(segs)-1):
        seg_bpe = bpe.apply([segs[i].strip()])[0]
        segs_bpe.append(seg_bpe)
        sp = sps[i]
        if sp[0:4] == '<SEP':
            sp = '<SEP>'
        segs_bpe.append(sp)
    
    segs_bpe.append(bpe.apply([segs[-1].strip()])[0])
    sys.stdout.write(' '.join(segs_bpe)+'\n')

