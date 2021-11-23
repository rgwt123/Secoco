import sys

import numpy as np


def _wer(ref, hypo):
    d = np.zeros((len(ref) + 1) * (len(hypo) + 1), dtype=np.uint8).reshape((len(ref) + 1, len(hypo) + 1))
    for i in range(len(ref) + 1):
        d[i][0] = i
    for j in range(len(hypo) + 1):
        d[0][j] = j
    for i in range(1, len(ref) + 1):
        for j in range(1, len(hypo) + 1):
            if ref[i - 1] == hypo[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                substitute = d[i - 1][j - 1] + 1
                insert = d[i][j - 1] + 1
                delete = d[i - 1][j] + 1
                d[i][j] = min(substitute, insert, delete)
    return d


def wer_sentence(ref: list, hypo: list, use_cpp=False):
    if type(hypo) == str and type(ref) == str:
        ref, hypo = ref.strip().split, hypo.strip().split()
    result = _wer(ref, hypo)
    return result


def get_step_list(ref, hypo, result):
    '''
    This function is to get the list of steps in the process of dynamic programming.
    Attributes:
        ref -> the list of words produced by splitting reference sentence.
        hypo -> the list of words produced by splitting hypothesis sentence.
        result -> the matrix built when calulating the editting distance of h and r.
    '''
    id_ref, id_hypo = len(ref), len(hypo)
    steps = []
    while True:
        if id_ref == 0 and id_hypo == 0:
            break
        elif id_ref >= 1 and id_hypo >= 1 and \
                result[id_ref][id_hypo] == result[id_ref - 1][id_hypo - 1] and \
                ref[id_ref - 1] == hypo[id_hypo - 1]:
            steps.append("e")
            id_ref -= 1
            id_hypo -= 1
        elif id_hypo >= 1 and result[id_ref][id_hypo] == result[id_ref][id_hypo - 1] + 1:
            steps.append(f"d-{hypo[id_hypo - 1]}")
            id_hypo -= 1
        elif id_ref >= 1 and id_hypo >= 1 and \
                result[id_ref][id_hypo] == result[id_ref - 1][id_hypo - 1] + 1:
            steps.append(f"s-{hypo[id_hypo - 1]}||{ref[id_ref - 1]}")
            id_ref -= 1
            id_hypo -= 1
        else:
            steps.append(f"i-{ref[id_ref - 1]}")
            id_ref -= 1
    return steps[::-1]


if __name__ == '__main__':
    with open(sys.argv[1]) as fref:
        for line in sys.stdin:
            ref = fref.readline().strip()
            line = line.strip()
            ref_tokens = ref.split()
            raw_tokens = line.split()
            steps = get_step_list(ref_tokens, raw_tokens, wer_sentence(ref_tokens, raw_tokens))
            print(steps)
