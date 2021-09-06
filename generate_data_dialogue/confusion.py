import random

class Confusion(object):
    def __init__(self):
        self.tables = {}
        name = 'generate_data_dialogue/confusion.filter.txt'
        with open(name) as f:
            for line in f:
                k, vs = line.strip().split(':')
                self.tables[k] = vs
        self.tables_keys = list(self.tables.keys())
    
    def getwords(self, key):
        return self.tables[key]

    def getword(self, key):
        return self.tables[key][random.randint(0, len(self.tables[key])-1)]

    def getrandword(self):
        return self.tables_keys[random.randint(0, len(self.tables)-1)]
    
    def changeword(self, x):
        x_len = len(x)
        try_times = 0
        while try_times < x_len:
            try_times += 1
            try_word = x[random.randint(0, len(x)-1)]
            if try_word in self.tables:
                rep_word = self.getword(try_word)
                return x.replace(try_word, rep_word)

        return x

    def changerandword(self, x):
        x_len = len(x)
        try_times = 0
        while try_times < x_len:
            try_times += 1
            try_word = x[random.randint(0, len(x)-1)]
            if try_word in self.tables:
                rep_word = self.getrandword()
                return x.replace(try_word, rep_word)

        return x

    def changeword_28(self, x):
        if random.random()<0.2:
            return self.changerandword(x)
        else:
            return self.changeword(x)

if __name__ == '__main__':
    sent = '这是 一个 安静 的 早上 。'
    words = sent.split()
    confusion = Confusion()
    for word in words:
        ret = confusion.changeword_28(word)
        if ret != word:
            print(ret)
