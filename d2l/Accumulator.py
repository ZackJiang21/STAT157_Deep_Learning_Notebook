class Accumulator(object):
    # Sum a list of numbers over time
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + b for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0] * len(self.data)

    def __getitem__(self, i):
        return self.data[i]
