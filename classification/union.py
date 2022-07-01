class DisjunctUnion:
    def __init__(self, nodes):
        self.array = [n for n in range(nodes)]
        self.weight = [n for n in range(nodes)]

    def root(self, parent):
        while parent != self.array[parent]:
            parent = self.array[parent]
        return parent

    def union(self, n1, n2):
        r1, r2 = self.root(n1), self.root(n2)

        if r1 == r2:
            return

        elif self.weight[r1] < self.weight[r2]:
            self.array[r1] = r2
            self.weight[n2] += self.weight[r1]

        else:
            self.array[r2] = r1
            self.weight[n1] += self.weight[r2]

    def connected(self, n1, n2):
        return self.root(n1) == self.root(n2)

    def components(self):
        components_dict = {}

        for i in range(len(self.array)):
            r = self.root(self.array[i])
            if r in components_dict.keys():
                components_dict[r].append(i)
            else:
                components_dict[r] = [i]

        return components_dict


