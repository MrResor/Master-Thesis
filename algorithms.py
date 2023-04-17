from __init__ import np


class Genetic:
    def __init__(self, params: dict) -> None:
        print('Genetic', params)


class Ant:
    def __init__(self, params: dict) -> None:
        self.tours, self.alpha, self.beta, self.rho = params.values()

    def run(self, d, size) -> None:
        self.d = d
        self.size = size
        best = self.ants_tables()
        for trip in range(self.tours):
            print('Trip', trip)
            self.ant_and_a()
            self.ants_traveling()
            # TODO there has to be a better way
            min = np.Inf
            index = 0
            for i in range(self.size):
                if self.ant[i][self.size] < min:
                    index = i
                    min = self.ant[i][self.size]
            if best[self.size] > self.ant[index][self.size]:
                best = self.ant[index]
            self.pheromones()
        print(best)

    def ants_tables(self) -> np.ndarray:
        # fucntion for creating necessary tables
        self.tau = np.full((self.size, self.size), 1/self.d.max())
        self.a = np.full((self.size, self.size), 0.0)
        self.n = 1/self.d
        np.fill_diagonal(self.n, 0)
        best = np.full(self.size + 1, 0.0)
        best[self.size] = np.Inf
        return best

    def ant_and_a(self) -> None:
        # creating anew ant table and filling a
        self.ant = np.full((self.size, self.size + 1),
                           self.size + 1, dtype="object")
        self.ant[:, self.size] = float(0)
        self.a = np.power(self.tau, self.alpha) * np.power(self.n, self.beta)
        self.start = np.random.randint(0, self.size, self.size)
        self.a = [val / sum(val) for val in self.a]

    def ants_traveling(self) -> None:
        # function that simulates "travelling" of ants
        self.ant[:, 0] = self.start
        for i in range(self.size):
            for j in range(1, self.size):
                self.p = [self.a[self.ant[i, j-1]][q] *
                          (q not in self.ant[i]) for q in range(self.size)]
                self.p = [val / sum(self.p) for val in self.p]
                r = 0
                while r == 0:
                    r = np.random.rand(1)
                for q in range(self.size):
                    r -= self.p[q]
                    if r <= 0:
                        self.ant[i][j] = q
                        self.ant[i][self.size] += self.d[
                            self.ant[i][j - 1]][self.ant[i][j]]
                        break
            self.ant[i][self.size] += self.d[
                self.ant[i][self.size-1]][self.ant[i][0]]

    def pheromones(self) -> None:
        # function for updating pheromones on each arc
        self.tau *= (1-self.rho)
        # TODO check if np.roll can do this faster
        for i in range(self.size):
            for j in range(1, self.size):
                self.tau[self.ant[i][j - 1]][
                    self.ant[i][j]] += 1/self.ant[i][self.size]
                self.tau[self.ant[i][j]][
                    self.ant[i][j - 1]] += 1/self.ant[i][self.size]
            self.tau[self.ant[i][0]][
                self.ant[i][self.size - 1]] += 1/self.ant[i][self.size]
            self.tau[self.ant[i][self.size - 1]][
                self.ant[i][0]] += 1/self.ant[i][self.size]
