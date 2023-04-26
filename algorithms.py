from __init__ import np, logging
import time


class Genetic:
    def __init__(self, params: dict) -> None:
        self.P, self.n, self.p, self.T = params.values()

    def run(self, d, size) -> None:

        start = time.perf_counter()
        logging.info("Staring algorithm",
                     extra={'runtime': time.perf_counter() - start})
        self.d = d
        self.size = size
        # Prepaire container for best and parents
        best = np.arange(self.size + 1, dtype='object')
        best[-1:] = np.Inf
        parents = np.zeros((self.P, self.size + 1), dtype='object')
        for parent in parents:
            parent[:-1] = np.random.permutation(self.size)
            parent[-1:] = self.d[
                parent[:-1].astype('int32'),
                np.roll(parent[:-1], -1).astype('int32')
            ].sum()

        # Simulate all the generations
        mating_pool = int(self.n*self.P)
        if mating_pool & 0x1:
            mating_pool -= 1
        for gen in range(self.T):
            # Selecting parents for mating and pair them together
            fitness = np.copy(parents[:, -1:])
            probs = np.max(fitness) + 1 - fitness
            probs /= np.sum(probs)
            mating = np.random.choice(np.arange(self.P),
                                      mating_pool,
                                      False,
                                      probs)
            mating = np.random.choice(mating, (int(mating_pool/2), 2), False)
            # Mating
            children = np.zeros((mating_pool, self.count + 1))
            for pair, (p1, p2) in enumerate(mating):
                child1 = np.copy(parents[p2, :-1])
                child2 = np.copy(parents[p1, :-1])

                gene = 0
                child1[gene], child2[gene] = child2[gene], child1[gene]
                while child1.shape != np.unique(child1).shape:
                    res = np.where(child1 == child1[gene])[0]
                    gene = res[np.where(res != gene)[0][0]]
                    child1[gene], child2[gene] = child2[gene], child1[gene]

                children[2*pair, :-1] = child1
                children[2*pair + 1, :-1] = child2

            # Mutation
            mutation = np.random.rand(mating_pool)
            mutation = np.where(mutation >= self.p)[0]
            genes = np.random.choice(np.arange(self.size),
                                     (mutation.shape[0], 2))
            for m, (g1, g2) in zip(mutation, genes):
                children[m,
                         g1], children[m, g2] = children[m, g2], children[m, g1]

            # Evaluate children
            for child in children:
                child[-1:] = self.d[
                    child[:-1].astype('int32'),
                    np.roll(child[:-1], -1).astype('int32')
                ].sum()

            # Create new parents
            cross_gen = np.concatenate((parents, children), axis=0)
            cross_gen = cross_gen[cross_gen[:, -1:].argsort(), :]
            parents = cross_gen[:self.P]
            if parents[0, -1:] < best[-1:]:
                best = parents[0]

        print(best[-1:])


class Ant:
    """ Class of Ant Colony Algorithm. Holds all the variables and functions
        needed for performing said algorihtm.\n

        Attributes:\n
        tours           -- Number of tours the program will simulate before
        outputing the best found result.\n
        alpha           -- Parameter infulencing impact of the pheromone on
        the ant decision making.\n
        beta            -- Parameter influencing impact of the distance
        between the nodes on the ant decision making.\n
        rho             -- Evporation rate of the pheromones.\n
        d               -- Distance between nodes.\n
        size            -- Number of nodes.\n
        tau             -- Pheromones for each edge between nodes.\n
        n               -- Inverse of distance between nodes.\n
        ants            -- Table with currently simulated ants.\n
        a               -- Table with decision making variables for each edge
        between nodes.\n
        start           -- Starting cities of simulated ants.\n

        Methods:\n
        run             -- Runs the prepaired algorithm with the parameters
        set in __init__ and with the distance matrix passed.\n
        ants_tables     -- Creates self.tau, self.n and best tables that are
        used later in the algorithm.\n
        ants_and_a      -- Creates self.ants, self.a and self.start for each
        tour.\n
        ants_traveling  -- Simulates the traveling process of ants along with
        decision making.\n
        pheromones      -- Simulates the evaporation of pheromones in self.tau.
    """

    def __init__(self, params: dict) -> None:
        """ Initialization of Ant class, takes dict as parameter and assigns
            values from it into corresponding variables.
        """

        self.tours, self.alpha, self.beta, self.rho = params.values()

    def run(self, d: np.ndarray, size: int) -> None:
        """ Runs the algorithm. Takes distance matrix and number of nodes as
            parameters using the parameters set in __init__ performs a
            simulation of Ant Colony finding shortest path.
        """

        start = time.perf_counter()
        logging.info("Staring algorithm",
                     extra={'runtime': time.perf_counter() - start})
        self.d = d
        self.size = size
        best = self.ants_tables()
        for trip in range(self.tours):
            logging.info('Trip %s',
                         str(trip),
                         extra={'runtime': time.perf_counter() - start})
            self.ants_and_a()
            self.ants_traveling()
            index = np.argmin(self.ants[:, self.size])
            best = self.ants[index] if best[self.size] > self.ants[index][
                self.size] else best
            self.pheromones()
        logging.info('Finished.',
                     extra={'runtime': time.perf_counter() - start})
        logging.info('Best Distance: %s',
                     str(best[self.size]),
                     extra={'runtime': 0})

    def ants_tables(self) -> np.ndarray:
        """ Creates necessary tables for the algorithm to run. These tables
            need to be created just once.
        """

        self.tau = np.full((self.size, self.size), 1/self.d.max())
        self.n = 1/self.d
        np.fill_diagonal(self.n, 0)
        best = np.arange(self.size + 1, dtype='object')
        best[self.size] = np.Inf
        return best

    def ants_and_a(self) -> None:
        """ Creates necessary tables for the algorithm to run. These tables
            are created anew for each tour.
        """

        self.ants = np.full((self.size, self.size + 1),
                            self.size + 1, dtype="object")
        self.ants[:, self.size] = float(0)
        self.a = self.tau ** self.alpha * self.n ** self.beta
        self.start = np.random.randint(0, self.size, self.size, dtype='int32')
        self.a /= self.a.sum(axis=1, keepdims=True)

    def ants_traveling(self) -> None:
        """ Simulates the behaviour and the traveling of ants in search of the
            optimal path.
        """

        self.ants[:, 0] = self.start
        for ant in self.ants:
            for j in range(1, self.size):
                p = [self.a[ant[j-1], q] * (q not in ant[0:-1])
                     for q in range(self.size)]
                p /= np.sum(p)
                q = np.argmax(np.random.multinomial(1, p, size=1))
                ant[j] = int(q)
            ant[self.size] = self.d[
                ant[:-1].astype('int32'),
                np.roll(ant[:-1], -1).astype('int32')
            ].sum()

    def pheromones(self) -> None:
        """ Updates the ammount of pheromone on each edge represented by tau.
        """

        self.tau *= (1-self.rho)
        for ant in self.ants:
            self.tau[ant[0:-1].astype('int32'), np.roll(ant[
                0:-1], -1).astype('int32')] += 1/ant[self.size]
            self.tau[ant[0:-1].astype('int32'), np.roll(ant[
                0:-1], 1).astype('int32')] += 1/ant[self.size]
