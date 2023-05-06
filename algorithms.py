from __init__ import logging
from time import perf_counter
from collections import defaultdict

from __init__ import np


class Genetic:
    """ Class of Genetic Algorithm. Holds all the variables and functions
        needed for performing said algorihtm.\n

        Attributes:\n
        P                       -- Size of initial population.\n
        n                       -- Multiplier that describes the children
        population, calculated as n*P.\n
        p                       -- Probability of mutation.\n
        T                       -- Number of generations.\n
        d                       -- Distance between nodes.\n
        size                    -- Number of nodes.\n
        best                    -- Best combination so far.\n
        parents                 -- List of current parents.\n
        mating_pool             -- Number of children, results of n*P with
        guard in case the result is odd.\n
        children                -- List of current children.\n

        Methods:\n
        run                     -- Runs the prepaired algorithm with the
        parameters set in __init__ and with the distance matrix and
        information about number of nodes passed.\n
        parents_and_best        -- Creates placeholders for parents and
        best solution.\n
        select_and_mate         -- Selects parents for mating and fills
        children table with newly created ones.\n
        mutate_evaluate_cull    -- Mutates the children, then evaluates their
        fitness and lastly takes old parents and children together and chooses
        P best ones to go to next generation.
    """

    def __init__(self, params: dict) -> None:
        """ Initialization of Genetic class, takes dict as parameter and
            assigns values from it into corresponding variables.
        """

        self.P, self.n, self.p, self.T = params.values()

    def run(self, d: np.ndarray, size: int) -> None:
        """ Runs the algorithm. Takes distance matrix and number of nodes as
            parameters and using the parameters set in __init__ performs a
            simulation of survival of the fittest.
        """

        start = perf_counter()
        logging.info(
            "Staring algorithm",
            extra={'runtime': perf_counter() - start}
        )
        self.d = d
        self.size = size
        self.parents_and_best()

        # Simulate all the generations
        self.mating_pool = int(self.n*self.P)
        if self.mating_pool & 0x1:
            self.mating_pool -= 1
        for gen in range(self.T):
            logging.info(
                'Generation %s',
                str(gen),
                extra={'runtime': perf_counter() - start}
            )
            self.select_and_mate()
            self.mutate_evaluate_cull()

        logging.info(
            'Finished.',
            extra={'runtime': perf_counter() - start}
        )
        logging.info(
            'Best Distance: %s',
            str(self.best[-1]),
            extra={'runtime': 0}
        )

    def parents_and_best(self) -> None:
        """ Creates placeholders for the fittest genome and parents.
        """

        # Prepaire container for best and parents
        self.best = np.arange(self.size + 1, dtype='object')
        self.best[-1] = np.Inf
        self.parents = np.zeros((self.P, self.size + 1), dtype='object')
        for parent in self.parents:
            tmp = np.random.permutation(self.size)
            parent[:-1] = tmp
            parent[-1] = self.d[tmp, np.roll(tmp, -1)].sum()

    def select_and_mate(self) -> None:
        """ Based on the fitness of parents, they are chosen to mate, and the
            mating itself is performed.
        """

        # Selecting parents for mating and pair them together
        fitness = np.copy(self.parents[:, -1])
        probs = np.max(fitness) + 1 - fitness
        probs /= np.sum(probs)
        mating = np.random.choice(
            np.arange(self.P),
            self.mating_pool,
            False,
            probs.astype('float64').flatten()
        )
        mating = np.random.choice(mating, (int(self.mating_pool/2), 2), False)
        # Mating
        self.children = np.zeros((self.mating_pool, self.size + 1))
        for pair, (p1, p2) in enumerate(mating):
            child1 = np.copy(self.parents[p2, :-1])
            child2 = np.copy(self.parents[p1, :-1])

            gene = 0
            child1[gene], child2[gene] = child2[gene], child1[gene]
            while child1.shape != np.unique(child1).shape:
                res = np.where(child1 == child1[gene])[0]
                gene = res[np.where(res != gene)[0][0]]
                child1[gene], child2[gene] = child2[gene], child1[gene]

            self.children[2*pair, :-1] = child1
            self.children[2*pair + 1, :-1] = child2

    def mutate_evaluate_cull(self) -> None:
        """ Performs mutation on children, calculates their fitness and then
            assigns P of the fittest genomes into new parents.
        """

        # Mutation
        mutation = np.random.rand(self.mating_pool)
        mutation = np.where(mutation >= self.p)[0]
        genes = np.random.choice(np.arange(self.size), (mutation.shape[0], 2))
        for m, (g1, g2) in zip(mutation, genes):
            child = self.children[m, :]
            child[g1], child[g2] = child[g2], child[g1]

        # Evaluate children
        for child in self.children:
            tmp = child[:-1].astype('int32')
            child[-1] = self.d[tmp, np.roll(tmp, -1)].sum()

        # Create new parents
        cross_gen = np.concatenate((self.parents, self.children), axis=0)
        cross_gen = cross_gen[cross_gen[:, -1].argsort()]
        self.parents = cross_gen[:self.P, :]
        self.best = self.parents[0, :] if self.best[-1] > self.parents[
            0, -1] else self.best


class Ant:
    """ Class of Ant Colony Algorithm. Holds all the variables and functions
        needed for performing said algorihtm.\n

        Attributes:\n
        tours           -- Number of tours simulated by the program.\n
        alpha           -- Impact of the pheromone on the ant decision making.
        \n
        beta            -- Impact of the distance between the nodes on the ant
        decision making.\n
        rho             -- Evaporation rate of the pheromone.\n
        d               -- Distance between nodes.\n
        size            -- Number of nodes.\n
        tau             -- Table of pheromone for each edge between nodes.\n
        n               -- Inverse of distance between nodes.\n
        ants            -- Table with currently simulated ants.\n
        a               -- Table with decision making variables for each edge
        between nodes.\n
        start           -- Starting cities of simulated ants.\n

        Methods:\n
        run             -- Runs the prepaired algorithm with the parameters
        parameters set in __init__ and with the distance matrix and
        information about number of nodes passed.\n
        ants_tables     -- Creates tau, n and best tables that are used in the
        algorithm.\n
        ants_and_a      -- Creates ants, a and start for each tour.\n
        ants_traveling  -- Simulates the traveling process of ants along with
        decision making.\n
        pheromones      -- Simulates the evaporation of pheromone in tau.
    """

    def __init__(self, params: dict) -> None:
        """ Initialization of Ant class, takes dictionary as parameter and
            assigns values from it into corresponding variables.
        """

        self.tours, self.alpha, self.beta, self.rho = params.values()

    def run(self, d: np.ndarray, size: int) -> None:
        """ Runs the algorithm. Takes distance matrix and number of nodes as
            parameters and using the parameters set in __init__ performs a
            simulation of Ant Colony finding shortest path.
        """

        start = perf_counter()
        logging.info(
            "Staring algorithm",
            extra={'runtime': perf_counter() - start}
        )
        self.d = d
        self.size = size
        best = self.ants_tables()
        for trip in range(self.tours):
            logging.info(
                'Trip %s',
                str(trip),
                extra={'runtime': perf_counter() - start}
            )
            self.ants_and_a()
            self.ants_traveling()
            index = np.argmin(self.ants[:, -1])
            best = self.ants[index] if best[-1] > self.ants[index][
                -1] else best
            self.pheromones()
        logging.info(
            'Finished.',
            extra={'runtime': perf_counter() - start}
        )
        logging.info(
            'Best Distance: %s',
            str(best[-1]),
            extra={'runtime': 0}
        )

    def ants_tables(self) -> np.ndarray:
        """ Creates necessary tables for the algorithm to run. These tables
            need to be created just once.
        """

        self.tau = np.full((self.size, self.size), 1/self.d.max())
        self.n = 1/self.d
        np.fill_diagonal(self.n, 0)
        best = np.arange(self.size + 1, dtype='object')
        best[-1] = np.Inf
        return best

    def ants_and_a(self) -> None:
        """ Creates necessary tables for the algorithm to run. These tables
            are created anew for each tour.
        """

        self.ants = np.full(
            (self.size, self.size + 1),
            self.size + 1,
            dtype="object"
        )
        self.ants[:, -1] = float(0)
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
                p = [
                    self.a[ant[j-1], q] * (q not in ant[0:-1])
                    for q in range(self.size)
                ]
                p /= np.sum(p)
                q = np.argmax(np.random.multinomial(1, p, size=1))
                ant[j] = int(q)
            ant[-1] = self.d[
                ant[:-1].astype('int32'),
                np.roll(ant[:-1], -1).astype('int32')
            ].sum()

    def pheromones(self) -> None:
        """ Updates the ammount of pheromone on each edge represented by tau.
        """

        self.tau *= (1-self.rho)
        for ant in self.ants:
            a = ant[0: -1].astype('int32')
            self.tau[a, np.roll(a, -1)] += 1/ant[-1]
            self.tau[a, np.roll(a, 1)] += 1/ant[-1]


class smallest_edge_algorithm:
    """ Class of Smallest Edge Algorithm. Because the method itself is so
        simple, it consists of single function that performs the algorithm.\n

        Methods:\n
        run             -- Runs the algorithm with the distance matrix and
        information about number of nodes passed.
    """

    def __init__(self, params) -> None:
        """ Dummy, made only to make sure there are no errors when empty
            params are passed.
        """

        pass

    def run(self, d: np.ndarray, size: int) -> None:
        """ Runs the algorithm. Takes distance matrix and number of nodes and
            performs the algorithm.
        """

        # transform distance matrix into list of vertices, with no duplicates.
        ind = np.triu_indices(size, 1)
        d = np.array(
            [[i1, i2, i3] for i1, i2, i3 in zip(ind[0], ind[1], d[ind])]
        )
        start = perf_counter()
        logging.info(
            "Staring algorithm",
            extra={'runtime': perf_counter() - start}
        )
        d = d[d[:, -1].argsort()]
        sol = 0
        included = defaultdict(lambda: 0)
        city_count = 0
        for i, row in enumerate(d):
            logging.info(
                'Row %s / %s',
                str(i), str(d.shape[0]),
                extra={'runtime': perf_counter() - start}
            )
            c1, c2, dist = row
            if included[c1] < 2 and included[c2] < 2:
                sol += dist
                city_count += 1
                included[c1] += 1
                included[c2] += 1
            if city_count == size:
                break
        logging.info(
            'Finished.',
            extra={'runtime': perf_counter() - start}
        )
        logging.info(
            'Best Distance: %s',
            str(sol),
            extra={'runtime': 0}
        )


class particle_swarm_optimisation:
    def __init__(self, params: dict) -> None:
        """ Initialization of PSO class, takes dict as parameter and assigns
            values from it into corresponding variables.
        """
        coefs, self.i, self.n = params.values()
        self.c1, self.c2, self.c3 = coefs

    def run(self, d: np.ndarray, size: int) -> None:
        self.d = d
        self.size = size

        # Prepaire container for best and parents
        self.particles = np.zeros((self.n, self.size + 1, 2), dtype='object')
        self.velocities = list(np.zeros((self.n, 1, 2)))
        for i, particle in enumerate(self.particles):
            tmp = np.random.permutation(self.size)
            particle[:-1, :] = np.array(list(zip(tmp, np.roll(tmp, -1))))
            particle[-1, 0] = self.d[tmp, np.roll(tmp, -1)].sum()
            self.velocities[i] = [
                list(particle[np.random.randint(self.size), :])]
        self.pbest = self.particles.copy()
        self.gbest = self.particles[np.argmin(
            self.particles[:, -1, 0]), :].copy()

        # self.velocities[0] = []
        # self.velocities[0].append([123, 123])

        for i in range(self.i):
            # VELOCITY (??) i update (??)
            Xgb = list({(p[0], p[1]): "" for p in
                        self.gbest[:-1]}.keys())
            for i, particle in enumerate(self.particles):
                X0 = list({(p[0], p[1]): "" for p in particle[:-1]}.keys())
                X0_set = list(map(set, X0))
                Xpbest = list({(p[0], p[1]): "" for p in
                               self.pbest[i, :-1]}.keys())
                Xpbest = [(1, p[0], p[1])
                          for p in Xpbest if set(p) not in X0_set]
                Xgbest = [(1, p[0], p[1])
                          for p in Xgb if set(p) not in X0_set]
                V0 = [(1, p[0], p[1]) for p in self.velocities[i]]
                # Create V1
                V1 = []
                deg = defaultdict(lambda: 0)
                for X in Xgbest:
                    if deg[X[1]] < 4 and deg[X[2]] < 4:
                        V1.append(
                            (self.c3 * np.random.rand() * X[0], X[1], X[2])
                        )
                        deg[X[1]] += 1
                        deg[X[2]] += 1
                for X in Xpbest:
                    if deg[X[1]] < 4 and deg[X[2]] < 4:
                        V1.append(
                            (self.c2 * np.random.rand() * X[0], X[1], X[2])
                        )
                        deg[X[1]] += 1
                        deg[X[2]] += 1
                for X in V0:
                    if deg[X[1]] < 4 and deg[X[2]] < 4:
                        V1.append(
                            (self.c1 * np.random.rand() * X[0], X[1], X[2])
                        )
                        deg[X[1]] += 1
                        deg[X[2]] += 1
                # create partial X1
                X1 = []
                # for V in V1:
                #     if

            self.c1 *= 0.95
            self.c2 *= 1.01
            self.c3 = 1 - self.c1 - self.c2


class opt2:
    """ Class of 2-Opt Algorithm. Because the method itself is so simple, it
        consists of single function that performs the algorithm.\n

        Methods:\n
        run             -- Runs the algorithm with the distance matrix and
        information about number of nodes passed.
    """

    def __init__(self, params: dict) -> None:
        """ Dummy, made only to make sure there are no errors when empty
            params are passed.
        """

        pass

    def run(self, d: np.ndarray, size: int) -> None:
        """ Runs the algorithm. Takes distance matrix and number of nodes and
            performs the algorithm.
        """

        path = np.random.permutation(size)
        cur_len = d[path, np.roll(path, -1)].sum()
        improved = True
        while (improved):
            improved = False
            for i in range(size - 2):
                for j in range(i + 1, size - 1):
                    diff = -d[
                        path[[i, j]],
                        path[[i + 1 % size, j + 1 % size]]
                    ].sum() + d[
                        path[[i, i + 1 % size]],
                        path[[j, j + 1 % size]]
                    ].sum()
                    if diff < 0:
                        path[i+1:j+1] = path[i+1:j+1][::-1]
                        cur_len += diff
                        improved = True
        print(cur_len)
