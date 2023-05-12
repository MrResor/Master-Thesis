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
        logging.info(
            'Best Path:\n%s',
            '->'.join([str(int(v)) for v in self.best[:-1]]),
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
            parent[:-1] = tmp.copy()
            parent[-1] = self.d[tmp, np.roll(tmp, -1)].sum()

    def select_and_mate(self) -> None:
        """ Based on the fitness of parents, they are chosen to mate, and the
            mating itself is performed.
        """

        # Selecting parents for mating and pair them together
        fitness = self.parents[:, -1].copy()
        probs = (np.max(fitness) + 1 - fitness).astype('float64')
        probs /= np.sum(probs)
        mating = np.random.choice(
            np.arange(self.P),
            self.mating_pool,
            False,
            probs.flatten()
        )
        mating = np.random.choice(mating, (int(self.mating_pool/2), 2), False)
        # Mating
        self.children = np.zeros((self.mating_pool, self.size + 1))
        for pair, (p1, p2) in enumerate(mating):
            par1 = self.parents[p1, :-1]
            par2 = self.parents[p2, :-1]
            cut1 = np.random.randint(self.size)
            cut2 = np.random.randint(cut1, self.size + 1)
            


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
        logging.info(
            'Best path:\n%s',
            '->'.join([str(v) for v in best[:-1]]),
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
        init_d = d
        # transform distance matrix into list of vertices, with no duplicates.
        ind = np.triu_indices(size, 1)
        d = np.array([i for i in zip(ind[0], ind[1], d[ind])] +\
                     [i for i in zip(ind[1], ind[0], d[ind])], dtype='object')
        start = perf_counter()
        logging.info(
            "Staring algorithm",
            extra={'runtime': perf_counter() - start}
        )
        d = d[d[:, -1].argsort(), :]
        added = []
        count = defaultdict(lambda: 0)
        free = defaultdict(lambda: True)
        edge = 0
        for i, row in enumerate(d):
            logging.info(
                'Row %s / %s',
                str(i),
                str(d.shape[0]),
                extra={'runtime': perf_counter() - start}
            )
            c1, c2, _ = row
            if count[c1] < 2 and count[c2] < 2 and free[(1, c1)] and free[(2, c2)]:
                new_path = added.copy()
                new_path.append((c1, c2))
                if not self.check_cycle(new_path) or edge == (size - 1):
                    added.append((c1, c2))
                    edge += 1
                    count[c1] += 1
                    count[c2] += 1
                    free[(1, c1)] = False
                    free[(2, c2)] = False

        path = np.zeros((size, 2))
        path[0] = added[0]
        for i, p in enumerate(path[:-1]):
            for a in added:
                if p[1] == a[0]:
                    path[i+1] = a
                    break
        path = np.array(path[:, 0]).astype('int32')
        sol = init_d[path, np.roll(path, 1)].sum()
        logging.info(
            'Finished.',
            extra={'runtime': perf_counter() - start}
        )
        logging.info(
            'Best Distance: %s',
            str(sol),
            extra={'runtime': 0}
        )
        logging.info(
            'Best Path:\n%s',
            '->'.join([str(v) for v in path]),
            extra={'runtime': 0}
        )

    def check_cycle(self, path) -> bool:
        cycle = False
        while path:
            visited = []
            p0 = path[0]
            while p0 not in visited:
                if len(visited) == len(path):
                    break
                visited.append(p0)
                for p1 in path:
                    if p0[1] == p1[0]:
                        p0 = p1
                        break
            if visited[0][0] == visited[-1][1]:
                cycle = True
            for v in visited:
                path.remove(v)
                
        return cycle


class particle_swarm_optimisation:
    def __init__(self, params: dict) -> None:
        """ Initialization of PSO class, takes dict as parameter and assigns
            values from it into corresponding variables.
        """
        coefs, self.i, self.n = params.values()
        self.c1, self.c2, self.c3 = coefs

    def run(self, d: np.ndarray, size: int) -> None:
        self.c1_init = 1
        self.c2 = 2
        self.c3 = 2
        self.d = d
        self.size = size

        # Prepaire container for best and parents
        self.particles = np.zeros((self.n, self.size + 1), dtype='object')
        # self.velocities = list(np.zeros((self.n, 1, 2)))
        for i, particle in enumerate(self.particles):
            tmp = np.random.permutation(self.size)
            particle[:-1] = tmp.copy()
            particle[-1] = self.d[tmp, np.roll(tmp, -1)].sum()
        self.pbest = self.particles.copy()
        self.gbest = self.particles[np.argmin(self.particles[:, -1]), :].copy()
        self.velocities = np.zeros((self.n, self.size))

        for i in range(self.i):
            self.c1 = self.c1_init - i / (2 * self.i)
            for particle, pbest, V in zip(self.particles, self.pbest, self.velocities):
                p = particle[:-1].astype('int32')
                y = np.array([1 - 2 * (np.random.rand() < 0.5) if x == pb and x == gb
                    else int(x == gb) - int(x == pb) for (x, pb, gb) in zip(p, pbest, self.gbest)])
                r1, r2 = np.random.rand(2)
                V1 = self.c1 * V + r1 *self.c2 * (- 1 - y) + r2 * self.c3 * (1 - y)
                lam = V1 + y
                alpha = 0.35
                y = [int(v > alpha) - int(v < -alpha) for v in lam]
                new_p = np.full_like(p, self.size + 1, dtype='int32')
                for j, v in enumerate(y):
                    flag = True
                    if v != 0:
                        r = pbest[j] if v == -1 else self.gbest[j]
                        if (r not in new_p):
                            new_p[j] = r
                            flag = False
                    if flag:
                        r = np.random.randint(self.size)
                        while (r in new_p):
                            r = np.random.randint(self.size)
                        new_p[j] = r
                p[:] = new_p.copy()
                particle[-1] = self.d[p, np.roll(p, -1)].sum()
                if particle[-1] < pbest[-1]:
                    pbest = particle.copy()
            candidate = self.particles[np.argmin(self.particles[-1]), :]
            if candidate[-1] < self.gbest[-1]:
                self.gbest = candidate.copy()
        print(self.gbest)
                


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

        start = perf_counter()
        logging.info(
            "Staring algorithm",
            extra={'runtime': perf_counter() - start}
        )
        path = np.random.permutation(size)
        cur_len = d[path, np.roll(path, -1)].sum()
        improved = True
        count = 0
        while (improved):
            logging.info(
                'Improvement attempt: %s',
                str(count),
                extra={'runtime': perf_counter() - start}
            )
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
            count += 1
        logging.info(
            'Finished.',
            extra={'runtime': perf_counter() - start}
        )
        logging.info(
            'Best Distance: %s',
            str(cur_len),
            extra={'runtime': 0}
        )
        logging.info(
            'Best Path:\n%s',
            '->'.join([str(v) for v in path]),
            extra={'runtime': 0}
        )
