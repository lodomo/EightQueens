###############################################################################
#
# Author: Lorenzo D. Moon
# Professor: Dr. Anthony Rhodes
# Course: CS-441
# Assignment: Programming Assignment 2
# Description: Solves the eight queens algorithm using a genetic algorithm
#              See README.md for more information
#
###############################################################################

import getopt
import sys

import numpy as np

# Global Constants
N_QUEENS = 8
COLUMNS = 8
ROWS = 8
MAX_FITNESS = 28  # 8c2 = 28

# Things the user can change
SET = {}
SET["POP_SIZE"] = 10
SET["MUT_RATE"] = 0.01
SET["MAX_GEN"] = 10
SET["VERBOSE"] = 0


def main(argv):
    process_options(argv)
    population = init_population()
    statistics = []
    solution = None

    # Logic Loop is as follow:
    #   1. Calculate the total fitness of the population
    #   2. Set the probability of each queen
    #   3. Sort the population by probability
    #   4. Get statistics on current generation
    #   5. Check for termination condition 
    #   6. Create a new generation
    #   7. Repeat
    while SET["MAX_GEN"] is None or SET["MAX_GEN"] > 0:
        total_fitness = sum([queen.fitness for queen in population])
        set_probabilities(population, total_fitness)
        population = np.sort(population)
        avg_fitness = total_fitness / len(population)
        best_fitness = population[0].fitness
        statistics.append((avg_fitness, best_fitness))
        if best_fitness == MAX_FITNESS:
            break
        population = next_generation(population)
    exit(0)


def process_options(argv):
    OPTIONS = "hp:m:g:v"

    try:
        opts, args = getopt.getopt(argv, OPTIONS)
    except getopt.GetoptError:
        print("Invalid arguments")
        sys.exit(2)

    for opt, arg in opts:
        if opt == "-h":
            print(
                "Usage: eight_queens.py -p <population_size> -m <mutation_rate> -g <max_generations>"
            )
            sys.exit(0)
        elif opt == "-p":
            SET["POP_SIZE"] = int(arg)
        elif opt == "-m":
            rate = float(arg) / 100.0
            if rate < 0:
                rate = 0
            elif rate > 1:
                rate = 1
            SET["MUT_RATE"] = rate
        elif opt == "-g":
            SET["MAX_GEN"] = int(arg)
        elif opt == "-v":
            SET["VERBOSE"] = 1
    return


def help(usage_only=False):
    usage = "Usage: eight_queens.py -p <population_size> "
    usage += "-m <mutation_rate> -g <max_generations>"

    if usage_only:
        print(usage)
        return

    print("8-Queens Help")
    print("-h\tprint this message")
    print("-p\tSet Population size. Default:100")
    print("-m\tSet Chance of Mutation as percentage [0-100] Default:0%")
    print("-g\tSet Max Generations. Default:Disabled")
    print("-v\tVerbose Mode. Default:Disabled")
    print(usage)


def init_population():
    def gen_random():
        # Create a numpy array of 8 digits, each representing a column
        # The value of the digit represents the row
        return np.random.randint(0, ROWS, N_QUEENS)

    # Create a numpy array of Conflicts
    population = np.empty(SET["POP_SIZE"], dtype=Queens)
    for i in range(SET["POP_SIZE"]):
        population[i] = Queens(soln=gen_random())
    return population


def set_probabilities(population, total_fitness):
    for queen in population:
        queen.set_probability(total_fitness)
    return


class Queens:
    def __init__(self, soln=None, predecessors=None):
        self.soln = soln
        self.predecessors = predecessors
        self.fitness = self.calc_fitness()
        self.probability = 0

    def __str__(self):
        string = f"{self.soln}, FIT:{self.fitness}, PROB:{self.probability}\n"
        return string

    def __repr__(self):
        return self.__str__()

    def __gt__(self, other):
        return self.probability > other.probability

    def __lt__(self, other):
        return self.probability < other.probability

    def __eq__(self, other):
        return self.probability == other.probability

    def __ne__(self, other):
        return not (self == other)

    def __ge__(self, other):
        return not (self < other)

    def __le__(self, other):
        return not (self > other)

    def calc_fitness(self):
        fitness = MAX_FITNESS

        for i in range(N_QUEENS):
            for j in range(i + 1, N_QUEENS):
                if self.soln[i] == self.soln[j]:
                    fitness -= 1
                elif self.soln[i] == self.soln[j] + (j - i):
                    fitness -= 1
        return fitness

    def set_probability(self, total_fitness):
        self.probability = self.fitness / total_fitness
        return


if __name__ == "__main__":
    main(sys.argv)
