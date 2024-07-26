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
import os

import numpy as np
import matplotlib.pyplot as plt

# Global Constants
N_QUEENS = 8
COLUMNS = 8
ROWS = 8
MAX_FITNESS = 28  # 8c2 = 28

# Things the user can change
SET = {}
SET['POP_SIZE'] = 1000
SET['MUT_RATE'] = 0.10
SET['MAX_GEN'] = 1000
SET['VERBOSE'] = 0
SET['PLOT'] = False
SET['DATA_OUTPUT'] = False

# Data Collection
DATA = {}
DATA['MUTATIONS_COUNT'] = 0
DATA['SPLIT_POINTS'] = np.zeros(8, dtype=int)
DATA['AVG_FITNESS'] = []
DATA['BEST_FITNESS'] = []
DATA['GENERATIONS'] = None
DATA['SOLUTION'] = None


def main(argv):
    process_options(argv)
    print(f"{SET['VERBOSE']}")
    verbose(f"Population Size: {SET['POP_SIZE']}")
    verbose(f"Mutation Rate: {SET['MUT_RATE']}")
    verbose(f"Max Generations: {SET['MAX_GEN']}")
    population = None
    solution = None
    max_gen = SET['MAX_GEN']
    cur_gen = 0

    # Logic Loop is as follow:
    #   1. Calculate the total fitness of the population
    #   2. Set the probability of each queen
    #   3. Sort the population by probability
    #   4. Get statistics on current generation
    #   5. Check for termination condition
    #   6. Create a new generation
    #   7. Repeat
    while cur_gen < max_gen or max_gen is None:
        population = next_generation(population)
        total_fitness = sum([queen.fitness for queen in population])
        set_probabilities(population, total_fitness)
        population = np.sort(population)[::-1]  # Sort in descending order
        avg_fitness = total_fitness / len(population)
        best_fitness = population[0].fitness
        DATA['AVG_FITNESS'].append(avg_fitness)
        DATA['BEST_FITNESS'].append(best_fitness)

        verbose(f"A generation is born. Generation: {cur_gen}")
        verbose(f"Average Fitness: {avg_fitness}")
        verbose(f"Best Fitness: {best_fitness}")
        verbose(f"{population[0]}", 2)
        if SET['VERBOSE'] >= 2:
            input("Press Enter to continue...")

        if best_fitness == MAX_FITNESS:
            break
        cur_gen += 1

    # Check if we found a solution
    if best_fitness == MAX_FITNESS:
        DATA['GENERATIONS'] = cur_gen
        DATA['SOLUTION'] = population[0]

        print("Solution Found")
        print(f"Solved in {cur_gen} generations")
        print(population[0])
    else:
        print("No Solution Found")
        print(f"Best Fitness: {best_fitness}")
        print(population[0])

    plot_data()
    output_data()
    exit(0)


def process_options(argv):
    OPTIONS = "hvp:m:g:PD"

    try:
        opts, args = getopt.getopt(argv[1:], OPTIONS)
        print(f"Current Line: {sys._getframe().f_lineno}")
        print(f"opts: {opts}")
        print(f"args: {args}")
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
            SET['POP_SIZE'] = int(arg)
        elif opt == "-m":
            rate = float(arg) / 100.0
            if rate < 0:
                rate = 0
            elif rate > 1:
                rate = 1
            SET['MUT_RATE'] = rate
        elif opt == "-g":
            SET['MAX_GEN'] = int(arg)
        elif opt == "-P":
            SET['PLOT'] = True
        elif opt == "-D":
            SET['DATA_OUTPUT'] = True
        elif opt == "-v":
            SET['VERBOSE'] += 1
            verbose("Verbose Mode Enabled")
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


def next_generation(population):
    if population is None:
        return init_population()

    # Create a new empty population
    # For each 2 children, select 2 parents, crossover, and mutate
    # Add the children to the new population
    # Repeat until the new population is full

    new_population = np.empty(SET["POP_SIZE"], dtype=Queens)

    # For loop but add two to the index each time
    for i in range(0, SET["POP_SIZE"], 2):
        parent1 = select_parent(population)
        parent2 = select_parent(population)

        # Ensure that the parents are not the same
        while parent1 == parent2:
            verbose("Parents are the same. Selecting new parent2", 2)
            parent2 = select_parent(population)

        children = crossover(parent1, parent2)
        children[0].mutate()
        children[1].mutate()
        new_population[i] = children[0]
        new_population[i + 1] = children[1]

        verbose(f"New Family:", 2)
        verbose(f"Parent : {parent1}", 2)
        verbose(f"Parent : {parent2}", 2)
        verbose(f"Child 1: {children[0]}", 2)
        verbose(f"Child 2: {children[1]}", 2)
        if SET['VERBOSE'] >= 3:
            input("Press Enter to continue...")

    return new_population


def select_parent(population):
    # This assumes the population is sorted by probability
    # Generate a random number between 0 and 1
    x = np.random.rand()

    # Find the first queen that has a probability greater than x
    for queen in population:
        if queen.probability > x:
            return queen
        x -= queen.probability


def crossover(parent1, parent2):
    # Generate a random number between 0 and 7 for the split point
    x = np.random.randint(0, N_QUEENS)
    verbose(f"Split Point: {x}", 2)
    DATA['SPLIT_POINTS'][x] += 1
    # Create a new solution by combining the first x elements of parent1
    # and the last 8-x elements of parent2
    new_soln1 = np.concatenate((parent1.soln[:x], parent2.soln[x:]))
    new_soln2 = np.concatenate((parent2.soln[:x], parent1.soln[x:]))
    child1 = Queens(soln=new_soln1, predecessors=[parent1, parent2])
    child2 = Queens(soln=new_soln2, predecessors=[parent1, parent2])

    return [child1, child2]


class Queens:
    def __init__(self, soln=None, predecessors=None):
        self.soln = soln
        self.predecessors = predecessors
        self.fitness = self.calc_fitness()
        self.probability = 0
        self.mut_position = None

    def __str__(self):
        # Bold the entry of the mut position if there is one.
        mut = self.mut_position
        soln = self.soln
        fit = self.fitness
        prob = self.probability

        string = ""
        if mut is not None:
            for i in range(N_QUEENS):
                if i == mut:
                    string += "\033[1m"
                string += f"{soln[i]} "
                if i == mut:
                    string += "\033[0m"
        else:
            for i in range(N_QUEENS):
                string += f"{soln[i]} "

        string += f" Fitness: {fit}"

        if prob != 0:
            string += f" Probability: {prob}"
        else:
            string += " Probability: New Born"
        return string

    def __repr__(self):
        return self.__str__()

    def __gt__(self, other):
        return self.probability > other.probability

    def __lt__(self, other):
        return self.probability < other.probability

    # *** BEEG NOTE ***
    # THIS EQUALITY IS NOT THE SAME AS THE INEQUALITIES.
    # THIS IS TO CHECK IF TWO QUEENS HAVE THE SAME SOLUTION
    # *** BEEG NOTE ***
    def __eq__(self, other):
        val = True
        for i in range(len(self.soln)):
            if self.soln[i] != other.soln[i]:
                val = False
        return val

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

    def mutate(self):
        # Generate a random number between 0 and 1
        x = np.random.rand()

        # If the random number is less than the mutation rate, mutate the queen
        if x < SET['MUT_RATE']:
            # Generate a random number between 0 and 7
            self.mut_position = np.random.randint(0, N_QUEENS)
            self.soln[self.mut_position] = np.random.randint(0, ROWS)
            DATA['MUTATIONS_COUNT'] += 1
            verbose(f"Mutation at position {self.mut_position}", 2)
        return


def verbose(message, level=1):
    if SET['VERBOSE'] >= level:
        print(message)
    return


def plot_data():
    if not SET['PLOT']:
        return

    dir = "./Plots"

    # Check if the directory exists
    if not os.path.exists(dir):
        os.makedirs(dir)

    # Create a plot of generation vs average fitness
    # Plot the line in red
    plt.plot(DATA['AVG_FITNESS'], 'r-')
    plt.xlabel("Generation")
    plt.ylabel("Average Fitness")
    plt.title("Generation vs Average Fitness")

    # Make the x axis integers
    # Make the steps 1 if the generation is less than 20
    # Make the steps 2 if the generation is less than 50
    # Make the steps 5 if the generation is less than 100
    # Make the steps 10 if the generation is less than 200
    steps = 0
    if DATA['GENERATIONS'] < 20:
        steps = 1
    elif DATA['GENERATIONS'] < 50:
        steps = 2
    elif DATA['GENERATIONS'] < 100:
        steps = 5
    elif DATA['GENERATIONS'] < 200:
        steps = 10
    else:
        steps = 20
    
    plt.xticks(np.arange(0, len(DATA['AVG_FITNESS']), step=steps))
    # Make the y axis from 0 to 28
    plt.yticks(np.arange(0, 29, step=1))

    # On top of the previous plot, plot the best fitness
    # Plot the line in blue
    plt.plot(DATA['BEST_FITNESS'], 'b-')
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.title("Generation vs Fitness")

    # Save the plot to a file
    # fitness_POPSIZE_MUTRATE_MAXGEN_ENDGEN.png
    name = f"{dir}/"
    name += f"{SET['POP_SIZE']}_{int(SET['MUT_RATE']*100)}_{SET['MAX_GEN']}"
    name += f"_{DATA['GENERATIONS']}"
    name += ".png"
    plt.savefig(name)
    return


def output_data():
    if not SET['DATA_OUTPUT']:
        return

    dir = "./Data"

    # Check if the directory exists
    if not os.path.exists(dir):
        os.makedirs(dir)

    file = f"{dir}/"
    file += f"{SET['POP_SIZE']}_{int(SET['MUT_RATE']*100)}_{SET['MAX_GEN']}"
    file += ".csv"

    # Create the file if it does not exist
    if not os.path.isfile(file):
        with open(file, "w") as f:
            f.write("Mutations,")
            f.write("Split Points,")
            f.write("Average Fitness,")
            f.write("Best Fitness,")
            f.write("Generations,")
            f.write("Solution,")
            f.write("\n")

    # Compile data into a string
    data = f"{DATA['MUTATIONS_COUNT']},"

    split_points = ""
    for i in range(N_QUEENS):
        split_points += f"{DATA['SPLIT_POINTS'][i]} "

    data += f"{split_points}, "

    avg_fitness = ""
    for i in range(len(DATA['AVG_FITNESS'])):
        avg_fitness += f"{DATA['AVG_FITNESS'][i]} "
    data += f"{avg_fitness}, "

    best_fitness = ""
    for i in range(len(DATA['BEST_FITNESS'])):
        best_fitness += f"{DATA['BEST_FITNESS'][i]} "
    data += f"{best_fitness}, "
    data += f"{DATA['GENERATIONS']}, "

    soln = ""
    for i in range(N_QUEENS):
        soln += f"{DATA['SOLUTION'].soln[i]} "
    data += f"{soln}\n"

    with open(file, "a") as f:
        f.write(data)
    return


if __name__ == "__main__":
    main(sys.argv)
