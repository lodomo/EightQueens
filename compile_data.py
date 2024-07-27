import matplotlib.pyplot as plt
from datetime import datetime

def main():
    # Open a file called ./Data/data_collection.csv
    # If it doesn't exist exit
    data_file = None
    try:
        data_file = open("./Data/data_collection.csv", "r")
    except FileNotFoundError:
        print("File not found")
        exit(1)

    # File format is as follows:
    POPULATIONS = [10, 100, 250, 500, 1000, 10000]
    MUTATION_RATES = [0.01, 0.10, 0.25, 0.50, 0.75, 1]

    DATA = {}

    # Read each line of the file and split into , separated values
    # Skip the first line
    for line in data_file:
        if line.startswith("Population"):
            continue

        values = line.split(",")
        population = int(values[0])
        mutation_rate = float(values[1])
        # max_generations = int(values[2])
        mutations = int(values[3])
        # split_points = values[4]
        # average_fitness = values[5]
        # best_fitness = values[6]
        generations = int(values[7])
        solution = values[8]
        delta_time = float(values[9])

        if (population, mutation_rate) not in DATA:
            DATA[(population, mutation_rate)] = Data(population, mutation_rate)

        cur_data = DATA[(population, mutation_rate)]
        cur_data.count += 1
        cur_data.generations += generations
        cur_data.mutations += mutations
        if solution != "No Solution":
            cur_data.solutions_found += 1
        cur_data.delta_time += delta_time

    data_file.close()

    for key in DATA:
        DATA[key].calc_averages()

    # Create a bar chart for each data set
    # Create 36 plots, one for each population/mutation rate combination
    # Each plot will have 4 bars, one for average generations, mutations, solutions found, and time
    # Save the plot as plot.png
    fig, axs = plt.subplots(6, 6, figsize=(20, 20))
    fig.suptitle("Genetic Algorithm Data")
    fig.tight_layout(pad=3.0)

    for i in range(6):
        for j in range(6):
            try:
                key = (POPULATIONS[i], MUTATION_RATES[j])
                cur_data = DATA[key]
            except KeyError:
                print(f"Key {POPULATIONS[i]}, {MUTATION_RATES[j]} not found")
                exit(1)

            axs[i, j].bar(["Gens", "Solns", "Time"],
                          [cur_data.avg_generations, cur_data.solutions_found, cur_data.avg_time])
            axs[i, j].set_title(f"Population: {POPULATIONS[i]}, Mutation Rate: {MUTATION_RATES[j]}")

    now = datetime.now()
    plt.savefig(f"./Plots/plot_{now.strftime('%Y%m%d%H%M%S')}.png")
    exit(0)


class Data:
    def __init__(self, population, mutation_rate):
        self.population = population
        self.mutation_rate = mutation_rate
        self.count = 0
        self.generations = 0
        self.mutations = 0
        self.solutions_found = 0
        self.delta_time = 0
        self.avg_generations = 0
        self.avg_time = 0
        self.avg_mutations = 0

    def calc_averages(self):
        if self.count == 0:
            self.count = 1000

        self.avg_generations = self.generations / self.count
        self.avg_time = self.delta_time / self.count
        self.avg_mutations = self.mutations / self.count


if __name__ == "__main__":
    main()
