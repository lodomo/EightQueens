from datetime import datetime

import matplotlib.pyplot as plt


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

    # Create the bar plot with adjusted bar width and spacing
    fig, ax = plt.subplots(figsize=(30, 15))

    # Add title big font
    plt.title("Pop/Mut Rate vs. Solutions Found in 1000 Generations", fontsize=40)

    keys = list(DATA.keys())
    solutions_found = [DATA[key].solutions_found for key in keys]
    labels = [f"{key[0]}/{key[1]}" for key in keys]

    bar_width = 0.75
    bars = ax.bar(labels, solutions_found, width=bar_width)

    # Add bar labels
    ax.bar_label(bars, fmt="%.2f")

    # Set labels
    ax.set_ylabel("Number of Solutions Found In 1000 Generations", fontsize=20)
    ax.set_xlabel("Population/Mutation Rate", fontsize=20)

    # Adjust x-axis labels to avoid overlap
    plt.xticks(rotation=45, ha="right", fontsize=15)

    # Save the plot with a timestamp in the filename
    now = datetime.now()
    plt.savefig(f"./Plots/plot_popmut_solutions{now.strftime('%Y%m%d%H%M%S')}.png")

    # Create a bar graph for each population/mutation combination.
    # X-axis is the population/mutation combination
    # Y-axis is average number of generations
    plt.clf()
    fig, ax = plt.subplots(figsize=(30, 15))

    plt.title("Pop/Mut Rate vs. Average Number of Generations Per Solution Found", fontsize=40)

    generations = [DATA[key].avg_generations for key in keys]
    bars = ax.bar(labels, generations, width=bar_width)

    # Add bar labels
    ax.bar_label(bars, fmt="%.2f", fontsize=15)

    # Set labels
    ax.set_ylabel("Average Number of Generations", fontsize=20)
    ax.set_xlabel("Population/Mutation Rate", fontsize=20)

    # Adjust x-axis labels to avoid overlap
    plt.xticks(rotation=45, ha="right")

    # Save the plot with a timestamp in the filename
    now = datetime.now()
    plt.savefig(f"./Plots/plot_popmut_generations{now.strftime('%Y%m%d%H%M%S')}.png")
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
