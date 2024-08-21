import numpy as np
from function import jianshe, yunying, yunshu, sunhao1to2, sunhao2to3


num_individuals = 100
num_generations = 1000
crossover_rate = 0.8
mutation_rate = 0.01
large_penalty = 1e6  
num_primary_nodes = 40
num_secondary_nodes = 7
num_tertiary_nodes = 3
epoch = '1'
method = 'node'
yun_price = 0
veg_price = []
volum_price = 0
veg_v = []
length_1to2 = []
length_2to3 = []
time_1to2 = []
time_2to3 = []

def test_fitness(solution):
    secondary_selection = solution[num_primary_nodes:num_primary_nodes + num_secondary_nodes]
    tertiary_selection = solution[
                         num_primary_nodes + num_secondary_nodes:num_primary_nodes + num_secondary_nodes + num_tertiary_nodes]
    path_1_to_2 = solution[
                  num_primary_nodes + num_secondary_nodes + num_tertiary_nodes:num_primary_nodes + num_secondary_nodes + num_tertiary_nodes + num_primary_nodes]
    path_2_to_3 = solution[num_primary_nodes + num_secondary_nodes + num_tertiary_nodes + num_primary_nodes:]

    secondary_selected_indices = np.where(secondary_selection == 1)[0]
    tertiary_selected_indices = np.where(tertiary_selection == 1)[0]

    if len(secondary_selected_indices) == 0 or len(tertiary_selected_indices) == 0:
        return -large_penalty  
    
    total_transport_cost = 0

    total_loss_cost = 0

    total_loss_cost_1to2 = 0
    total_loss_cost_2to3 = 0

    total_veg_v2 = [[0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0]]

    for i in range(num_primary_nodes):
        j = path_1_to_2[i]
        if j in secondary_selected_indices:

            lenth12 = length_1to2[i][j]
            total_veg = veg_v[i][0] + veg_v[i][1] + veg_v[i][2] + veg_v[i][3] + veg_v[i][4] + veg_v[i][5]
            total_transport_cost += yunshu(lenth12, total_veg, yun_price)

            time12 = time_1to2[i][j]

            total_loss_cost_1to2 += sunhao1to2(time12, total_veg)
            total_veg_v2[j][0] += veg_v[i][0]
            total_veg_v2[j][1] += veg_v[i][1]
            total_veg_v2[j][2] += veg_v[i][2]
            total_veg_v2[j][3] += veg_v[i][3]
            total_veg_v2[j][4] += veg_v[i][4]
            total_veg_v2[j][5] += veg_v[i][5]

        else:
            total_transport_cost += large_penalty

        all_veg_V2 = total_veg_v2[j][0] + total_veg_v2[j][1] + total_veg_v2[j][2] + total_veg_v2[j][3] + \
                     total_veg_v2[j][4] + total_veg_v2[j][5]

        if all_veg_V2 > 100:
            total_loss_cost_1to2 += large_penalty

    for j in secondary_selected_indices:
        k = path_2_to_3[j]
        if k in tertiary_selected_indices:

            lenth23 = length_2to3[j][k]
            total_veg2 = total_veg_v2[j][0] + total_veg_v2[j][1] + total_veg_v2[j][2] + total_veg_v2[j][3] + total_veg_v2[j][4] + \
                        total_veg_v2[j][5]
            total_transport_cost += yunshu(lenth23, total_veg2, yun_price)

            time23 = time_2to3[j][k]
            total_loss_cost_2to3 += sunhao2to3(time23, total_veg2)

        else:
            total_transport_cost += large_penalty

    total_construction_cost = jianshe(0, len(secondary_selected_indices), len(tertiary_selected_indices))

    total_operation_cost = yunying(volum_price, veg_v, num_primary_nodes, len(secondary_selected_indices),
                                   len(tertiary_selected_indices))

    total_cost = total_transport_cost + total_construction_cost + total_operation_cost + total_loss_cost_1to2 + total_loss_cost_2to3

    f = open(method + '_generation_best_cost_' + epoch + '.txt', 'a')
    f.write("\"best_jianshe\":\"" + "{}\"\n".format(total_construction_cost))
    f.write("\"best_yunyin\":\"" + "{}\"\n".format(total_operation_cost))
    f.write("\"best_yunshu\":\"" + "{}\"\n".format(total_transport_cost))
    f.write("\"best_sunhao1to2\":\"" + "{}\"\n".format(total_loss_cost_1to2))
    f.write("\"best_sunhao2to3\":\"" + "{}\"\n".format(total_loss_cost_2to3))
    f.write("\"best_cost\":\"" + "{}\"\n".format(total_cost))
    f.close()

    return -total_cost


def fitness(solution):
    secondary_selection = solution[num_primary_nodes:num_primary_nodes + num_secondary_nodes]
    tertiary_selection = solution[num_primary_nodes + num_secondary_nodes:num_primary_nodes + num_secondary_nodes + num_tertiary_nodes]
    path_1_to_2 = solution[num_primary_nodes + num_secondary_nodes + num_tertiary_nodes:num_primary_nodes + num_secondary_nodes + num_tertiary_nodes + num_primary_nodes]
    path_2_to_3 = solution[num_primary_nodes + num_secondary_nodes + num_tertiary_nodes + num_primary_nodes:]

    secondary_selected_indices = np.where(secondary_selection == 1)[0]
    tertiary_selected_indices = np.where(tertiary_selection == 1)[0]

    total_transport_cost = 0
    total_loss_cost = 0

    total_veg_v2 = [[0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0]]

    for i in range(num_primary_nodes):
        j = path_1_to_2[i]
        if j in secondary_selected_indices:

            lenth12 = length_1to2[i][j]
            total_veg = veg_v[i][0] + veg_v[i][1] + veg_v[i][2] + veg_v[i][3] + veg_v[i][4] + veg_v[i][5]
            total_transport_cost += yunshu(lenth12, total_veg, yun_price)

            time12 = time_1to2[i][j]

            total_loss_cost += sunhao1to2(time12, total_veg)

            total_veg_v2[j][0] += veg_v[i][0]
            total_veg_v2[j][1] += veg_v[i][1]
            total_veg_v2[j][2] += veg_v[i][2]
            total_veg_v2[j][3] += veg_v[i][3]
            total_veg_v2[j][4] += veg_v[i][4]
            total_veg_v2[j][5] += veg_v[i][5]

        else:
            total_transport_cost += large_penalty

        all_veg_V2 = total_veg_v2[j][0] + total_veg_v2[j][1] + total_veg_v2[j][2] + total_veg_v2[j][3] + \
                     total_veg_v2[j][4] + total_veg_v2[j][5]

        if all_veg_V2 > 100:
            total_loss_cost += large_penalty

    for j in secondary_selected_indices:
        k = path_2_to_3[j]
        if k in tertiary_selected_indices:
            lenth23 = length_2to3[j][k]
            total_veg2 = total_veg_v2[j][0] + total_veg_v2[j][1] + total_veg_v2[j][2] + total_veg_v2[j][3] + \
                         total_veg_v2[j][4] + total_veg_v2[j][5]

            total_transport_cost += yunshu(lenth23, total_veg2, yun_price)

            time23 = time_2to3[j][k]
            total_loss_cost += sunhao2to3(time23, total_veg2)

        else:
            total_transport_cost += large_penalty

    total_construction_cost = jianshe(0, len(secondary_selected_indices), len(tertiary_selected_indices))

    total_operation_cost = yunying(volum_price, veg_v, num_primary_nodes, len(secondary_selected_indices),
                                   len(tertiary_selected_indices))

    total_cost = total_transport_cost + total_construction_cost + total_operation_cost + total_loss_cost
    return -total_cost


def generate_initial_population():
    population = []
    for _ in range(num_individuals):
        individual = np.zeros(num_primary_nodes + num_secondary_nodes + num_tertiary_nodes + num_primary_nodes + num_secondary_nodes, dtype=int)
        individual[:num_primary_nodes] = 1
        individual[num_primary_nodes:num_primary_nodes + num_secondary_nodes] = np.random.randint(0, 2, num_secondary_nodes)
        while np.sum(individual[num_primary_nodes:num_primary_nodes + num_secondary_nodes]) < 4:
            individual[num_primary_nodes:num_primary_nodes + num_secondary_nodes] = np.random.randint(0, 2, num_secondary_nodes)
        individual[num_primary_nodes + num_secondary_nodes:num_primary_nodes + num_secondary_nodes + num_tertiary_nodes] = np.random.randint(0, 2, num_tertiary_nodes)
        while np.sum(individual[num_primary_nodes + num_secondary_nodes:num_primary_nodes + num_secondary_nodes + num_tertiary_nodes]) < 1:
            individual[num_primary_nodes + num_secondary_nodes:num_primary_nodes + num_secondary_nodes + num_tertiary_nodes] = np.random.randint(0, 2, num_tertiary_nodes)
        secondary_selected_indices = np.where(individual[num_primary_nodes:num_primary_nodes + num_secondary_nodes] == 1)[0]
        tertiary_selected_indices = np.where(individual[num_primary_nodes + num_secondary_nodes:num_primary_nodes + num_secondary_nodes + num_tertiary_nodes] == 1)[0]

        if len(secondary_selected_indices) > 0:
            individual[num_primary_nodes + num_secondary_nodes + num_tertiary_nodes:num_primary_nodes + num_secondary_nodes + num_tertiary_nodes + num_primary_nodes] = np.random.choice(secondary_selected_indices, num_primary_nodes)
        if len(tertiary_selected_indices) > 0:
            individual[num_primary_nodes + num_secondary_nodes + num_tertiary_nodes + num_primary_nodes:] = np.random.choice(tertiary_selected_indices, num_secondary_nodes)
        population.append(individual)
    return np.array(population)


def fix_solution(solution):
    secondary_selection = solution[num_primary_nodes:num_primary_nodes + num_secondary_nodes]
    tertiary_selection = solution[num_primary_nodes + num_secondary_nodes:num_primary_nodes + num_secondary_nodes + num_tertiary_nodes]
    secondary_selected_indices = np.where(secondary_selection == 1)[0]
    tertiary_selected_indices = np.where(tertiary_selection == 1)[0]

    if len(secondary_selected_indices) > 0:
        solution[num_primary_nodes + num_secondary_nodes + num_tertiary_nodes:num_primary_nodes + num_secondary_nodes + num_tertiary_nodes + num_primary_nodes] = np.random.choice(secondary_selected_indices, num_primary_nodes)
    if len(tertiary_selected_indices) > 0:
        solution[num_primary_nodes + num_secondary_nodes + num_tertiary_nodes + num_primary_nodes:] = np.random.choice(tertiary_selected_indices, num_secondary_nodes)


population = generate_initial_population()
most_best_cost = large_penalty
most_best_individual = 0

for generation in range(num_generations):
    
    fitness_values = np.array([fitness(individual) for individual in population])
    max_fitness = np.min(fitness_values)
    transformed_fitness_values = max_fitness - fitness_values

    if np.sum(transformed_fitness_values) == 0:
        probabilities = np.ones(num_individuals) / num_individuals
    else:
        probabilities = transformed_fitness_values / np.sum(transformed_fitness_values)

    selected_indices = np.random.choice(np.arange(num_individuals), size=num_individuals, p=probabilities)
    selected_population = population[selected_indices]

    for i in range(0, num_individuals, 2):
        if np.random.rand() < crossover_rate:
            crossover_point = np.random.randint(1, len(population[0]))
            selected_population[i, crossover_point:], selected_population[i + 1, crossover_point:] = \
                selected_population[i + 1, crossover_point:], selected_population[i, crossover_point:]

    for individual in selected_population:
        if np.random.rand() < mutation_rate:
            mutation_point = np.random.randint(len(population[0]))
            if mutation_point < num_primary_nodes:
                individual[mutation_point] = 1
            elif mutation_point < num_primary_nodes + num_secondary_nodes + num_tertiary_nodes:
                individual[mutation_point] = 1 - individual[mutation_point]
            elif mutation_point < num_primary_nodes + num_secondary_nodes + num_tertiary_nodes + num_primary_nodes:
                secondary_selected_indices = np.where(individual[num_primary_nodes:num_primary_nodes + num_secondary_nodes] == 1)[0]
                if len(secondary_selected_indices) > 0:
                    individual[mutation_point] = np.random.choice(secondary_selected_indices)
            else:
                tertiary_selected_indices = np.where(individual[num_primary_nodes + num_secondary_nodes:num_primary_nodes + num_secondary_nodes + num_tertiary_nodes] == 1)[0]
                if len(tertiary_selected_indices) > 0:
                    individual[mutation_point] = np.random.choice(tertiary_selected_indices)
        fix_solution(individual)

    population = selected_population
    best_individual = population[np.argmax([fitness(ind) for ind in population])]
    best_cost = -fitness(best_individual)
    f = open(method + '_generation_best_cost_' + epoch + '.txt', 'a')
    f.write("\"generation\":\"" + "{}\"\n".format(generation + 1))
    f.write("\"best_individual\":\"" + "{}\"\n".format(best_individual))
    f.write("\"best_cost\":\"" + "{}\"\n".format(best_cost))
    f.close()

    if best_cost < most_best_cost:
        most_best_cost = best_cost
        most_best_individual = best_individual


most_best_cost = -test_fitness(most_best_individual)
print("most_best_individual：", most_best_individual)
print("most_best_cost：", most_best_cost)
