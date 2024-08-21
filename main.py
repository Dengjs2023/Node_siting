import numpy as np
from function import jianshe, yunying, yunshu, sunhao1to2, sunhao2to3


# 参数定义
num_individuals = 100
num_generations = 1000
crossover_rate = 0.8
mutation_rate = 0.01
large_penalty = 1e6  # 大的惩罚值，用来代替无穷大

# 节点数量
num_primary_nodes = 40
num_secondary_nodes = 7
num_tertiary_nodes = 3

epoch = '1'
method = 'node'

# 单位运输费用
yun_price = 0
# 蔬菜单价
veg_price = []
# 一级节点单位运营费用
volum_price = 0
# 一级节点蔬菜体积和质量
veg_v = []

# 节点间距离
length_1to2 = []
length_2to3 = []
# 节点间时间
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
        return -large_penalty  # 如果没有选中任何二级或三级节点，适应度极低

    # 总运输成本
    total_transport_cost = 0

    # 总损耗成本
    total_loss_cost = 0

    total_loss_cost_1to2 = 0
    total_loss_cost_2to3 = 0

    # 减去损耗后的二级节点蔬菜量
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
            total_transport_cost += large_penalty  # 如果路径无效，增加一个很大的成本

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
            total_transport_cost += large_penalty  # 如果路径无效，增加一个很大的成本

    # 总建设成本
    total_construction_cost = jianshe(0, len(secondary_selected_indices), len(tertiary_selected_indices))

    # 总运营成本
    total_operation_cost = yunying(volum_price, veg_v, num_primary_nodes, len(secondary_selected_indices),
                                   len(tertiary_selected_indices))


    print("建设成本：", total_construction_cost)
    print("运营成本：", total_operation_cost)
    print("运输成本：", total_transport_cost)
    print("一至二级节点损耗成本：", total_loss_cost_1to2)
    print("二至三级节点损耗成本：", total_loss_cost_2to3)

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


# 适应度函数
def fitness(solution):
    secondary_selection = solution[num_primary_nodes:num_primary_nodes + num_secondary_nodes]
    tertiary_selection = solution[num_primary_nodes + num_secondary_nodes:num_primary_nodes + num_secondary_nodes + num_tertiary_nodes]
    path_1_to_2 = solution[num_primary_nodes + num_secondary_nodes + num_tertiary_nodes:num_primary_nodes + num_secondary_nodes + num_tertiary_nodes + num_primary_nodes]
    path_2_to_3 = solution[num_primary_nodes + num_secondary_nodes + num_tertiary_nodes + num_primary_nodes:]

    secondary_selected_indices = np.where(secondary_selection == 1)[0]
    tertiary_selected_indices = np.where(tertiary_selection == 1)[0]

    total_transport_cost = 0

    # 总损耗成本
    total_loss_cost = 0

    # 减去损耗后的二级节点蔬菜量
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
            total_transport_cost += large_penalty  # 如果路径无效，增加一个很大的成本

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
            total_transport_cost += large_penalty  # 如果路径无效，增加一个很大的成本


    # 总建设成本
    total_construction_cost = jianshe(0, len(secondary_selected_indices), len(tertiary_selected_indices))

    # 总运营成本
    total_operation_cost = yunying(volum_price, veg_v, num_primary_nodes, len(secondary_selected_indices),
                                   len(tertiary_selected_indices))



    total_cost = total_transport_cost + total_construction_cost + total_operation_cost + total_loss_cost
    return -total_cost

# 初始化种群
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

# 修正路径选择
def fix_solution(solution):
    secondary_selection = solution[num_primary_nodes:num_primary_nodes + num_secondary_nodes]
    tertiary_selection = solution[num_primary_nodes + num_secondary_nodes:num_primary_nodes + num_secondary_nodes + num_tertiary_nodes]
    secondary_selected_indices = np.where(secondary_selection == 1)[0]
    tertiary_selected_indices = np.where(tertiary_selection == 1)[0]

    if len(secondary_selected_indices) > 0:
        solution[num_primary_nodes + num_secondary_nodes + num_tertiary_nodes:num_primary_nodes + num_secondary_nodes + num_tertiary_nodes + num_primary_nodes] = np.random.choice(secondary_selected_indices, num_primary_nodes)
    if len(tertiary_selected_indices) > 0:
        solution[num_primary_nodes + num_secondary_nodes + num_tertiary_nodes + num_primary_nodes:] = np.random.choice(tertiary_selected_indices, num_secondary_nodes)

# 初始化种群
population = generate_initial_population()

most_best_cost = large_penalty
most_best_individual = 0

# 遗传算法主循环
for generation in range(num_generations):
    # 计算适应度
    fitness_values = np.array([fitness(individual) for individual in population])

    # 进行适应度值转换
    max_fitness = np.min(fitness_values)
    transformed_fitness_values = max_fitness - fitness_values

    # 避免除零错误和负值概率
    if np.sum(transformed_fitness_values) == 0:
        probabilities = np.ones(num_individuals) / num_individuals
    else:
        probabilities = transformed_fitness_values / np.sum(transformed_fitness_values)

    # 选择
    selected_indices = np.random.choice(np.arange(num_individuals), size=num_individuals, p=probabilities)
    selected_population = population[selected_indices]

    # 交叉
    for i in range(0, num_individuals, 2):
        if np.random.rand() < crossover_rate:
            crossover_point = np.random.randint(1, len(population[0]))
            selected_population[i, crossover_point:], selected_population[i + 1, crossover_point:] = \
                selected_population[i + 1, crossover_point:], selected_population[i, crossover_point:]

    # 变异
    for individual in selected_population:
        if np.random.rand() < mutation_rate:
            mutation_point = np.random.randint(len(population[0]))
            if mutation_point < num_primary_nodes:
                individual[mutation_point] = 1  # 确保一级节点全选
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
        fix_solution(individual)  # 修正路径选择

    # 更新种群
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

# 输出最佳解
most_best_cost = -test_fitness(most_best_individual)
print("最佳节点选择方案：", most_best_individual)
print("总成本：", most_best_cost)
