# Simple simulated annealing algorithm
# Adapted from http://katrinaeg.com/simulated-annealing.html for learning purposes

import math, random

# Acceptance threshold function: determines how likely we are to accept a new solution,
# given its cost, the current solution's cost, and the current temperature
#
# NOTE: This is supposedly a "standard" temperature function:
#       - better solutions are always accepted
#       - likelihood of accepting worse solutions approaches zero
#         as temperature decreases and as solutions improve
def accept(old_cost, new_cost, temperature):
    return math.e ** ((new_cost - old_cost) / temperature)

# Generates a random solution somewhere close to the current one
def neighbor(solution):
    return solution + random.uniform(-1, 1)

# Determines the overall "cost" (fitness) of a solution, i.e. how close to optimal it is
def fitness(solution):
    return -1 * (solution - 4) ** 2  # this is just a dummy quadratic function

# All the moving parts, put together
# initial_solution: can be completely random, has little bearing on final result
# alpha: factor by which the temperature will change with each iteration;
#        0.8 (fast) - 0.99 (slow) is typical
def simulated_annealing(initial_solution, alpha, random_neighbor, fitness, threshold=accept):
    temperature = 1.0
    min_temperature = 0.00001
    
    current_solution = initial_solution
    current_solution_cost = fitness(current_solution)
    while temperature > min_temperature:
        for i in range(100):
            new_solution = random_neighbor(current_solution)
            new_solution_cost = fitness(new_solution)
            prob = threshold(current_solution_cost, new_solution_cost, temperature)
            if prob > random.random():
                current_solution = new_solution
                current_solution_cost = new_solution_cost
                    
        temperature *= alpha
    
    return current_solution, current_solution_cost
