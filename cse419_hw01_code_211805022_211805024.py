import random
import numpy as np
import time

# 211805022 - Berke PEKER
# 211805024 - Kemal Eren KAYA
# CSE 419 ARTIFICIAL INTELLIGENCE ASSIGNMENT#1
 
# Main function to run the Cutting Stock Problem program
def main():
    while True:
        print("\n📦 Cutting Stock Problem Input")

        # Ask the user to input the roll length
        roll_input = input("➤ Enter stock roll length (e.g., 100) or type 'quit' to exit: ").strip()
        if roll_input.lower() == 'quit':
            print("👋 Exiting the program. Goodbye!")
            break

        try:
            ROLL_LENGTH = int(roll_input)
        except ValueError:
            print("❌ Please enter a valid number.")
            continue

        DEMAND = {}
        while True:
            # Ask the user to input piece lengths and their quantities
            demand_input = input("➤ Enter piece lengths and quantities (e.g., 10-5,15-3): ").strip()
            DEMAND.clear()
            entries = demand_input.split(',')
            all_valid = True

            for part in entries:
                part = part.strip()
                try:
                    length, count = map(int, part.split('-'))
                    DEMAND[length] = count
                except:
                    print(f"⚠️ Invalid format: {part}")
                    all_valid = False
                    break

            # Check if all inputs are valid
            if all_valid:
                break
            else:
                print("🔁 Please re-enter the piece lengths and quantities in correct format.")

        while True:
            # Ask the user to choose the algorithms to use
            agents_input = input("➤ Choose algorithms (SA,HC,GA comma-separated): ").strip()
            agents_raw = [a.strip().upper() for a in agents_input.split(',') if a.strip()]
            valid_choices = {'SA', 'HC', 'GA'}
            AGENTS = [a for a in agents_raw if a in valid_choices]
            invalid_agents = [a for a in agents_raw if a not in valid_choices]

            # Validate algorithm choices
            if invalid_agents:
                print(f"⚠️ Invalid algorithms: {', '.join(invalid_agents)}")
            if not AGENTS:
                print("⚠️ Please enter at least one valid algorithm: SA, HC, or GA.")
            else:
                break

        # Run all selected agents
        run_all_agents(ROLL_LENGTH, DEMAND, AGENTS)

# Simulated Annealing algorithm implementation
def sa_run(lengths, demand, roll_length, max_iter=1000, temp_init=100, alpha=0.95, init_solution=None):
    total_pieces = []
    for l, count in demand.items():
        total_pieces += [l] * count

    # Generate an initial solution (either random or given)
    def get_initial_solution():
        if init_solution:
            return init_solution.copy()
        random.shuffle(total_pieces)
        return total_pieces.copy()

    # Evaluate the solution by calculating the waste
    def evaluate_solution(solution):
        rolls = []
        current = []
        for piece in solution:
            if sum(current) + piece <= roll_length:
                current.append(piece)
            else:
                rolls.append(current)
                current = [piece]
        if current:
            rolls.append(current)
        waste = sum([roll_length - sum(r) for r in rolls])
        return waste, rolls

    # Generate a neighbor solution by swapping two random pieces
    def neighbor(solution):
        a, b = random.sample(range(len(solution)), 2)
        sol = solution.copy()
        sol[a], sol[b] = sol[b], sol[a]
        return sol

    current = get_initial_solution()
    current_score, _ = evaluate_solution(current)
    best = current
    best_score = current_score
    T = temp_init

    # Perform Simulated Annealing for a number of iterations
    for _ in range(max_iter):
        candidate = neighbor(current)
        candidate_score, _ = evaluate_solution(candidate)
        delta = candidate_score - current_score
        if delta < 0 or random.random() < np.exp(-delta / T):
            current = candidate
            current_score = candidate_score
            if current_score < best_score:
                best = current
                best_score = current_score
        T *= alpha

    _, final_rolls = evaluate_solution(best)
    return final_rolls, best_score, best

# Hill Climbing algorithm implementation
def hc_run(lengths, demand, roll_length, max_iter=1000, init_solution=None):
    total_pieces = []
    for l, count in demand.items():
        total_pieces += [l] * count

    # Generate an initial solution (either random or given)
    def get_initial_solution():
        if init_solution:
            return init_solution.copy()
        random.shuffle(total_pieces)
        return total_pieces.copy()

    # Evaluate the solution by calculating the waste
    def evaluate_solution(solution):
        rolls = []
        current = []
        for piece in solution:
            if sum(current) + piece <= roll_length:
                current.append(piece)
            else:
                rolls.append(current)
                current = [piece]
        if current:
            rolls.append(current)
        waste = sum([roll_length - sum(r) for r in rolls])
        return waste, rolls

    # Generate a neighbor solution by swapping two random pieces
    def neighbor(solution):
        a, b = random.sample(range(len(solution)), 2)
        sol = solution.copy()
        sol[a], sol[b] = sol[b], sol[a]
        return sol

    current = get_initial_solution()
    current_score, _ = evaluate_solution(current)

    # Perform Hill Climbing for a number of iterations
    for _ in range(max_iter):
        candidate = neighbor(current)
        candidate_score, _ = evaluate_solution(candidate)
        if candidate_score < current_score:
            current = candidate
            current_score = candidate_score

    _, final_rolls = evaluate_solution(current)
    return final_rolls, current_score, current

# Genetic Algorithm implementation
def ga_run(lengths, demand, roll_length, pop_size=30, generations=100, mutation_rate=0.1, init_solution=None):
    total_pieces = []
    for l, count in demand.items():
        total_pieces += [l] * count

    num_genes = len(total_pieces)

    # Decode the individual genetic representation to a real solution
    def decode(individual):
        sequence = [total_pieces[i] for i in individual]
        solution = []
        current_roll = []
        for piece in sequence:
            if sum(current_roll) + piece <= roll_length:
                current_roll.append(piece)
            else:
                solution.append(current_roll)
                current_roll = [piece]
        if current_roll:
            solution.append(current_roll)
        return solution

    # Calculate the fitness score of an individual
    def fitness(individual):
        sequence = [total_pieces[i] for i in individual]
        piece_counts = {l: 0 for l in demand}
        for l in sequence:
            if l in piece_counts:
                piece_counts[l] += 1

        penalty = 0
        for l in demand:
            diff = piece_counts[l] - demand[l]
            if diff < 0:
                penalty += abs(diff) * 1000
            elif diff > 0:
                penalty += diff * 500

        solution = decode(individual)
        waste = sum([roll_length - sum(roll) for roll in solution])
        return waste + penalty

    # Crossover between two individuals
    def crossover(p1, p2):
        point = random.randint(1, num_genes - 1)
        child = p1[:point] + [x for x in p2 if x not in p1[:point]]
        return child

    # Mutate an individual by swapping two random genes
    def mutate(individual):
        if random.random() < mutation_rate:
            i, j = random.sample(range(num_genes), 2)
            individual[i], individual[j] = individual[j], individual[i]
        return individual

    if init_solution:
        # init_solution here represents the total_pieces list
        base = list(range(num_genes))
        initial_population = [random.sample(base, len(base)) for _ in range(pop_size)]
    else:
        initial_population = [random.sample(range(num_genes), num_genes) for _ in range(pop_size)]

    population = initial_population

    # Perform Genetic Algorithm for a number of generations
    for _ in range(generations):
        population.sort(key=fitness)
        next_gen = population[:5]
        while len(next_gen) < pop_size:
            p1, p2 = random.sample(population[:15], 2)
            child = mutate(crossover(p1, p2))
            next_gen.append(child)
        population = next_gen

    best = min(population, key=fitness)
    best_solution = decode(best)
    best_waste = fitness(best)
    return best_solution, best_waste, best

# Function to run all selected algorithms (SA, HC, GA)
def run_all_agents(ROLL_LENGTH, DEMAND, AGENTS):
    results = []
    lengths = list(DEMAND.keys())
    raw_solutions = {}
    summary = []

    print("\n=== Initial Independent Agent Runs ===")
    for agent in AGENTS:
        start_time = time.time()
        if agent == 'SA':
            sol, waste, encoding = sa_run(lengths, DEMAND, ROLL_LENGTH)
        elif agent == 'HC':
            sol, waste, encoding = hc_run(lengths, DEMAND, ROLL_LENGTH)
        elif agent == 'GA':
            sol, waste, encoding = ga_run(lengths, DEMAND, ROLL_LENGTH)
        else:
            continue
        duration = time.time() - start_time
        print(f"\n{agent} → Waste: {waste} | Runtime: {duration:.3f} sec | Rolls Used: {len(sol)}")
        for idx, roll in enumerate(sol, 1):
            print(f"Roll {idx}: {roll} → Used: {sum(roll)} / Waste: {ROLL_LENGTH - sum(roll)}")
        results.append((sol, waste, agent, encoding, duration, len(sol)))
        raw_solutions[agent] = encoding

    best_encoding = min(results, key=lambda x: x[1])[3]
    print("\n=== Collaborative Refinement Phase ===")

    final_results = []
    for agent in AGENTS:
        start_time = time.time()
        if agent == 'SA':
            sol, waste, _ = sa_run(lengths, DEMAND, ROLL_LENGTH, init_solution=best_encoding)
        elif agent == 'HC':
            sol, waste, _ = hc_run(lengths, DEMAND, ROLL_LENGTH, init_solution=best_encoding)
        elif agent == 'GA':
            sol, waste, _ = ga_run(lengths, DEMAND, ROLL_LENGTH, init_solution=best_encoding)
        else:
            continue
        duration = time.time() - start_time
        print(f"\n{agent} (refined) → Waste: {waste} | Runtime: {duration:.3f} sec | Rolls Used: {len(sol)}")
        for idx, roll in enumerate(sol, 1):
            print(f"Roll {idx}: {roll} → Used: {sum(roll)} / Waste: {ROLL_LENGTH - sum(roll)}")
        final_results.append((sol, waste, agent, duration, len(sol)))

    min_waste = min(final_results, key=lambda x: x[1])[1]
    best_algorithms = [r for r in final_results if r[1] == min_waste]

    print("\n=== Best Collaborative Solution(s) ===")
    for _, waste, algo, duration, rolls in best_algorithms:
        print(f"Algorithm: {algo} | Waste: {waste} meters | Runtime: {duration:.3f} sec | Rolls Used: {rolls}")

# Entry point for the program
if __name__ == '__main__':
    main()
