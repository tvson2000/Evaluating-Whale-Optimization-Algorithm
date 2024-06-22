import numpy as np
import random

class WhaleOptimizationGeneticAlgorithm():
    def __init__(self, problem, nsols, b, a, a_step, maximize=False, budget=None):
        self.problem = problem
        self.budget = 20000#budget if budget is not None else 50 * problem.meta_data.n_variables**2 #This is the max number of generations
        self._constraints = list(zip(problem.bounds.lb, problem.bounds.ub))
        self._sols = self._init_solutions(nsols) 
        self._b = b
        self._a = a
        self._a_step = a_step
        self._maximize = maximize
        self._best_solutions = []
        self._er = 0.05 #elitism rate
        self._pc = 0.8 #probability crossover
        self._pm = 0.05 #probability mutation
        self._global_best = None
        self._local_best = None
        
    def get_solutions(self):
        return self._sols
                  
    
    def optimize(self):
        ranked_sol = self._rank_solutions()
        self._local_best = ranked_sol[0]
        self._global_best = ranked_sol[0]
        
        while self.budget > 0:
            nr_elites = int(len(ranked_sol) * self._er)
            elites = ranked_sol[:nr_elites]
            
            new_sols = []
            for s in ranked_sol[nr_elites:]:
                self.budget -= 1 
                if np.random.uniform(0.0, 1.0) > 0.5:
                    A = self._compute_A()
                    norm_A = np.linalg.norm(A)
                    if norm_A < 1.0:
                        new_s = self._encircle(s, self._local_best, A)
                    else:
                        random_sol = self._sols[np.random.randint(self._sols.shape[0])]
                        new_s = self._search(s, random_sol, A)
                else:
                    new_s = self._attack(s, self._local_best)
                new_s = self._constrain_solution(new_s)

                new_sols.append(new_s)
                
            binary_sols = self._encode(new_sols)
            
            binary_sols = self._wheel_of_fortune(binary_sols, new_sols)
            binary_sols = self._one_point_crossover(binary_sols)
            binary_sols = self._bit_mutation(binary_sols)
            
            decoded_sols = self._decode(binary_sols)
            self._sols = np.array(elites + decoded_sols)
            ranked_sol = self._rank_solutions()
            self._local_best = ranked_sol[0]
            
            if self.problem(self._local_best) < self.problem(self._global_best):
                self._global_best = self._local_best
                        
                     
            if(self._a > 0):
                self._a -= self._a_step
            
    
    def _wheel_of_fortune(self, binary_sols, new_sols):
        fitnesses = np.array([self.problem(s) for s in new_sols])

        fitnesses = fitnesses - np.min(fitnesses) + 1
        epsilon = 1e-10 #this prevents us from dividing by 0
        inverted_fitness = 1.0 / (fitnesses + epsilon)

        probabilities = inverted_fitness / np.sum(inverted_fitness)  

       
        selected_indices = np.random.choice(len(binary_sols), size=len(new_sols), p=probabilities)
        return [binary_sols[i] for i in selected_indices]

    
    def _crossover(self, a, b, index):
        return b[:index] + a[index:], a[:index] + b[index:]
    
    def _one_point_crossover(self, binary_sols):
        sols = []
        for i in range(0, len(binary_sols), 2):
            if i + 1 < len(binary_sols):
                parent1 = binary_sols[i]
                parent2 = binary_sols[i + 1]
                if np.random.rand() < self._pc:
                    index = random.randint(1, len(parent1) -1)
                    child1, child2 = self._crossover(parent1, parent2, index)
                    sols.extend([child1, child2])
                else:
                    sols.extend([parent1, parent2])
            else:
                sols.append(binary_sols[i])
        return sols
    
    def _mutate(self, bit):
        if np.random.rand() < self._pm:
            return '1' if bit == '0' else '0'
        return bit 
    
    def _bit_mutation(self, binary_sols):
        sols = []
        for sol in binary_sols:
            mutated = ''.join([self._mutate(bit) for bit in sol])
            sols.append(mutated)
        return sols
    
                    
    def _encode(self, new_sols):
        binary_sols = []
        for sol in new_sols:
            binary_s = ""
            for c, s in zip(self._constraints, sol):
                c_min = c[0]
                c_max = c[1]
                normalized = (s - c_min) / (c_max - c_min)
                scaled = int(normalized * (2 ** 32 - 1)) 
                binary_s += format(scaled, '032b')
            binary_sols.append(binary_s)
        return binary_sols

    def _decode(self, binary_sols):
        sols = []
        for binary_s in binary_sols:
            sol = []
            for i in range(len(self._constraints)):
                decimal = int(binary_s[i * 32:(i + 1) * 32], 2)
                c_min, c_max = self._constraints[i]
                denormalized = decimal / (2 ** 32 - 1) * (c_max - c_min) + c_min
                sol.append(denormalized)
            sols.append(sol)
        return sols 

    def _init_solutions(self, nsols):
        sols = []
        for c in self._constraints:
            sols.append(np.random.uniform(c[0], c[1], size=nsols))
                                                                            
        sols = np.stack(sols, axis=-1)
        return sols

    def _constrain_solution(self, sol):
        constrain_s = []
        for c, s in zip(self._constraints, sol):
            if c[0] > s:
                s = c[0]
            elif c[1] < s:
                s = c[1]
            constrain_s.append(s)
        return constrain_s

    def _rank_solutions(self):
        fitnesses = [self.problem(s) for s in self._sols]
        sol_fitness = list(zip(fitnesses, self._sols))
        ranked_sol = list(sorted(sol_fitness, key=lambda x:x[0], reverse=self._maximize))
        self._best_solutions.append(ranked_sol[0])
        return [ s[1] for s in ranked_sol] 

    def print_best_solutions(self):
        print('generation best solution history')
        print('([fitness], [solution])')
        for s in self._best_solutions:
            print(s)
        print('\n')
        print('best solution')
        print('([fitness], [solution])')
        print(sorted(self._best_solutions, key=lambda x:x[0], reverse=self._maximize)[0])

    def _compute_A(self):
        r = np.random.uniform(0.0, 1.0, size=self._sols[0].shape)
        return (2.0*np.multiply(self._a, r))-self._a

    def _compute_C(self):
        return 2.0*np.random.uniform(0.0, 1.0, size=self._sols[0].shape)
                                                                 
    def _encircle(self, sol, best_sol, A):
        D = self._encircle_D(sol, best_sol)
        return best_sol - np.multiply(A, D)
                                                                 
    def _encircle_D(self, sol, best_sol):
        C = self._compute_C()
        return np.multiply(C, best_sol) - sol 

    def _search(self, sol, rand_sol, A):
        D = self._search_D(sol, rand_sol)
        return rand_sol - np.multiply(A, D)
  
    def _search_D(self, sol, rand_sol):
        C = self._compute_C()
        return np.multiply(C, rand_sol) - sol 
    
    def _attack(self, sol, best_sol):
        D = best_sol - sol 
        L = np.random.uniform(-1.0, 1.0, size=len(sol))
        spiral_component = np.multiply(np.exp(self._b * L) * np.cos(2 * np.pi * L), D)
        return best_sol + spiral_component