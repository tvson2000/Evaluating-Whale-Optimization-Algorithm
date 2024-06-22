import numpy as np

class WhaleOptimization():
    def __init__(self, problem, nsols, b, a, a_step, maximize=False, budget=None):
        self.problem = problem
        self.budget = 20000 #budget if budget is not None else 50 * problem.meta_data.n_variables**2
        self._constraints = list(zip(problem.bounds.lb, problem.bounds.ub))
        self._sols = self._init_solutions(nsols) 
        self._b = b
        self._a = a
        self._a_step = a_step
        self._maximize = maximize
        self._best_solutions = []
        self.total_budget = self.budget
        
    def get_solutions(self):
        return self._sols
                  
    
    def optimize(self):
        while self.budget > 0:
            ranked_sol = self._rank_solutions()
            best_sol = ranked_sol[0]  
            new_sols = [best_sol]

            for s in ranked_sol[1:]:
        
                if self.budget <= 0: 
                    break
                self.budget -= 1  
                new_s = None
                if np.random.uniform(0.0, 1.0) > 0.5:

                    A = self._compute_A()

                    norm_A = np.linalg.norm(A)

                    if norm_A < 1.0:

                        new_s = self._encircle(s, best_sol, A)

                    else:

                        random_sol = self._sols[np.random.randint(self._sols.shape[0])]
                        new_s = self._search(s, random_sol, A)

                else:
                    new_s = self._attack(s, best_sol)
                    
                new_s = self._constrain_solution(new_s)

                new_sols.append(new_s)
                if self.budget <= 0: 
                    break
            self._sols = np.stack(new_sols)
            
            if(self._a > 0):
                self._a -= self._a_step
           

    def _init_solutions(self, nsols):
        sols = []
        for c in self._constraints:
            sols.append(np.random.uniform(c[0], c[1], size=nsols))
                                                                            
        sols = np.stack(sols, axis=-1)
        return sols
    
    def _constrain_solution(self, sol):
        """ensure solutions are valid wrt to constraints"""
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