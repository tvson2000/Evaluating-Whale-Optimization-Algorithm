import sys
import numpy as np

from ioh import get_problem, logger, ProblemClass, Experiment
from whale_optimization import WhaleOptimization
from whale_optimization_levy_flight import WhaleOptimizationLevyFlight
from whale_optimization_genetic_algorithm import WhaleOptimizationGeneticAlgorithm



def WhaleOptimizationRunner(problem, budget = None):
    nsols = 30
    b = 1.0
    a = 2.0
    a_step = 0.0666667
    maximize = False
    woa = WhaleOptimization(problem=problem, nsols=nsols, b=b, a=a, a_step=a_step, maximize=maximize, budget=budget)
    print("hier beginnen we")
    woa.optimize()
    woa.print_best_solutions()

def WhaleOptimizationLevyFlightRunner(problem, budget = None):
    nsols = 30
    b = 1.0
    a = 2.0
    a_step = 0.0666667
    maximize = False
    woa = WhaleOptimizationLevyFlight(problem=problem, nsols=nsols, b=b, a=a, a_step=a_step, maximize=maximize, budget=budget)
    print("hier beginnen we")
    woa.optimize()
    woa.print_best_solutions()
    
def WhaleOptimizationGeneticAlgorithmRunner(problem, budget = None):
    nsols = 30
    b = 1.0
    a = 2.0
    a_step = 0.0666667
    maximize = False
    woa = WhaleOptimizationGeneticAlgorithm(problem=problem, nsols=nsols, b=b, a=a, a_step=a_step, maximize=maximize, budget=budget)
    print("hier beginnen we")
    woa.optimize()
    woa.print_best_solutions()

    
exp = Experiment(algorithm=WhaleOptimizationRunner, fids=[1, 2, 3, 4, 5 , 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24], iids=[1],
                 dims=[2, 5, 10, 20, 40], reps=50, problem_class=ProblemClass.BBOB,
                 output_directory="WOA-final", folder_name="WOA-final",
                 algorithm_name="WOA", algorithm_info="", store_positions=True, 
                 merge_output=True, zip_output=True, remove_data=True)


exp2 = Experiment(algorithm=WhaleOptimizationLevyFlightRunner, fids=[1, 2, 3, 4, 5 , 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24], iids=[1],
                 dims=[2, 5, 10, 20, 40], reps=50, problem_class=ProblemClass.BBOB,
                 output_directory="ILWOA", folder_name="ILWOA",
                 algorithm_name="ILWOA", algorithm_info="", store_positions=True, 
                 merge_output=True, zip_output=True, remove_data=True)

exp3 = Experiment(algorithm=WhaleOptimizationGeneticAlgorithmRunner, fids=[1, 2, 3, 4, 5 , 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24], iids=[1],
                 dims=[40], reps=50, problem_class=ProblemClass.BBOB,
                 output_directory="WOA-GA-40", folder_name="WOA-GA-40",
                 algorithm_name="WOA-GA", algorithm_info="", store_positions=True, 
                 merge_output=True, zip_output=True, remove_data=True)



def main():
    print("Select the experiment to run: ")
    print("1: Original Whale Optimization")
    print("2: Whale Optimization with Levy Flight")
    print("3: Whale Optimization integrated with Genetic Algorithm")
    choice = input("Enter your choice (1, 2 or 3): ")
    
    if choice == "1":
        exp()
    elif choice == "2":
        exp2()
    elif choice == "3":
        exp3()
    else:
        print("Invalid choice")

if __name__ == "__main__":
    main()