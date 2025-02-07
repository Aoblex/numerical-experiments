from typing import List, Callable
from .datasets import BaseOT
import os
import pickle
import numpy as np
from tqdm import tqdm

SAVE = 'save'
RESULTS = 'results'
PLOTS = 'plots'

class OTsolver:

    def __init__(
        self,
        method: Callable,
        method_name: str,
        **kwargs,
    ):
        self.method = method
        self.method_name = method_name
        self.kwargs = kwargs
    
    def solve(self, problem: BaseOT):
        M, a, b = problem.M, problem.a, problem.b
        result = self.method(M, a, b, **self.kwargs)
        return {
            'method_name': self.method_name,
            'obj_vals': result.obj_vals, # objective values of the dual problem
            'run_times': [run_time/1000 for run_time in result.run_times], # cumulative run times in seconds (converted from milliseconds)
            'log10_mar_errs': np.log10(result.mar_errs), # log10 of the marginals error, which is also the gradient norm
            'iterations': np.arange(len(result.obj_vals)), # number of iterations
        }

class OTtask:

    def __init__(
        self,
        problems: List[BaseOT],
        solvers: List[OTsolver],
    ):
        self.problems = problems
        self.solvers = solvers
    
    def run(
        self,
        save_results: bool = True,
        save_path: str = os.path.join(SAVE, RESULTS),
        force_rerun: bool = False,
    ) -> List[dict]:
        """Return a list of dictionary:
        [
            {
                'problem_description': [
                    {
                        'method_name': str,
                        'obj_vals': np.ndarray,
                        'run_times': np.ndarray,
                        'log10_mar_errs': np.ndarray,
                        'iterations': np.ndarray,
                    },
                    ...
                ],
            },
            ...
        ]
        """
        results = [] # each element in results draws a single plot

        if save_results:
            if not os.path.exists(save_path):
                os.makedirs(save_path)

        for problem in self.problems:
            current_results_path = os.path.join(save_path, problem.description + '.pkl')

            print(f"Current OT: {problem.description}")

            # check if the results are already saved
            if os.path.exists(current_results_path) and not force_rerun:
                with open(current_results_path, 'rb') as f:
                    current_results = pickle.load(f)
                print(f"Results loaded from {current_results_path}")
            else:
                current_results = []
                for solver in tqdm(self.solvers, desc='Solvers'):
                    current_results.append(solver.solve(problem))
                if save_results:
                    with open(current_results_path, 'wb') as f:
                        pickle.dump(current_results, f)
                    print(f"Results saved to {current_results_path}")
                print(f"Results of {problem.description} are ready")
            results.append({problem.description: current_results})
        return results
    
    def plot(
        self,
        plot_path: str = os.path.join(SAVE, PLOTS),
    ) -> None:

        if not os.path.exists(plot_path):
            os.makedirs(plot_path)
