from typing import List, Callable
from .datasets import BaseOT
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

SAVE = 'save'
RESULTS = 'results'
PLOTS = 'plots'

class OTsolver:

    # specifying colors: https://matplotlib.org/stable/users/explain/colors/colors.html
    # specifying line styles: https://matplotlib.org/stable/gallery/lines_bars_and_markers/linestyles.html
    def __init__(
        self,
        method: Callable,
        method_name: str,
        **kwargs,
    ):
        self.method = method
        self.method_name = method_name
        self.color = kwargs.pop('color', 'b')
        self.linestyle = kwargs.pop('linestyle', 'solid')
        self.linewidth = kwargs.pop('linewidth', 2.5)
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
            'linestyle': self.linestyle, # line style
            'linewidth': self.linewidth, # line width
            'color': self.color, # color
        }

class OTtask:

    def __init__(
        self,
        problem: BaseOT,
        solvers: List[OTsolver],
        results_path: str = os.path.join(SAVE, RESULTS),
        plots_path: str = os.path.join(SAVE, PLOTS),
    ):
        self.problem = problem
        self.solvers = solvers
        self.results_path = results_path
        self.plots_path = plots_path
    
    def run(
        self,
        save_results: bool = True,
        force_rerun: bool = True,
    ) -> dict:
        """Return a dictionary:
        {
            'problem_description': 'str',
            'probelm_solutions': [
                {
                    'method_name': str,
                    'obj_vals': np.ndarray,
                    'run_times': np.ndarray,
                    'log10_mar_errs': np.ndarray,
                    'iterations': np.ndarray,
                    'linestyle': str | tuple,
                    'linewidth': float,
                    'color': str | tuple,
                },
                ...
            ],
        }
        """
        # initialize the results
        results = {
            "problem_description": self.problem.description,
            "problem_solutions": [],
        }
        # check if the results are already saved
        if save_results:
            if not os.path.exists(self.results_path):
                os.makedirs(self.results_path)
        # get the path to save the results
        results_filepath = os.path.join(self.results_path,
                                        self.problem.description + '.pkl')

        print(f"OT problem: {self.problem.description}")
        # check if the results are already saved
        if os.path.exists(results_filepath) and not force_rerun:
            with open(results_filepath, 'rb') as f:
                results = pickle.load(f)
            print(f"Results loaded from {results_filepath}")
        else:
            # run the solvers
            for solver in tqdm(self.solvers, desc='Solving the OT problems'):
                results['problem_solutions'].append(solver.solve(self.problem))
            if save_results:
                with open(results_filepath, 'wb') as f:
                    pickle.dump(results, f)
                print(f"Results saved to {results_filepath}")
        return results
    
    def plot_for_problem(
        self,
        x_key: str = 'iterations',
        x_label: str = 'Iteration Number',
        y_key: str = 'log10_mar_errs',
        y_label: str = 'Log10 Gradient Norm',
        force_rerun: bool = True,
        selected_methods: List[str] = None,
    ) -> None:
        """Plot a single plot of a single problem with multiple methods"""
        
        if not os.path.exists(self.plots_path):
            os.makedirs(self.plots_path)
        
        # set the plot configurations
        plt.rcParams.update({
            'font.size': 20,
            'axes.labelsize': 20,
            'axes.titlesize': 20,
            'axes.titlepad': 10,
            'legend.fontsize': 15,
            'xtick.labelsize': 20,
            'ytick.labelsize': 20,
            # 'savefig.format': 'pdf', # https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.savefig.html
        })
        
        plt.figure(figsize=(10, 6))

        # set the x-axis limits
        # if x_key == 'iterations':
        #     plt.xlim(0, 300)
        # elif x_key == 'run_times':
        #     plt.xlim(0, 3)
        # else:
        #     pass

        # title
        # title = "\n".join(self.problem.description.split('[')).split(']')[0]
        title = self.problem.title
        plt.title(title)
        # labels
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        # grid
        plt.grid(True)
        # plot the solutions
        results = self.run(save_results=True, force_rerun=force_rerun)
        for solution in results["problem_solutions"]:
            # check if the method is selected
            if selected_methods is not None:
                if solution['method_name'] not in selected_methods:
                    continue
            # plot the solution
            x = solution[x_key]
            y = solution[y_key]
            plt.plot(
                x, y,
                color=solution['color'],
                linestyle=solution['linestyle'],
                linewidth=solution['linewidth'],
                label=solution['method_name'],
            )
        # legend
        plt.legend(loc='upper right')
        # save the plot
        savefig_path = os.path.join(self.plots_path,
                                   f"({x_key}){self.problem.description}.pdf")
        plt.savefig(savefig_path, bbox_inches='tight')
        plt.close()
        print(f"Plot saved to {savefig_path}")