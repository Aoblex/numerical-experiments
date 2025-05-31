from typing import List, Callable
from .datasets import BaseOT
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from datetime import datetime

SAVE = 'save'
PICKLE = 'pickle'
PLOTS = 'plots'
LOGS = 'logs'

# set the plot configurations
sns.set_theme(style="whitegrid")
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

def setup_logging(log_file=None):
    """Configure logging to output to both file and console"""
    log_dir = os.path.join(SAVE, LOGS)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    if log_file is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = os.path.join(log_dir, f"ot_experiments_{timestamp}.log")
    
    # Configure logging to output to both file and console
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger()

# Create logger
logger = setup_logging()

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
        task_name: str,
        pickle_path: str = os.path.join(SAVE, PICKLE),
        plots_path: str = os.path.join(SAVE, PLOTS),
    ):
        self.problem = problem
        self.solvers = solvers
        self.task_name = task_name
        self.pickle_path = pickle_path
        self.plots_path = plots_path
    
    def run(
        self,
        save_pickle: bool = True,
        force_rerun: bool = True,
    ) -> dict:
        """Return a dictionary:
        {
            method_name1 (str): result1 (dict),
            method_name2 (str): result2 (dict),
            ...
        }
        """
        logger.info(f"Current dataset: {self.problem.data_name}")
        logger.info(f"Current settings: {self.problem.description}")
        problem_pickle_path = os.path.join(
            self.pickle_path,
            self.problem.data_name,
            self.problem.description
        )
        if not os.path.exists(problem_pickle_path):
            os.makedirs(problem_pickle_path)

        results = {}
        for solver in self.solvers:
            logger.info(f"Current method: '{solver.method_name}'")
            pickle_path = os.path.join(problem_pickle_path, f"{solver.method_name}.pkl")
            if os.path.exists(pickle_path) and not force_rerun:
                logger.info(f"Cached pickle found at: {pickle_path}")
                with open(pickle_path, 'rb') as f:
                    results[solver.method_name] = pickle.load(f)
                continue
            
            logger.info(f"{solver.method_name} is running...")
            results[solver.method_name] = solver.solve(self.problem)
            logger.info(f"{solver.method_name} iterations: {len(results[solver.method_name]['iterations'])}")
            logger.info(f"{solver.method_name} run time: {results[solver.method_name]['run_times'][-1]} seconds")

            if save_pickle:
                with open(pickle_path, 'wb') as f:
                    pickle.dump(results[solver.method_name], f)
                logger.info(f"pickle saved to: {pickle_path}")
        
        logger.info(f"Task completed for {self.problem.data_name}: {self.problem.description}")
        return results 

    
    def plot_for_problem(
        self,
        x_key: str = 'iterations',
        x_label: str = 'Iteration Number',
        y_key: str = 'log10_mar_errs',
        y_label: str = 'Log10 Gradient Norm',
        force_rerun: bool = True,
    ) -> None:
        """Plot a single plot of a single problem with multiple methods"""
        
        logger.info(f"{y_key} vs {x_key} plot for {self.problem.data_name}: {self.problem.description}")
        
        if not os.path.exists(self.plots_path):
            os.makedirs(self.plots_path)
        
        title = self.problem.title
        plt.figure(figsize=(10, 6))
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.grid(True)

        # plot the solutions
        results = self.run(save_pickle=True, force_rerun=force_rerun)
        for method_name, result in results.items():
            # plot the solution
            x = result[x_key]
            y = result[y_key]
            plt.plot(
                x, y,
                color=result['color'],
                linestyle=result['linestyle'],
                linewidth=result['linewidth'],
                label=method_name,
            )
        # legend
        plt.legend(loc='upper right')
        # save the plot
        savefig_path = os.path.join(self.plots_path,
                                    self.task_name,
                                    x_key,
                                    f"{self.problem.description}.pdf")
        os.makedirs(os.path.dirname(savefig_path), exist_ok=True)
        plt.savefig(savefig_path, bbox_inches='tight')
        plt.close()
        logger.info(f"Plot saved to: {savefig_path}")