import argparse
import math
from utils import get_solvers
from ot.datasets import Synthetic2OT
from ot.experiments import OTtask

# Parse command line arguments
parser = argparse.ArgumentParser(description='Run Synthetic2 OT experiments')
parser.add_argument('--task-name', type=str, default="Synthetic II",
                    help='Name of current OT task, also the name of plot folder (default: Synthetic II)')
parser.add_argument('--reg', nargs='+', type=float, default=[0.001],
                    help='List of regularization parameters (default: [0.001])')
parser.add_argument('--size', nargs='+', type=int, default=[1000],
                    help='List of problem sizes (default: [1000])')
parser.add_argument('--max-iter', type=int, default=500,
                    help='Maximum number of iterations (default: 500)')
parser.add_argument('--tol', type=float, default=1e-6,
                    help='Tolerance for convergence (default: 1e-6)')
parser.add_argument('--methods', nargs='+', type=str, default=None,
                    help='List of methods to use, separated by space (default: suitable methods for each size)')
parser.add_argument('--plot-xlim', type=float, default=math.inf,
                    help='The x-axis of the plot will be limited at this value for better visualization (default: math.inf)')
parser.add_argument('--force-rerun', action='store_true',
                    help='Force rerun of the experiments even if results exist')
args = parser.parse_args()

task_name = args.task_name
reg_list = args.reg
problem_sizes = args.size
max_iter, tol = args.max_iter, args.tol
plot_xlim = args.plot_xlim
force_rerun = args.force_rerun
synthetic2_methods = args.methods

for reg in reg_list:
    for problem_size in problem_sizes:
        
        # Solving the Synthetic2 OT problem
        synthetic2_ot_problem = Synthetic2OT(
            n=problem_size,
            m=problem_size,
            reg=reg,
        )
        synthetic2_solvers = get_solvers(reg=reg, max_iter=max_iter, tol=tol,
                                         selected=synthetic2_methods)
        synthetic2_task = OTtask(
            task_name=task_name,
            problem=synthetic2_ot_problem,
            solvers=synthetic2_solvers
        )
        
        synthetic2_task.plot_for_problem(x_key='iterations',
                                         x_label='Iteration Number',
                                         y_label='Log10 Gradient Norm',
                                         force_rerun=force_rerun)
        synthetic2_task.plot_for_problem(x_key='run_times',
                                         x_label='Run time(seconds)',
                                         y_label='Log10 Gradient Norm',
                                         x_lim=plot_xlim,
                                         force_rerun=force_rerun)

