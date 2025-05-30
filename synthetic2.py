import argparse
from utils import get_solvers
from ot.datasets import Synthetic2OT
from ot.experiments import OTtask

# Parse command line arguments
parser = argparse.ArgumentParser(description='Run Synthetic2 OT experiments')
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
parser.add_argument('--force-rerun', action='store_true',
                    help='Force rerun of the experiments even if results exist')
args = parser.parse_args()

# Use parsed arguments
reg_list = args.reg
problem_sizes = args.size
max_iter, tol = args.max_iter, args.tol
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
        synthetic2_task = OTtask(problem=synthetic2_ot_problem, solvers=synthetic2_solvers)
        
        print(f"Running Synthetic2 experiment with size {problem_size}, reg {reg}")
        synthetic2_task.plot_for_problem(x_key='iterations',
                                         x_label='Iteration Number',
                                         y_label='Log10 Gradient Norm',
                                         force_rerun=force_rerun)
        synthetic2_task.plot_for_problem(x_key='run_times',
                                         x_label='Run time(seconds)',
                                         y_label='Log10 Gradient Norm',
                                         force_rerun=force_rerun)
