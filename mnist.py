import argparse
from utils import get_solvers
from ot.datasets import MnistOT
from ot.experiments import OTtask

# Parse command line arguments
parser = argparse.ArgumentParser(description='Run MNIST OT experiments')
parser.add_argument('--reg', nargs='+', type=float, default=[0.001],
                    help='List of regularization parameters (default: [0.001])')
parser.add_argument('--norm', nargs='+', type=str, default=['l1'],
                    choices=['l1', 'l2'], 
                    help='List of distance norms to use (default: [l1])')
parser.add_argument('--source', nargs='+', type=int, default=[2],
                    help='Source MNIST index (default: [2])')
parser.add_argument('--target', nargs='+', type=int, default=[54698],
                    help='Target MNIST index (default: [54698])')
parser.add_argument('--max-iter', type=int, default=500,
                    help='Maximum number of iterations (default: 500)')
parser.add_argument('--tol', type=float, default=1e-6,
                    help='Tolerance for convergence (default: 1e-6)')
parser.add_argument('--methods', nargs='+', type=str, default=None,
                    help='List of methods to use, separated by space(default: suitable methods)')
parser.add_argument('--force-rerun', action='store_true',
                    help='Force rerun of the experiments even if results exist')
args = parser.parse_args()

# Use parsed arguments
reg_list = args.reg
norm_list = args.norm
max_iter, tol = args.max_iter, args.tol
idx_pairs = list(zip(args.source, args.target))
mnist_methods = args.methods if args.methods else [
    'BCD', 'APDAGD', # first-order methods
    'LBFGS-Dual', 'Newton', # second-order methods
    'SSNS', # sparse method
    'SPLR', # new methods
]
force_rerun = args.force_rerun

for reg in reg_list:
    for norm in norm_list:
        for source_idx, target_idx in idx_pairs:
            # Solving the MNIST OT problem
            mnist_ot_problem = MnistOT(
                source_idx=source_idx, target_idx=target_idx,
                reg=reg, distance=norm
            )
            mnist_solvers = get_solvers(reg=reg, max_iter=max_iter, tol=tol,
                                        selected=mnist_methods)
            mnist_task = OTtask(problem=mnist_ot_problem, solvers=mnist_solvers)
            mnist_task.plot_for_problem(x_key='iterations',
                                        x_label='Iteration Number',
                                        y_label='Log10 Gradient Norm',
                                        force_rerun=force_rerun)
            mnist_task.plot_for_problem(x_key='run_times',
                                        x_label='Run time(seconds)',
                                        y_label='Log10 Gradient Norm',
                                        force_rerun=force_rerun)