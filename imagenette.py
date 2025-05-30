import argparse
from utils import get_solvers
from ot.datasets import ImagenetteOT
from ot.experiments import OTtask

# Parse command line arguments
parser = argparse.ArgumentParser(description='Run Imagenette OT experiments')
parser.add_argument('--reg', nargs='+', type=float, default=[0.001],
                    help='List of regularization parameters (default: [0.001])')
parser.add_argument('--norm', nargs='+', type=str, default=['l1'],
                    choices=['l1', 'l2'], 
                    help='List of distance norms to use (default: [l1])')
parser.add_argument('--source', nargs='+', type=str, default=['tench'],
                    help='Source Imagenette class (default: [\'tench\'])',
                    choices=['tench', 'English springer', 'cassette player', 
                             'chain saw', 'church', 'French horn', 'garbage truck', 
                             'gas pump', 'golf ball', 'parachute'])
parser.add_argument('--target', nargs='+', type=str, default=['cassette player'],
                    help='Target Imagenette class(default: [\'cassette player\'])',
                    choices=['tench', 'English springer', 'cassette player', 
                             'chain saw', 'church', 'French horn', 'garbage truck', 
                             'gas pump', 'golf ball', 'parachute'])
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
source_classes = args.source
target_classes = args.target
class_pairs = list(zip(source_classes, target_classes))
imagenette_methods = args.methods if args.methods else [
    'BCD', 'APDAGD', # first-order methods
    'LBFGS-Dual', 'Newton', # second-order methods
    'SSNS', # sparse method
    'SPLR' # new methods
]
force_rerun = args.force_rerun

for reg in reg_list:
    for norm in norm_list:
        for source_class, target_class in class_pairs:
            # Solving the Imagenette OT problem
            imagenette_ot_problem = ImagenetteOT(
                source_classname=source_class,
                target_classname=target_class,
                reg=reg, distance=norm
            )
            imagenette_solvers = get_solvers(reg=reg, max_iter=max_iter, tol=tol,
                                            selected=imagenette_methods)
            imagenette_task = OTtask(problem=imagenette_ot_problem, solvers=imagenette_solvers)
            imagenette_task.plot_for_problem(x_key='iterations',
                                             x_label='Iteration Number',
                                             y_label='Log10 Gradient Norm',
                                             force_rerun=force_rerun)
            imagenette_task.plot_for_problem(x_key='run_times',
                                             x_label='Run time(seconds)',
                                             y_label='Log10 Gradient Norm',
                                             force_rerun=force_rerun)
