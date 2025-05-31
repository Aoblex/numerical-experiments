from ot.experiments import OTsolver
from typing import List
import regot

# palette: https://colorhunt.co/palettes/random
# color picker: https://www.w3schools.com/colors/colors_picker.asp
#               https://htmlcolorcodes.com/color-picker 

def get_solvers(reg, max_iter, tol, selected: None | List[str] = None) -> List:
    sinkhorn_bcd = OTsolver(method=regot.sinkhorn_bcd, method_name='BCD',
                            color='#c44e52', linestyle=(0, (3, 1, 2, 1)),
                            reg=reg, max_iter=max_iter, tol=tol)
    sinkhorn_apdagd = OTsolver(method=regot.sinkhorn_apdagd, method_name='APDAGD',
                            color='#937860', linestyle=(0, (2, 2)),
                            reg=reg, max_iter=max_iter, tol=tol)
    sinkhorn_lbfgs_dual = OTsolver(method=regot.sinkhorn_lbfgs_dual, method_name='LBFGS-Dual',
                                color='#ff8000', linestyle=(0, (3, 1)),
                                reg=reg, max_iter=max_iter, tol=tol)
    sinkhorn_newton = OTsolver(method=regot.sinkhorn_newton, method_name='Newton',
                            color='#da8bc3', linestyle=(0, (4, 2, 1, 2)),
                            reg=reg, max_iter=max_iter, tol=tol)
    sinkhorn_ssns = OTsolver(method=regot.sinkhorn_ssns, method_name='SSNS', shift=1e-6,
                            color='#55a868', linestyle=(0, (2, 1)),
                            reg=reg, max_iter=max_iter, tol=tol)
    sinkhorn_sparse_newton = OTsolver(method=regot.sinkhorn_sparse_newton, method_name='Sparse Newton', shift=1e-6,
                                    color='#e6a682', linestyle=(0, (1, 1)),
                                    reg=reg, max_iter=max_iter, tol=tol)
    sinkhorn_splr = OTsolver(method=regot.sinkhorn_splr, method_name='SPLR', shift=1e-6,
                            color='#4c72b0', linestyle='solid',
                            reg=reg, max_iter=max_iter, tol=tol)

    all_solvers = {
        'BCD': sinkhorn_bcd,
        'APDAGD': sinkhorn_apdagd,
        'LBFGS-Dual': sinkhorn_lbfgs_dual,
        'Newton': sinkhorn_newton,
        'SSNS': sinkhorn_ssns,
        'Sparse Newton': sinkhorn_sparse_newton,
        'SPLR': sinkhorn_splr,
    }

    if selected is not None:
        return [v for k, v in all_solvers.items() if k in selected]
    else:
        return all_solvers.values()

