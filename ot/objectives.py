import numpy as np

class OT:
    def __init__(self, source: np.ndarray, target: np.ndarray, cost_matrix: np.ndarray):
        """Initialize the OT problem"""
        self.source = source
        self.target = target
        self.cost_matrix = cost_matrix
    
    def __str__(self):
        source_shape = self.source.shape
        target_shape = self.target.shape
        cost_matrix_shape = self.cost_matrix.shape
        return f"Source shape: {source_shape}, Target shape: {target_shape}, Cost matrix shape: {cost_matrix_shape}"

class SinkhornOT(OT):
    def __init__(self, source: np.ndarray, target: np.ndarray, cost_matrix: np.ndarray,
                 reg: float = 0.01):
        """Initialize the Sinkhorn OT problem"""
        super().__init__(source, target, cost_matrix)
        self.reg = reg