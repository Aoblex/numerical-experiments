import os
import argparse
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

np.set_printoptions(precision=2, suppress=True)
np.random.seed(42)

SAVE = 'save'
EIGEN = 'eigen'

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

def get_sparsification(n: int, m: int, num_zeros: int, repeat: int) -> list[list[tuple[int, int]]]:
    rows = np.arange(1, n)
    cols = np.arange(1, m - 1)
    coords = np.array(np.meshgrid(rows, cols)).T.reshape(-1, 2)
    sparsifications = []
    for _ in range(repeat):
        np.random.shuffle(coords)
        sparsifications.append([tuple(coord) for coord in coords[:num_zeros]])
    return sparsifications

def Hessian_eigenvalue(n, m, sparsifications: list[list[tuple[int, int]]], stride: int) \
    -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    results = []

    for sparsification in sparsifications:

        T = np.random.rand(n, m)
        H = np.zeros((n + m - 1, n + m - 1))
        H[:n, :n] = np.diag(T.sum(axis=1))
        H[n:, n:] = np.diag(T[:, :-1].sum(axis=0))
        H[:n, n:] = T[:, :-1]
        H[n:, :n] = T[:, :-1].T

        max_eigenvalues = []
        min_eigenvalues = []
        num_sparsifications = []

        for k, (i, j) in tqdm(enumerate(sparsification),
                              desc="Sparsification Progress"):

            if k % stride == 0:
                eigenvalues = np.linalg.eigvalsh(H)
                min_eigenvalues.append(eigenvalues[0])
                max_eigenvalues.append(eigenvalues[-1])
                num_sparsifications.append(k)

            H[i, n + j] = 0
            H[n + j, i] = 0

        results.append((
            np.array(min_eigenvalues),
            np.array(max_eigenvalues),
            np.array(num_sparsifications)
        ))
    
    return results

def main(n, m, num_zeros, stride, repeat):
    sparsifications = get_sparsification(n, m, num_zeros, repeat)
    results = Hessian_eigenvalue(n, m, sparsifications, stride)

    eigen_path = os.path.join(SAVE, EIGEN)
    if not os.path.exists(eigen_path):
        os.makedirs(eigen_path)

    min_eig_list, max_eig_list, steps_list = zip(*results)
    min_eig_list = np.array(min_eig_list)
    max_eig_list = np.array(max_eig_list)
    log10_condition_number_list = np.log10(max_eig_list / min_eig_list)
    steps = steps_list[0]  # Assuming all simulations have the same steps
    
    min_eig_min = min_eig_list.min(axis=0)
    min_eig_max = min_eig_list.max(axis=0)
    min_eig_mean = min_eig_list.mean(axis=0)

    max_eig_min = max_eig_list.min(axis=0)
    max_eig_max = max_eig_list.max(axis=0)
    max_eig_mean = max_eig_list.mean(axis=0)

    log10_condition_number_min = log10_condition_number_list.min(axis=0)
    log10_condition_number_max = log10_condition_number_list.max(axis=0)
    log10_condition_number_mean = log10_condition_number_list.mean(axis=0)

    plt.figure(num=1, figsize=(10, 6), dpi=300)
    plt.title(f'Hessian Eigenvalues (n={n}, m={m})')
    plt.xlabel('Sparsification Step')
    plt.ylabel('Eigenvalue')
    plt.grid(True)
    plt.plot(steps, min_eig_mean, color='blue', linestyle='dashed', linewidth=0.8, label='Min Eigenvalue')
    plt.fill_between(steps, min_eig_min, min_eig_max,     
                     color='blue', alpha=0.2)
    plt.plot(steps, max_eig_mean, color='red', linestyle='solid', linewidth=0.8, label='Max Eigenvalue')
    plt.fill_between(steps, max_eig_min, max_eig_max,
                     color='red', alpha=0.2)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(eigen_path, f'eigenvalues_n{n}_m{m}.png'))

    plt.figure(num=2, figsize=(10, 6), dpi=300)
    plt.title(f'Hessian Condition Number (n={n}, m={m})')
    plt.xlabel('Sparsification Step')
    plt.ylabel('Log10 Condition Number')
    plt.grid(True)
    plt.plot(steps, log10_condition_number_mean, color='green', linestyle='solid', linewidth=0.8,
             label='Log10 Condition Number')
    plt.fill_between(steps, log10_condition_number_min, log10_condition_number_max,
                     color='green', alpha=0.2)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(eigen_path, f'condition_number_n{n}_m{m}.png'))

    plt.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Test how the eigenvalues change after sparsification')
    parser.add_argument('--n', type=int, default=100, 
                        help='Number of rows (default: 100)')
    parser.add_argument('--m', type=int, default=100,
                        help='Number of columns (default: 100)')
    parser.add_argument('--num-zeros', type=int, default=None,
                        help='Number of zeros to introduce (default: None, set [1:, 1:] to zero)')
    parser.add_argument('--stride', type=int, default=1,
                        help='Stride for eigenvalue calculation (default: 1)')
    parser.add_argument('--repeat', type=int, default=3,
                        help='Number of times to repeat the sparsification (default: 10)')
    args = parser.parse_args()

    n, m = args.n, args.m
    num_zeros = min((n-1)*(m-2), args.num_zeros) \
                if args.num_zeros is not None \
                else (n-1) * (m-2)
    stride = args.stride
    repeat = args.repeat

    main(n, m, num_zeros, stride, repeat)