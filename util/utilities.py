import os
import numpy as np
from matplotlib import image


def load_images_to_dict(path='./images/original/', suffix='.png') -> dict:
    '''
    Loads all png files found in path
    :param path: path to folder containing images
    :return: list of numpy arrays containing loaded png files
    '''

    assert os.path.exists(path)
    png_names = [f for f in os.listdir(path) if f.endswith(suffix)]
    imgs = {}

    if len(png_names) == 0:
        print(f"No PNG files found in {path}\n returning empty dict.")

    try:
        # try to load all images in png_names
        imgs = {f: image.imread(os.path.join(path, f)) for f in png_names}
    except FileNotFoundError as fnf:
        print(fnf)

    return imgs


def move_color_axis(img, source, target):

    return np.moveaxis(img, source, target)

###################
# FROM https://stackoverflow.com/questions/6284396/permutations-with-unique-values#6285203 without comments
class unique_element:
    def __init__(self, value, occurrences):
        self.value = value
        self.occurrences = occurrences


def perm_unique(elements):
    eset = set(elements)
    listunique = [unique_element(i, elements.count(i)) for i in eset]
    u = len(elements)
    return perm_unique_helper(listunique, [0] * u, u - 1)


def perm_unique_helper(listunique, result_list, d):
    if d < 0:
        yield tuple(result_list)
    else:
        for i in listunique:
            if i.occurrences > 0:
                result_list[d] = i.value
                i.occurrences -= 1
                for g in perm_unique_helper(listunique, result_list, d - 1):
                    yield g
                i.occurrences += 1
###################


def accel_asc(n):
    '''
    partitioning from http://jeromekelleher.net/generating-integer-partitions.html without comments

    Integer partitioning (https://en.wikipedia.org/wiki/Partition_(number_theory)) is the problem that asks for all
    ways a natural number can be expressed as a sum of other natural numbers up to commutativity
    '''
    a = [0 for i in range(n + 1)]
    k = 1
    y = n - 1
    while k != 0:
        x = a[k - 1] + 1
        k -= 1
        while 2 * x <= y:
            a[k] = x
            y -= x
            k += 1
        l = k + 1
        while x <= y:
            a[k] = x
            a[l] = y
            yield a[:k + 2]
            x += 1
            y -= 1
        a[k] = x + y
        y = x + y - 1
        yield a[:k + 1]


def multinom_coefficient(n, k):
    return np.math.factorial(n) / np.prod([np.math.factorial(_k) for _k in k])


def transform_data(X, poly_degree=4, n_cpus=None):
    '''
        computes phi(x) such that <phi(x_1), phi(x_2)> = (<x_1, x_2>)^poly_degree but only for
        cases where data dimensionality >= poly_degree.
        based on the multinomial theorem (https://en.wikipedia.org/wiki/Multinomial_theorem)

        :parameter X dataset, (row, col) = (samples, features)
        :parameter poly_degree Degree of the polynomial
        :parameter n_cpus to use for multiprocessing; if None n_cpus will be max(ceil(cpu_count/2), 5)
    '''

    n_samples, dim = X.shape
    assert dim >= poly_degree

    target_dim = binom(dim + poly_degree -1, dim - 1).astype(np.int)
    print(f"data will be mapped to a {target_dim}d space")

    # partitioning for |k|==n
    part = [i + [0]*(dim-len(i)) for i in accel_asc(poly_degree)]
    # print(part)

    ## all unique permutations for all partitionings
    permutations = []
    for k in part:
        # print(f"perm for {k}")
        for i in perm_unique(k):
            permutations.append(i)

    ## precompute all coefficients and take the square root
    coefficients = []
    for k in permutations:
        coefficients.append(
            np.sqrt(multinom_coefficient(poly_degree, k))
        )

    if n_samples < 1000:
        return _transform(X, permutations, coefficients, target_dim)
    else:
        return _multiproc_transform(X, permutations, coefficients, target_dim, n_cpus)


def _multiproc_transform(X, permutations, coefficients, target_dim, n_cpus=4):
    n_samples, _ = X.shape
    if n_cpus is None:
        n_cpus = np.int(np.ceil(mp.cpu_count()/2))
        n_cpus = min(n_cpus, 6)

    s = np.ceil(n_samples / n_cpus)
    ranges = [[np.int(i*s), np.int((i+1)*s)] for i in range(n_cpus)]
    ranges[-1][1] = n_samples

    q = mp.Queue()
    processes = []
    print(f"dispatching processes; data split in ranges {ranges}")
    for r in ranges:
        p = mp.Process(target=_transform, args=(X[r[0]:r[1]].copy(), permutations.copy(),
                                                coefficients.copy(), target_dim, (r[0], q)))
        p.start()
        processes.append(p)

    results = []
    for _ in range(n_cpus):
        results.append(q.get())

    for p in processes:
        p.join()

    # sort by idx_start
    results = sorted(results)

    return np.vstack([_x for _, _x in results])

def _transform(X, permutations, coefficients, target_dim, multi_proc_args=None):
    _x_transformed = np.zeros((X.shape[0], target_dim))
    for i, x in enumerate(tqdm(X)):
        _x = []
        for c, k in zip(coefficients, permutations):
            _x.append(c * np.prod(x ** k))
        _x_transformed[i] = np.array(_x)
    if multi_proc_args is None:
        return _x_transformed
    else: # multiproc
        idx_start, q = multi_proc_args
        return q.put((idx_start, _x_transformed))

if __name__ =='__main__':

    x = np.array([[1,1,2,3,4,5,6,7,8,9,10,11],
                  [1,1,2,3,4,5,6,7,8,9,10,11],
                  [1,1,2,3,4,5,6,7,8,9,10,11]])
    print(x)
    _x = transform_data(x, 2)
    print(np.sort(_x[0]))
