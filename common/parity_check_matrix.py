import itertools as it
from learned_BP.pynite_field import vec2de, de2vec
import numpy as np
import scipy
import torch


class ParityCheckMatrix:
    def __init__(self, H, use_cuda=True):
        self.H = torch.tensor(H)
        # number of factor nodes, variable nodes, and edges in factor graph
        (self.M, self.N), self.E = H.shape, int(H.sum())
        # Construct auxiliary matrix for sparse row sum and sparse column sum
        I_row, I_col = np.where(H != 0)
        self.I_row = torch.tensor(I_row)
        self.I_col = torch.tensor(I_col)
        # Create indices for sparse tensor for row/column sum, sort in lexicographic order
        order = np.arange(self.E)
        r = torch.LongTensor([I_row, order])
        c = torch.LongTensor([np.sort(I_col), order[np.argsort(I_col)]])
        # Construct sparse tensor for row/column sum
        ones = torch.tensor(np.ones(order.shape)).float()
        self.S_row = torch.sparse.FloatTensor(r, ones, torch.Size((self.M, self.E)))
        self.S_col = torch.sparse.FloatTensor(c, ones, torch.Size((self.N, self.E)))
        # Dispatch data to GPU
        if use_cuda:
            self.cuda()

    def __repr__(self):
        return 'Parity check matrix of size ({0}, {1})'.format(self.M, self.N)

    # Define functions for row/column gather, sparse row/column sum and sparse leave-one-out row/column sum
    def row_gather(self, msg):
        # length-M input, length-E output
        return msg[self.I_row]

    def col_gather(self, msg):
        # length-N input, length-E output
        return msg[self.I_col]

    def row_sum(self, msg):
        # length-E input, length-M output
        return self.S_row @ msg

    def col_sum(self, msg):
        # length-E input, length-N output
        return self.S_col @ msg

    def row_sum_loo(self, msg):
        # length-E input, length-E output
        return self.row_gather(self.row_sum(msg)) - msg

    def col_sum_loo(self, msg):
        # length-E input, length-E output
        return self.col_gather(self.col_sum(msg)) - msg

    def cuda(self):
        self.H = self.H.cuda()
        self.I_row = self.I_row.cuda()
        self.I_col = self.I_col.cuda()
        self.S_row = self.S_row.cuda()
        self.S_col = self.S_col.cuda()


def gf2_rank(A):
    '''
    Return the rank of a binary matrix
    '''
    (m, n), j = A.shape, 0
    Ar, rank = A.copy(), 0
    for i in range(min(m, n)):
        # Find value and index of non-zero element in the remainder of column i.
        while j < n:
            temp = np.where(Ar[i:, j] != 0)[0]
            if len(temp) == 0:
                # If the lower half of j-th row is all-zero, check next column
                j += 1
            else:
                # Swap i-th and k-th rows
                k, rank = temp[0] + i, rank + 1
                if i != k:
                    Ar[[i, k], j:] = Ar[[k, i], j:]
                # Save the right hand side of the pivot row
                row = Ar[i, j:].reshape((1, -1))
                col = np.concatenate([np.zeros(i + 1, dtype=int), Ar[i + 1:, j]]).reshape((-1, 1))
                Ar[:, j:] ^= col * row
                break
        j += 1
    return rank


def girth(H, mode=None):
    '''
    Compute the girth of a bipartite graph
    mode = 0: run BFS on all check nodes
    mode = 1: run BFS on all variable nodes
    '''
    # Construct adjacency list
    C_link = [set(np.where(row)[0]) for c, row in enumerate(H)]
    V_link = [set(np.where(col)[0]) for v, col in enumerate(H.T)]
    if mode is None:
        mode = int(H.shape[0] > H.shape[1])
    # Use BFS the length of shortest cycle starting from the given node

    def shortest_cycle(x, part=mode):
        level, L = {(x, -1)} if mode == 0 else {(-1, x)}, 0
        while level:
            new_level, L, part = set(), L + 1, 1 - part
            if part:  # check to variable
                for c, v0 in level:
                    for v in C_link[c] - {v0}:
                        new_level.add((c, v))
                        if v == x and mode == 1:
                            return L
            else:     # variable to check
                for c0, v in level:
                    for c in V_link[v] - {c0}:
                        new_level.add((c, v))
                        if c == x and mode == 0:
                            return L
            level = new_level
        return float('inf')
    return min(shortest_cycle(x, mode) for x in range(H.shape[mode]))


def cycle_count(H):
    '''
    Compute the girth g, number of cycles with length g and (g+2)
    Reference:  Halford, Chugg, 2006
                An Algorithm for Counting Short Cycles in Bipartite Graphs
    '''
    def Diag(A):
        return np.diag(np.diag(A))                       # A o I

    def Z(A):
        return A - Diag(A)                               # Eq 2

    def Binom(A, k):
        return scipy.special.binom(A, k).astype(int)  # Eq 3

    def Max0(A, k):
        return np.maximum(A - k, 0)                   # Eq 4 & 6

    (m, n), g = H.shape, girth(H)
    if np.isinf(g):
        return g, 0, 0, 0
    # 2k, m-by-m: # length-2k paths from check a to check b
    # 2k+1, m-by-n: # length-(2k+1) paths from check a to variable i
    PC = {0: np.eye(m, dtype=int), 1: H}
    # 2k, n-by-n: # length-2k paths from variable i to variable j
    # 2k+1, n-by-m: # length-(2k+1) paths from variable i to check a
    PV = {0: np.eye(n, dtype=int), 1: H.T}
    # (2k', 2k-2k'), m-by-m: # (2k', 2k-2k')-lolipops from check a to check b
    # (2k'+1, 2k-2k'), m-by-n: # (2k', 2k-2k')-lolipops from check a to variable i
    LC = {(0, 0): np.zeros((m, m), dtype=int)}
    # (2k', 2k-2k'), m-by-m: # (2k', 2k-2k')-lolipops from variable i to variable j
    # (2k'+1, 2k-2k'), n-by-m: # (2k', 2k-2k')-lolipops from variable i to check a
    LV = {(0, 0): np.zeros((n, n), dtype=int)}

    # Length-2 cycle, path
    LC[(0, 2)] = Diag(PC[1] @ H.T)           # Eq 11
    LV[(0, 2)] = Diag(PV[1] @ H)           # Eq 12
    PC[2] = PC[1] @ H.T - LC[(0, 2)]         # Eq 14
    PV[2] = PV[1] @ H - LV[(0, 2)]         # Eq 15

    # Length-3 lolipop, path
    LC[(1, 2)] = H @ Max0(LV[(0, 2)], 1)    # Eq 26
    LV[(1, 2)] = H.T @ Max0(LC[(0, 2)], 1)    # Eq 27
    PC[3] = PC[2] @ H - LC[(1, 2)]         # Eq 16
    PV[3] = PV[2] @ H.T - LV[(1, 2)]         # Eq 17

    # Length-4, 5, ..., g-1 cycle, lolipop, path
    LC[(2, 2)] = Z(H   @ LV[(1, 2)])        # Eq 28
    LV[(2, 2)] = Z(H.T @ LC[(1, 2)])        # Eq 29
    for k in range(2, g // 2):
        PC[2 * k] = PC[2 * k - 1] @ H.T - LC[(2 * k - 2, 2)]                                   # Eq 14
        PV[2 * k] = PV[2 * k - 1] @ H - LV[(2 * k - 2, 2)]                                   # Eq 15
        LC[(2 * k - 1, 2)] = H   @ LV[(2 * k - 2, 2)] - Max0(LC[(0, 2)], 1) @ LC[(2 * k - 3, 2)]    # Eq 30
        LV[(2 * k - 1, 2)] = H.T @ LC[(2 * k - 2, 2)] - Max0(LV[(0, 2)], 1) @ LV[(2 * k - 3, 2)]    # Eq 30'

        PC[2 * k + 1] = PC[2 * k] @ H - LC[(2 * k - 1, 2)]                                   # Eq 16
        PV[2 * k + 1] = PV[2 * k] @ H.T - LV[(2 * k - 1, 2)]                                   # Eq 17
        LC[(2 * k, 2)] = H   @ LV[(2 * k - 1, 2)] - Max0(LC[(0, 2)], 1) @ LC[(2 * k - 2, 2)]      # Eq 31
        LV[(2 * k, 2)] = H.T @ LC[(2 * k - 1, 2)] - Max0(LV[(0, 2)], 1) @ LV[(2 * k - 2, 2)]      # Eq 31'

    # Length-g cycle, path
    LC[(0, g)] = Diag(PC[g - 1] @ H.T)                 # Eq 11
    LV[(0, g)] = Diag(PV[g - 1] @ H)                 # Eq 12
    PC[g] = PC[g - 1] @ H.T - LC[(0, g)] - LC[(g - 2, 2)]  # Eq 18
    PV[g] = PV[g - 1] @ H - LV[(0, g)] - LV[(g - 2, 2)]  # Eq 19

    # Length-(g+1) lolipop, path
    LC[(1, g)] = H   @ LV[(0, g)] - 2 * PC[g - 1] * H                                     # Eq 35
    LV[(1, g)] = H.T @ LC[(0, g)] - 2 * PV[g - 1] * H.T                                   # Eq 35'
    LC[(g - 1, 2)] = H   @ LV[(g - 2, 2)] - PC[g - 1] * H - Max0(LC[(0, 2)], 1) @ LC[(g - 3, 2)]  # Eq 32
    LV[(g - 1, 2)] = H.T @ LC[(g - 2, 2)] - PV[g - 1] * H.T - Max0(LV[(0, 2)], 1) @ LV[(g - 3, 2)]  # Eq 32'
    PC[g + 1] = PC[g] @ H - LC[(1, g)] - LC[(g - 1, 2)]                                     # Eq 20
    PV[g + 1] = PV[g] @ H.T - LV[(1, g)] - LV[(g - 1, 2)]                                     # Eq 21

    # Length-(g+2) cycle, lolipop, path
    LC[(0, g + 2)] = Diag(PC[g + 1] @ H.T)       # Eq 11
    LV[(0, g + 2)] = Diag(PV[g + 1] @ H)       # Eq 12
    LC[(2, g)] = Z(H   @ LV[(1, g)])      # Eq 36
    LV[(2, g)] = Z(H.T @ LC[(1, g)])      # Eq 36'
    if g == 4:
        LC[(2, g)] -= 6 * Binom(PC[2], 3)    # Eq 36
        LV[(2, g)] -= 6 * Binom(PV[2], 3)    # Eq 36
    LC[(g, 2)] = Z(H   @ LV[(g - 1, 2)]) - Max0(LC[(0, 2)], 1) @ LC[(g - 2, 2)] + PC[g - 2] * PC[2]    # Eq 33
    LV[(g, 2)] = Z(H.T @ LC[(g - 1, 2)]) - Max0(LV[(0, 2)], 1) @ LV[(g - 2, 2)] + PV[g - 2] * PV[2]    # Eq 33'
    if g == 4:
        LC[(g, 2)] -= PC[2]                  # Eq 33
        LV[(g, 2)] -= PV[2]                  # Eq 33'
    PC[g + 2] = PC[g + 1] @ H.T - LC[(0, g + 2)] - LC[(2, g)] - LC[(g, 2)]   # Eq 22
    PV[g + 2] = PV[g + 1] @ H - LV[(0, g + 2)] - LV[(2, g)] - LV[(g, 2)]   # Eq 23

    # Length-(g+3) lolipop, path
    LC[(1, g + 2)] = H   @ LV[(0, g + 2)] - 2 * PC[g + 1] * H               # Eq 38
    LV[(1, g + 2)] = H.T @ LC[(0, g + 2)] - 2 * PV[g + 1] * H.T             # Eq 38'
    if g == 4:
        LC[(1, g + 2)] += 2 * (-Binom(PC[3], 2) + H   @ Binom(PV[2], 2) + Binom(PC[2], 2) @ H - 2 * PC[3]) * H         # Eq 38
        LV[(1, g + 2)] += 2 * (-Binom(PV[3], 2) + H.T @ Binom(PC[2], 2) + Binom(PV[2], 2) @ H.T - 2 * PV[3]) * H.T       # Eq 38'
    LC[(3, g)] = H   @ LV[(2, g)] - Max0(LC[(0, 2)], 1) @ LC[(1, g)]    # Eq 37
    LV[(3, g)] = H.T @ LC[(2, g)] - Max0(LV[(0, 2)], 1) @ LV[(1, g)]    # Eq 37'
    if g == 4:
        LC[(3, g)] += (-4 * Binom(PC[3], 2) + 6 * H   @ Binom(PV[2], 2) + 4 * Binom(PC[2], 2) @ H - 10 * PC[3]) * H   # Eq 37
        LV[(3, g)] += (-4 * Binom(PV[3], 2) + 6 * H.T @ Binom(PC[2], 2) + 4 * Binom(PV[2], 2) @ H.T - 10 * PV[3]) * H.T  # Eq 37'
    elif g == 6:
        LC[(3, g)] -= 6 * Binom(PC[3], 3)    # Eq 37
        LV[(3, g)] -= 6 * Binom(PV[3], 3)    # Eq 37'
    LC[(g + 1, 2)] = H   @ LV[(g, 2)] - LC[(0, g)] @ LC[(1, 2)] - PC[g + 1] * H - Max0(LC[(0, 2)], 1) @ LC[(g - 1, 2)] \
        + (2 * PC[g - 1] @ Max0(LV[(0, 2)], 2) + LC[(g - 1, 2)] + 2 * PC[g - 1]) * H      # Eq 34
    LV[(g + 1, 2)] = H.T @ LC[(g, 2)] - LV[(0, g)] @ LV[(1, 2)] - PV[g + 1] * H.T - Max0(LV[(0, 2)], 1) @ LV[(g - 1, 2)] \
        + (2 * PV[g - 1] @ Max0(LC[(0, 2)], 2) + LV[(g - 1, 2)] + 2 * PV[g - 1]) * H.T    # Eq 34'
    if g == 4:
        LC[(g + 1, 2)] += 2 * (Binom(PC[2], 2) @ H - PC[3]) * H        # Eq 34
        LV[(g + 1, 2)] += 2 * (Binom(PV[2], 2) @ H.T - PV[3]) * H.T      # Eq 34'
    PC[g + 3] = PC[g + 2] @ H - LC[(1, g + 2)] - LC[(3, g)] - LC[(g + 1, 2)]     # Eq 24
    PV[g + 3] = PV[g + 2] @ H.T - LV[(1, g + 2)] - LV[(3, g)] - LV[(g + 1, 2)]     # Eq 25

    # Length-(g+4) cycles
    LC[(0, g + 4)] = Diag(PC[g + 3] @ H.T)       # Eq 11
    LV[(0, g + 4)] = Diag(PV[g + 3] @ H)       # Eq 12

    # Return girth, number of g/g+2/g+4-cycles
    def cycle_num(x):
        return np.diag(LV[(0, x)]).sum() // x     # Eq 13
    return g, cycle_num(g), cycle_num(g + 2), cycle_num(g + 4)


def one_reduction(H, order=1, verbose=False):
    '''
    Try to reduce number of ones in parity check matrix
    '''
    m, Hp, num1 = H.shape[0], H.copy(), H.sum()

    def all_case():
        for r in range(m):
            for t in range(1, order + 1):
                for rest in it.combinations((k for k in range(m) if k != r), t):
                    yield r, rest

    def case_with(r0):
        for t in range(1, order + 1):
            for rest in it.combinations((k for k in range(m) if k != r0), t):
                yield r0, rest
        for r in (k for k in range(m) if k != r0):
            yield r, (r0,)
            for t in range(1, order):
                for rest in it.combinations((k for k in range(m) if k not in [r, r0]), t):
                    yield r, tuple([r0] + list(rest))
                yield r, rest
    test_cases = set(all_case())
    while test_cases:
        new_test_cases = test_cases.copy()
        rs, row_s, new_num1 = None, None, num1
        for r, rest in test_cases:
            row = Hp[list(rest)].sum(axis=0) % 2
            reduce_1 = Hp[r].sum() - (Hp[r] ^ row).sum()
            if reduce_1 <= 0:
                new_test_cases.discard((r, rest))
            if new_num1 > num1 - reduce_1:
                new_num1, rs, row_s = num1 - reduce_1, r, row
                if verbose:
                    print('{0} {1} {2} {3}'.format(rs, reduce_1, new_num1, len(new_test_cases)))
        if rs is None:
            break
        else:
            Hp[rs], num1 = Hp[rs] ^ row_s, new_num1
            new_test_cases |= set(case_with(rs))
        test_cases = new_test_cases
    return Hp


def list_all_codewords(G, w_min, w_max, M=float('inf')):
    '''
    list at least M codewords with weight in range [w_min, w_max]
    '''
    k, n = G.shape
    if k < 20:
        V = de2vec(np.arange(2**k))
        C = (V @ G) % 2
        return C[(w_min <= C.sum(axis=1)) & (C.sum(axis=1) <= w_max)]
    else:
        ans = []
        for v in it.product([0, 1], repeat=k):
            c = (np.array(v) @ G) % 2
            if w_min <= c.sum() <= w_max:
                ans.append(c)
                print('{0}, {1}'.format(len(ans), vec2de(v[::-1])))
            if len(ans) >= M:
                break
        return np.array(ans)


def random_invertible_matrix(rows, m, ignore_rows=None, verbose=1):
    '''
    Given a list of rows, generate a random full-rank matrix by picking m rows
    with minimum number of ones
    '''
    n = rows.shape[1]
    ans = np.zeros((m, n), dtype=int)
    ignore_rows = ignore_rows if ignore_rows is not None else np.zeros((0, n))
    ignore = set() if ignore_rows is None else set(vec2de(ignore_rows))
    index = set((w, idx) for w, idx in zip(rows.sum(axis=1), vec2de(rows)) if idx not in ignore)
    for i in range(m):
        candidate = sorted(index)
        candidate = [idx for w, idx in candidate if w == candidate[0][0]]
        # ans[i] = de2vec(np.random.choice(candidate), n=n)
        ans[i] = de2vec(candidate[0], n=n)
        min_cycle_count, best_cand = cycle_count(np.vstack((ignore_rows, ans[:i + 1]))), candidate[0]
        for j, cand in enumerate(candidate, 1):
            if np.isinf(min_cycle_count[0]):
                break
            ans[i] = de2vec(cand, n=n)
            curr_cycle_count = cycle_count(np.vstack((ignore_rows, ans[:i + 1])))
            if curr_cycle_count[0] > min_cycle_count[0] or curr_cycle_count[1:] < min_cycle_count[1:]:
                if verbose >= 2:
                    print(j, len(candidate), curr_cycle_count, min_cycle_count)
                min_cycle_count, best_cand = curr_cycle_count, cand
        ans[i] = de2vec(best_cand, n=n)
        if verbose >= 1:
            print(i, min_cycle_count, '\n' * (verbose == 2))

        if i < m - 1:
            if i < 0:
                V = (de2vec(np.arange(1, 2**(i + 1))) @ ans[:i + 1]) % 2
                index -= set((v.sum(), vec2de(v)) for v in V)
            else:
                for w, idx in list(index):
                    ans[i + 1] = de2vec(idx, n=n)
                    if gf2_rank(ans[:i + 2]) != i + 2:
                        index.discard((w, idx))
                ans[i + 1] = np.zeros(n)
        if len(index) == 0 and i < m - 1:
            print('Failed to create full rank matrix, only find {} rows'.format(i + 1))
            break
    return ans


def cycle_reduction(H, verbose=False):
    '''
    Tanner graph cycle reduction
    Reference:  Halford, Chugg 2006
                Random Redundant Soft-In Soft-Out Decoding of Linear Block Codes
    '''
    m, Hp = H.shape[0], H.copy()
    gs, Ngs, Ng2s, _ = cycle_count(Hp)
    if verbose:
        print(gs, Ngs, Ng2s)
    while True:
        r1s, r2s = None, None
        for r1, r2 in it.product(range(m), repeat=2):
            # For r1 != r2, do the following
            if r1 == r2:
                continue
            # Replace row r2 in H' with r1 + r2 in modulo 2
            Hp[r2] = (Hp[r1] + Hp[r2]) % 2
            g, Ng, Ng2, _ = cycle_count(Hp)
            # If the new factor graph has larger g
            # Or has same g but smaller N_g
            # Or has same g and same N_g, but smaller N_{g+2}
            # Save the new changes
            if (-g, Ng, Ng2) < (-gs, Ngs, Ng2s):
                r1s, r2s = r1, r2
                gs, Ngs, Ng2s = g, Ng, Ng2
                if verbose:
                    print((r1, r2), gs, Ngs, Ng2s, Hp.sum())
            # Undo the replacement
            Hp[r2] = (Hp[r2] - Hp[r1]) % 2
        if r1s is None and r2s is None:
            # If r1* and r2* are not assigned
            # No progress can be made, jump out the while loop
            break
        else:
            # If r1* and r2* are assigned
            # Replace row r2* in H' with r1* + r2* in modulo 2
            Hp[r2s] = (Hp[r1s] + Hp[r2s]) % 2
    return Hp
