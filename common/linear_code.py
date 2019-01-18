from learned_BP.common.parity_check_matrix import ParityCheckMatrix
from learned_BP.pynite_field import vec2de, de2vec
import numpy as np
import os
import scipy.io
import torch
from torch.distributions import Binomial


class LinearCode:
    def __init__(self, name, G, H, use_cuda=True):
        # If G is unknown, just give all-zero matrix with correct shape
        if G.shape[1] != H.shape[1]:
            raise Exception('The shape of G and H do not match')
        self.use_cuda = use_cuda
        self.name, self.suffix = name, ''
        self.G_unknown = np.count_nonzero(G) == 0
        self.G = torch.tensor(G).float()
        if self.use_cuda:
            self.G = self.G.cuda()
        self.K, self.N = G.shape
        self.rate = self.K / self.N
        self.H = ParityCheckMatrix(H, use_cuda)
        self.G_np, self.H_np = G, H
        self.generate_automorphism()

    def __repr__(self):
        return f'{self.full_name()}: G ({self.K}, {self.N}), H ({self.H.M}, {self.H.N})'

    def change_H(self, H, suffix='custom'):
        self.suffix = suffix
        self.H = ParityCheckMatrix(H)
        self.H_np = H

    def full_name(self):
        return f'{self.name}_{self.suffix}' if self.suffix else self.name

    def generate_codeword(self, batch_size, all_zero):
        if not self.G_unknown and not all_zero:
            if not self.use_cuda:
                # If G is known and not transmit all-zero codewords
                m = Binomial(total_count=1, probs=torch.ones((self.K, batch_size)) / 2).sample()
            else:
                m = torch.cuda.FloatTensor(self.K, batch_size).bernoulli_(0.5)
            x = (self.G.t() @ m) % 2
        else:
            x = torch.zeros((self.N, batch_size))
            if self.use_cuda:
                x = x.cuda()
        return x

    def generate_automorphism(self, perm_num=10000):
        self.automorphisms = np.arange(self.N, dtype=int).reshape((1, -1))
        self.automorphisms_inv = np.arange(self.N, dtype=int).reshape((1, -1))
        self.automorphisms = torch.tensor(self.automorphisms)
        self.inv_automorphisms = torch.tensor(self.inv_automorphisms)
        if self.use_cuda:
            self.automorphisms = self.automorphisms.cuda()
            self.inv_automorphisms = self.inv_automorphisms.cuda()
        print('Automorphism sampling not implemented for {0}, use trivial identity permutation'.format(self.name))

    def is_automorphism(self, perm):
        def is_row_full_rank(G):
            A = G.copy()
            (m, n), j, rank = A.shape, 0, 0
            for i in range(min(m, n)):
                # Find value and index of non-zero element in the remainder of column i.
                while j < n:
                    temp = np.where(A[i:, j] != 0)[0]
                    if len(temp) == 0:
                        # If the lower half of j-th row is all-zero, check next column
                        j += 1
                    else:
                        # Swap i-th and k-th rows
                        k, rank = temp[0] + i, rank + 1
                        if i != k:
                            A[[i, k], j:] = A[[k, i], j:]
                        A[i + 1:, j:] ^= A[i + 1:, j].reshape((-1, 1)) * A[i, j:].reshape((1, -1))
                        break
                j += 1
            return rank == min(m, n)

        G = self.G_np if not self.G_unknown else self.H_np
        if not is_row_full_rank(G) or G.shape[0] > G.shape[1]:
            raise Exception('G must be row full rank, fat matrix')
        # find column basis S
        S, GS, i = [0], G[:, [0]], 1
        for i in range(G.shape[1]):
            temp = np.bmat([GS, G[:, [i]]])
            if is_row_full_rank(temp):
                S, GS = S + [i], temp
            if len(S) == G.shape[0]:
                break
        # solve for A
        GP, GPS = G[:, perm], G[:, perm][:, S]
        A = np.mod(np.round(GPS @ np.linalg.inv(GS) * np.linalg.det(GS)), 2)
        if np.all(np.mod(A @ G, 2) == GP):
            return A
        else:
            print('input permutation is not in automorphism group')

    def random_automorphism(self, T=1):
        # Pick T random permutation automorphisms of the code
        # Return an T-by-N TF tensor, each row is a permutation
        choice = np.random.choice(self.automorphisms.shape[0], T)
        return self.automorphisms[choice], self.inv_automorphisms[choice]


class RM_Code(LinearCode):
    def __init__(self, n, k, mode='', use_cuda=True):
        mat = scipy.io.loadmat(os.path.join(os.path.split(__file__)[0], 'RM_matrices.mat'))
        code_name, H_sfx = f'RM_{n}_{k}', mode if mode in ['oc', 'sq', 'cr'] else ''
        G, H = mat[f'{code_name}_G'], mat[f'{code_name}_H{H_sfx}']
        if mode == 'oc':
            H = H.toarray()
        super(RM_Code, self).__init__(code_name, G, H, use_cuda)
        self.suffix = mode
        print('Successfully created linear code', repr(self))

    def generate_automorphism(self, perm_num=10000):
        def rand_GL_2_m(m):
            while True:
                A0, rank = np.random.randint(low=0, high=2, size=(m, m)), 0
                A = A0.copy()
                for i in range(m):
                    try:
                        k, rank = np.where(A[i:, i] != 0)[0][0] + i, rank + 1
                        if i != k:
                            A[[i, k], i:] = A[[k, i], i:]
                        A[i + 1:, i:] ^= A[i + 1:, i].reshape((-1, 1)) * A[i, i:].reshape((1, -1))
                    except Exception:
                        break
                if rank == m:
                    return A0
        m, col_index = int(np.log2(self.N)), de2vec(np.arange(self.N)).T
        self.automorphisms = np.zeros((perm_num, self.N), dtype=int)
        self.inv_automorphisms = np.zeros((perm_num, self.N), dtype=int)
        for i in range(perm_num):
            # Randomly pick affine transformation of vector index of columns
            D, b = rand_GL_2_m(m), np.random.randint(low=0, high=2, size=(m, 1))
            # Find the column permutation induced by the affine transformation
            self.automorphisms[i, :] = vec2de(np.mod(D @ col_index + b, 2).astype(int).T)
            self.inv_automorphisms[i, :] = np.argsort(self.automorphisms[i, :])
        self.automorphisms = torch.tensor(self.automorphisms)
        self.inv_automorphisms = torch.tensor(self.inv_automorphisms)
        if self.use_cuda:
            self.automorphisms = self.automorphisms.cuda()
            self.inv_automorphisms = self.inv_automorphisms.cuda()


class BCH_Code(LinearCode):
    def __init__(self, n, k, mode='', use_cuda=True):
        mat = scipy.io.loadmat(os.path.join(os.path.split(__file__)[0], 'BCH_matrices.mat'))
        code_name, H_sfx = f'BCH_{n}_{k}', mode if mode in ['sq', 'cr'] else ''
        G, H = mat[f'{code_name}_G'], mat[f'{code_name}_H{H_sfx}']
        super(BCH_Code, self).__init__(code_name, G, H, use_cuda)
        self.suffix = H_sfx
        print('Successfully created linear code', repr(self))

    def generate_automorphism(self, perm_num=10000):
        m, col_index = int(np.log2(self.N + 1)), np.arange(self.N, dtype=int)
        self.automorphisms = np.zeros((perm_num, self.N), dtype=int)
        self.inv_automorphisms = np.zeros((perm_num, self.N), dtype=int)
        for i in range(perm_num):
            # Randomly pick a and i for mapping z -> a * z^(2^i)
            a, j = np.random.randint(low=1, high=self.N), np.random.randint(m)
            # Find the column permutation induced by the affine transformation
            self.automorphisms[i, :] = np.mod(col_index * (2**j) + a, self.N)
            self.inv_automorphisms[i, :] = np.argsort(self.automorphisms[i, :])
        self.automorphisms = torch.tensor(self.automorphisms)
        self.inv_automorphisms = torch.tensor(self.inv_automorphisms)
        if self.use_cuda:
            self.automorphisms = self.automorphisms.cuda()
            self.inv_automorphisms = self.inv_automorphisms.cuda()


class EGolay24_Code(LinearCode):
    def __init__(self, cycle_reduced=True, use_cuda=True):
        H = np.array([
            list(map(int, '100110101111000001010011')),
            list(map(int, '110011010111100000101001')),
            list(map(int, '011001101011110000010101')),
            list(map(int, '001100110101111000001011')),
            list(map(int, '100110011010111100000101')),
            list(map(int, '010011001101011110000011')),
            list(map(int, '101001100110101111000001')),
            list(map(int, '010100110011010111100001')),
            list(map(int, '001010011001101011110001')),
            list(map(int, '000101001100110101111001')),
            list(map(int, '000010100110011010111101')),
            list(map(int, '111111111111111111111111'))
        ])
        H_cr = np.array([
            list(map(int, '100110101111000001010011')),
            list(map(int, '010010001100010000101001')),
            list(map(int, '111000110000000000010101')),
            list(map(int, '000100100001101010000110')),
            list(map(int, '100001011011110000000000')),
            list(map(int, '000001000001001110101010')),
            list(map(int, '101001100110101111000001')),
            list(map(int, '001101011000100111110100')),
            list(map(int, '000111000001001100000101')),
            list(map(int, '001000010100010010001101')),
            list(map(int, '001010110010001000110000')),
            list(map(int, '110101100110010100001110'))
        ])
        G, H = np.zeros((12, 24)), H_cr if cycle_reduced else H
        super(EGolay24_Code, self).__init__('EGolay24', G, H, use_cuda)
        print('Successfully created linear code', repr(self))

    def generate_automorphism(self, perm_num=10000):
        # Four generators of Mathiue group M24
        S = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 0, 23], dtype=int)
        V = np.array([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23], dtype=int)
        T = np.array([23, 22, 11, 15, 17, 9, 19, 13, 20, 5, 16, 2, 21, 7, 18, 3, 10, 4, 14, 6, 8, 12, 1, 0], dtype=int)
        W = np.array([23, 17, 22, 15, 19, 8, 14, 12, 7, 20, 9, 1, 10, 21, 2, 3, 5, 6, 11, 18, 13, 16, 4, 0], dtype=int)
        SS = [S, V, T, W]
        self.automorphisms = np.zeros((perm_num, 24), dtype=int)
        self.inv_automorphisms = np.zeros((perm_num, self.N), dtype=int)
        for k in range(perm_num):
            i, j = np.random.choice(4, size=2, replace=False)
            SS[i] = SS[i][SS[j]]
            self.automorphisms[k, :] = SS[i]
            self.inv_automorphisms[k, :] = np.argsort(self.automorphisms[k, :])
        self.automorphisms = torch.tensor(self.automorphisms)
        self.inv_automorphisms = torch.tensor(self.inv_automorphisms)
        if self.use_cuda:
            self.automorphisms = self.automorphisms.cuda()
            self.inv_automorphisms = self.inv_automorphisms.cuda()
