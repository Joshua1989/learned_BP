import math
import numpy as np
from scipy.ndimage.interpolation import shift
from functools import reduce
from copy import deepcopy
from collections import defaultdict

display_mode = 'int'  # full/int/exp/poly
check_field_mismatch = False


class GFException(Exception):
    def __init__(self, msg):
        self.msg = msg


def de2vec(x, base=2, n=1):
    X = np.array(x, dtype=int)
    if round(n) != n or n < np.log(X.max() + 1e-6) / np.log(base):
        n = int(math.ceil(np.log(X.max() + 1e-6) / np.log(base)))
    ans = np.zeros((X.size, n), dtype=int)
    for i in range(n):
        ans[:, i], X = np.mod(X, base), X // base
    return ans[0] if isinstance(x, int) else ans


def vec2de(v, base=2):
    scalar_output, V = False, np.mod(np.array(v, dtype=int), base)
    if len(V.shape) == 1:
        scalar_output, V = True, V.reshape((1, -1))
    ans = V @ base ** np.arange(V.shape[1], dtype=int)
    return ans.item(0) if scalar_output else ans


def print_poly(poly, sym='D'):
    poly = np.array(poly, dtype=int)
    if np.all(poly == 0):
        return '0'
    else:
        def monomial(c, i): return int(c != 0) * '{0}{1}'.format(
            str(c) if (c != 1 or i == 0) else '',
            sym * (i > 0) + ('^' + str(i)) * (i > 1)
        )
        monimials = [monomial(c, i) for i, c in enumerate(poly) if monomial(c, i)]
        return ' + '.join([m for m in monimials if m])


GF_default_irreducible_polys = {
    2: {
        1: [[0, 1], [1, 1]],
        2: [[1, 1, 1]],
        3: [[1, 1, 0, 1], [1, 0, 1, 1]],
        4: [[1, 1, 0, 0, 1], [1, 0, 0, 1, 1], [1, 1, 1, 1, 1]],
        5: [[1, 0, 1, 0, 0, 1],
            [1, 0, 0, 1, 0, 1],
            [1, 1, 1, 1, 0, 1],
            [1, 1, 1, 0, 1, 1],
            [1, 1, 0, 1, 1, 1],
            [1, 0, 1, 1, 1, 1]],
        6: [[1, 1, 0, 0, 0, 0, 1],
            [1, 0, 0, 1, 0, 0, 1],
            [1, 1, 1, 0, 1, 0, 1],
            [1, 1, 0, 1, 1, 0, 1],
            [1, 0, 0, 0, 0, 1, 1],
            [1, 1, 1, 0, 0, 1, 1],
            [1, 0, 1, 1, 0, 1, 1],
            [1, 1, 0, 0, 1, 1, 1],
            [1, 0, 1, 0, 1, 1, 1]],
        7: [[1, 1, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 1, 0, 0, 0, 1],
            [1, 1, 1, 1, 0, 0, 0, 1],
            [1, 0, 0, 0, 1, 0, 0, 1],
            [1, 0, 1, 1, 1, 0, 0, 1],
            [1, 1, 1, 0, 0, 1, 0, 1],
            [1, 1, 0, 1, 0, 1, 0, 1],
            [1, 0, 0, 1, 1, 1, 0, 1],
            [1, 1, 1, 1, 1, 1, 0, 1],
            [1, 0, 0, 0, 0, 0, 1, 1],
            [1, 1, 0, 1, 0, 0, 1, 1],
            [1, 1, 0, 0, 1, 0, 1, 1],
            [1, 0, 1, 0, 1, 0, 1, 1],
            [1, 0, 1, 0, 0, 1, 1, 1],
            [1, 1, 1, 1, 0, 1, 1, 1],
            [1, 0, 0, 0, 1, 1, 1, 1],
            [1, 1, 1, 0, 1, 1, 1, 1],
            [1, 0, 1, 1, 1, 1, 1, 1]],
        8: [[1, 1, 0, 1, 1, 0, 0, 0, 1],
            [1, 0, 1, 1, 1, 0, 0, 0, 1],
            [1, 1, 0, 1, 0, 1, 0, 0, 1],
            [1, 0, 1, 1, 0, 1, 0, 0, 1],
            [1, 0, 0, 1, 1, 1, 0, 0, 1],
            [1, 1, 1, 1, 1, 1, 0, 0, 1],
            [1, 0, 1, 1, 0, 0, 1, 0, 1],
            [1, 1, 1, 1, 1, 0, 1, 0, 1],
            [1, 1, 0, 0, 0, 1, 1, 0, 1],
            [1, 0, 1, 0, 0, 1, 1, 0, 1],
            [1, 0, 0, 1, 0, 1, 1, 0, 1],
            [1, 0, 0, 0, 1, 1, 1, 0, 1],
            [1, 1, 1, 0, 1, 1, 1, 0, 1],
            [1, 1, 0, 1, 1, 1, 1, 0, 1],
            [1, 1, 1, 0, 0, 0, 0, 1, 1],
            [1, 1, 0, 1, 0, 0, 0, 1, 1],
            [1, 0, 1, 1, 0, 0, 0, 1, 1],
            [1, 1, 1, 1, 1, 0, 0, 1, 1],
            [1, 1, 0, 0, 0, 1, 0, 1, 1],
            [1, 0, 0, 1, 0, 1, 0, 1, 1],
            [1, 0, 0, 0, 1, 1, 0, 1, 1],
            [1, 0, 1, 1, 1, 1, 0, 1, 1],
            [1, 1, 0, 0, 0, 0, 1, 1, 1],
            [1, 1, 1, 1, 0, 0, 1, 1, 1],
            [1, 1, 1, 0, 1, 0, 1, 1, 1],
            [1, 0, 1, 1, 1, 0, 1, 1, 1],
            [1, 1, 1, 0, 0, 1, 1, 1, 1],
            [1, 1, 0, 0, 1, 1, 1, 1, 1],
            [1, 0, 1, 0, 1, 1, 1, 1, 1],
            [1, 0, 0, 1, 1, 1, 1, 1, 1]],
        9: [[1, 1, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 1, 0, 0, 0, 0, 1],
            [1, 1, 1, 0, 1, 0, 0, 0, 0, 1],
            [1, 1, 0, 1, 1, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 1, 0, 0, 0, 1],
            [1, 0, 1, 1, 0, 1, 0, 0, 0, 1],
            [1, 1, 0, 0, 1, 1, 0, 0, 0, 1],
            [1, 1, 0, 1, 0, 0, 1, 0, 0, 1],
            [1, 0, 0, 1, 1, 0, 1, 0, 0, 1],
            [1, 1, 1, 1, 1, 0, 1, 0, 0, 1],
            [1, 0, 1, 0, 0, 1, 1, 0, 0, 1],
            [1, 0, 0, 1, 0, 1, 1, 0, 0, 1],
            [1, 1, 1, 1, 0, 1, 1, 0, 0, 1],
            [1, 1, 1, 0, 1, 1, 1, 0, 0, 1],
            [1, 0, 1, 1, 1, 1, 1, 0, 0, 1],
            [1, 1, 1, 0, 0, 0, 0, 1, 0, 1],
            [1, 0, 1, 0, 1, 0, 0, 1, 0, 1],
            [1, 0, 0, 1, 1, 0, 0, 1, 0, 1],
            [1, 1, 0, 0, 0, 1, 0, 1, 0, 1],
            [1, 0, 1, 0, 0, 1, 0, 1, 0, 1],
            [1, 1, 1, 1, 0, 1, 0, 1, 0, 1],
            [1, 1, 1, 0, 1, 1, 0, 1, 0, 1],
            [1, 0, 1, 1, 1, 1, 0, 1, 0, 1],
            [1, 1, 1, 1, 0, 0, 1, 1, 0, 1],
            [1, 0, 0, 0, 1, 0, 1, 1, 0, 1],
            [1, 1, 0, 1, 1, 0, 1, 1, 0, 1],
            [1, 0, 1, 0, 1, 1, 1, 1, 0, 1],
            [1, 0, 0, 1, 1, 1, 1, 1, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 1, 1],
            [1, 1, 0, 0, 1, 0, 0, 0, 1, 1],
            [1, 0, 1, 0, 1, 0, 0, 0, 1, 1],
            [1, 1, 1, 1, 1, 0, 0, 0, 1, 1],
            [1, 1, 0, 0, 0, 1, 0, 0, 1, 1],
            [1, 0, 0, 0, 1, 1, 0, 0, 1, 1],
            [1, 1, 0, 1, 1, 1, 0, 0, 1, 1],
            [1, 0, 0, 1, 0, 0, 1, 0, 1, 1],
            [1, 1, 1, 1, 0, 0, 1, 0, 1, 1],
            [1, 1, 0, 1, 1, 0, 1, 0, 1, 1],
            [1, 0, 0, 0, 0, 1, 1, 0, 1, 1],
            [1, 1, 0, 1, 0, 1, 1, 0, 1, 1],
            [1, 0, 1, 1, 0, 1, 1, 0, 1, 1],
            [1, 1, 0, 0, 1, 1, 1, 0, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 0, 1, 1],
            [1, 0, 1, 0, 0, 0, 0, 1, 1, 1],
            [1, 1, 1, 1, 0, 0, 0, 1, 1, 1],
            [1, 0, 0, 0, 0, 1, 0, 1, 1, 1],
            [1, 0, 1, 0, 1, 1, 0, 1, 1, 1],
            [1, 0, 0, 1, 1, 1, 0, 1, 1, 1],
            [1, 1, 1, 0, 0, 0, 1, 1, 1, 1],
            [1, 1, 0, 1, 0, 0, 1, 1, 1, 1],
            [1, 0, 1, 1, 0, 0, 1, 1, 1, 1],
            [1, 0, 1, 0, 1, 0, 1, 1, 1, 1],
            [1, 0, 0, 1, 1, 0, 1, 1, 1, 1],
            [1, 1, 0, 0, 0, 1, 1, 1, 1, 1],
            [1, 0, 0, 1, 0, 1, 1, 1, 1, 1],
            [1, 1, 0, 1, 1, 1, 1, 1, 1, 1]]
    },
    3: {
        1: [[0, 1], [1, 1]],
        2: [[1, 0, 1], [2, 1, 1]],
        3: [[1, 2, 0, 1], [2, 0, 1, 1], [2, 1, 1, 1]],
        4: [[2, 1, 0, 0, 1],
            [2, 0, 1, 0, 1],
            [1, 1, 1, 0, 1],
            [2, 0, 0, 1, 1],
            [1, 2, 0, 1, 1],
            [1, 0, 1, 1, 1],
            [1, 1, 1, 1, 1]],
        5: [[1, 2, 0, 0, 0, 1],
            [2, 1, 1, 0, 0, 1],
            [1, 1, 0, 1, 0, 1],
            [2, 0, 1, 1, 0, 1],
            [2, 2, 1, 1, 0, 1],
            [2, 0, 0, 0, 1, 1],
            [2, 1, 0, 0, 1, 1],
            [1, 0, 1, 0, 1, 1],
            [1, 1, 1, 0, 1, 1],
            [1, 1, 0, 1, 1, 1],
            [1, 2, 1, 1, 1, 1],
            [1, 0, 0, 2, 1, 1],
            [2, 0, 1, 2, 1, 1]],
        6: [[2, 1, 0, 0, 0, 0, 1],
            [1, 1, 1, 0, 0, 0, 1],
            [1, 0, 2, 0, 0, 0, 1],
            [2, 1, 0, 1, 0, 0, 1],
            [1, 0, 1, 1, 0, 0, 1],
            [2, 2, 1, 1, 0, 0, 1],
            [1, 0, 2, 0, 1, 0, 1],
            [2, 1, 2, 0, 1, 0, 1],
            [1, 0, 0, 1, 1, 0, 1],
            [1, 1, 0, 1, 1, 0, 1],
            [2, 2, 1, 1, 1, 0, 1],
            [1, 0, 0, 0, 2, 0, 1],
            [2, 0, 1, 1, 2, 0, 1],
            [2, 0, 0, 0, 0, 1, 1],
            [2, 1, 0, 0, 0, 1, 1],
            [1, 1, 1, 0, 0, 1, 1],
            [2, 0, 0, 1, 0, 1, 1],
            [1, 1, 0, 1, 0, 1, 1],
            [1, 0, 1, 1, 0, 1, 1],
            [2, 1, 1, 1, 0, 1, 1],
            [1, 0, 0, 0, 1, 1, 1],
            [1, 1, 0, 0, 1, 1, 1],
            [2, 2, 1, 0, 1, 1, 1],
            [2, 0, 2, 0, 1, 1, 1],
            [2, 1, 0, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1],
            [1, 0, 2, 2, 1, 1, 1],
            [1, 0, 0, 2, 2, 1, 1]]
    },
    5: {
        1: [[0, 1], [1, 1]],
        2: [[2, 0, 1], [1, 1, 1]],
        3: [[1, 1, 0, 1], [1, 0, 1, 1], [3, 1, 1, 1]],
        4: [[2, 0, 0, 0, 1],
            [4, 1, 0, 0, 1],
            [2, 0, 1, 0, 1],
            [1, 1, 1, 0, 1],
            [4, 0, 0, 1, 1],
            [3, 1, 0, 1, 1],
            [1, 0, 1, 1, 1],
            [3, 1, 1, 1, 1]]
    },
    7: {
        1: [[0, 1], [1, 1]],
        2: [[1, 0, 1], [3, 1, 1]],
        3: [[2, 0, 0, 1], [1, 1, 0, 1], [1, 0, 1, 1], [2, 1, 1, 1]]
    },
    11: {
        1: [[0, 1], [1, 1]], 2: [[1, 0, 1], [1, 1, 1]]
    },
    13: {
        1: [[0, 1], [1, 1]], 2: [[2, 0, 1], [2, 1, 1]]
    },
    17: {
        1: [[0, 1], [1, 1]], 2: [[3, 0, 1], [1, 1, 1]]
    },
    19: {
        1: [[0, 1], [1, 1]], 2: [[1, 0, 1], [2, 1, 1]]
    },
    23: {
        1: [[0, 1], [1, 1]], 2: [[1, 0, 1], [1, 1, 1]]
    },
    29: {
        1: [[0, 1], [1, 1]], 2: [[2, 0, 1], [1, 1, 1]
                                 ]},
    31: {
        1: [[0, 1], [1, 1]], 2: [[2, 0, 1], [1, 1, 1]
                                 ]}
}


class GF:
    def __init__(self, p=2, d=3, poly=None):
        if poly is None:
            try:
                poly = GF_default_irreducible_polys[p][d][0]
            except Exception:
                raise GFException('No default irreducible polynomial available for p={0}, d={1}'.format(p, d))
        poly = np.array(poly, dtype=int)
        if poly[-1] != 1:
            raise GFException('The generating polynomial must be monic and in right MSB order')
        if p < 2 or round(p) != p or not all(p % i for i in range(2, p)):
            raise GFException('Characteristic must be a prime number')
        if d < 1 or round(d) != d:
            raise GFException('Extension degree must be a positive integer')
        if not np.all(np.mod(poly, 1) == 0) or np.any(poly < 0) or np.any(poly > p - 1):
            raise GFException('Generating polynomial must have integer coefficients between 0 and {0}'.format(p - 1))
        if poly.size != d + 1:
            raise GFException('Degree of generating polynomial does not match extention degree')

        self.char, self.ext_deg = p, d
        self.irreducible_poly = poly

        # Compute x^i for i =0, 1, .., 2d for polynomial multiplication
        self.x_powers, x = {}, de2vec(1, p, d)
        for i in range(2 * d + 1):
            self.x_powers[i] = x.copy()
            x = np.mod(shift(x, 1, cval=0) + x[-1] * poly[:-1], p)

        # Find a primitive element with minimum decimal representation
        prim_elems = []
        for i in range(1, p**d):
            ele, seen = de2vec(i, p, d), set()
            x = ele.copy()
            for i in range(p**d):
                if tuple(x) in seen:
                    break
                seen.add(tuple(x))
                x = self.poly_mult(x, ele)
            if len(seen) == p**d - 1:
                prim_elems.append(ele)
        if len(prim_elems) == 0:
            raise GFException('Unable to construct field from polynomial', print_poly(poly))

        # Build list of field elements in integer order
        self.build(prim_elems[0])
        self.prim_elems = [GFElem(self, x, mode='poly') for x in prim_elems]
        self.base_prim_elem = self.prim_elems[0]
        self.normal_basis_construction()

    def poly_mult(self, a, b):
        p, d = self.characteristic(), self.extension_degree()
        temp = np.mod(np.convolve(a, b), p)
        x = temp[:d]
        for i, c in enumerate(temp[d:], d):
            x = np.mod(x - c * self.x_powers[i], p)
        return x

    def gen_add_table(self, order_by='int'):
        order = [x[1] for x in sorted([(x.exp, x.int) for x in self.elems])] if order_by == 'exp' else list(range(self.field_size()))
        if not hasattr(self, 'add_table'):
            q = self.char ** self.ext_deg
            self.add_table = [[None] * q for _ in range(q)]
            for i, a in enumerate(self.elems):
                for j, b in enumerate(self.elems):
                    c_poly = np.mod(a.poly + b.poly, self.char)
                    c_int = self.poly_lookup[tuple(c_poly)]['int']
                    self.add_table[i][j] = self.elems[c_int]
            self.add_table = np.array(self.add_table)
        return self.add_table[:, order][order]

    def gen_sub_table(self, order_by='int'):
        order = [x[1] for x in sorted([(x.exp, x.int) for x in self.elems])] if order_by == 'exp' else list(range(self.field_size()))
        if not hasattr(self, 'sub_table'):
            q = self.char ** self.ext_deg
            self.sub_table = [[None] * q for _ in range(q)]
            for i, a in enumerate(self.elems):
                for j, b in enumerate(self.elems):
                    c_poly = np.mod(a.poly - b.poly, self.char)
                    c_int = self.poly_lookup[tuple(c_poly)]['int']
                    self.sub_table[i][j] = self.elems[c_int]
            self.sub_table = np.array(self.sub_table)
        return self.sub_table[:, order][order]

    def gen_mul_table(self, order_by='int'):
        order = [x[1] for x in sorted([(x.exp, x.int) for x in self.elems])] if order_by == 'exp' else list(range(self.field_size()))
        if not hasattr(self, 'mul_table'):
            q = self.char ** self.ext_deg
            self.mul_table = [[None] * q for _ in range(q)]
            for i, a in enumerate(self.elems):
                for j, b in enumerate(self.elems):
                    if a.exp == -np.inf or b.exp == -np.inf:
                        self.mul_table[i][j] = self.elems[0]
                    else:
                        c_exp = np.mod(a.exp + b.exp, q - 1)
                        c_int = self.exp_lookup[c_exp]['int']
                        self.mul_table[i][j] = self.elems[c_int]
            self.mul_table = np.array(self.mul_table)
        return self.mul_table[:, order][order]

    def gen_neg_table(self, order_by='int'):
        order = [x[1] for x in sorted([(x.exp, x.int) for x in self.elems])] if order_by == 'exp' else list(range(self.field_size()))
        if not hasattr(self, 'neg_table'):
            q = self.char ** self.ext_deg
            self.neg_table = [[None] * q for _ in range(q)]
            for i, a in enumerate(self.elems):
                c_poly = np.mod(-a.poly, self.char)
                c_int = self.poly_lookup[tuple(c_poly)]['int']
                self.neg_table[i] = self.elems[c_int]
            self.neg_table = np.array(self.neg_table)
        return self.neg_table[order]

    def gen_pow_table(self, order_by='int'):
        order = [x[1] for x in sorted([(x.exp, x.int) for x in self.elems])] if order_by == 'exp' else list(range(self.field_size()))
        if not hasattr(self, 'pow_table'):
            q = self.char ** self.ext_deg
            self.pow_table = [[None] * (q - 1) for _ in range(q)]
            for i, a in enumerate(self.elems):
                for t in range(q - 1):
                    if t == 0:
                        self.pow_table[i][t] = self.elems[1]
                    else:
                        if a.exp == -np.inf:
                            self.pow_table[i][t] = self.elems[0]
                        else:
                            c_exp = np.mod(a.exp * t, q - 1)
                            c_int = self.exp_lookup[c_exp]['int']
                            self.pow_table[i][t] = self.elems[c_int]
            self.pow_table = np.array(self.pow_table)
        return self.pow_table[order]

    def gen_inverseFrobenius_table(self, order_by='int'):
        order = [x[1] for x in sorted([(x.exp, x.int) for x in self.elems])] if order_by == 'exp' else list(range(self.field_size()))
        if not hasattr(self, 'inv_frob_table'):
            p, q = self.char, self.char ** self.ext_deg
            self.inv_frob_table = [[None] * q for _ in range(q)]
            for i, a in enumerate(self.elems):
                c_exp = -np.inf if a.exp == -np.inf else np.mod(a.exp * p, q - 1)
                c_int = self.exp_lookup[c_exp]['int']
                self.inv_frob_table[c_int] = self.elems[i]
            self.inv_frob_table = np.array(self.inv_frob_table)
        return self.inv_frob_table[order]

    def gen_tables(self):
        self.gen_add_table()
        self.gen_sub_table()
        self.gen_mul_table()
        self.gen_neg_table()
        self.gen_pow_table()
        self.gen_inverseFrobenius_table()

    def build(self, poly):
        # Build table for field elements for exponent of primitive element, integer represent, polynomial
        p, d = self.characteristic(), self.extension_degree()
        table = {'exp': [-np.inf, 0], 'int': [0, 1], 'poly': [de2vec(0, p, d), de2vec(1, p, d)]}
        x = poly.copy()
        for i in range(1, p**d - 1):
            table['exp'].append(i)
            table['int'].append(vec2de(x, base=p))
            table['poly'].append(x)
            x = self.poly_mult(x, poly)
        # Build hash table indexed by different representations
        self.exp_lookup, self.int_lookup, self.poly_lookup = {}, {}, {}
        for e, i, p in zip(table['exp'], table['int'], table['poly']):
            item = {'exp': e, 'int': i, 'poly': p}
            self.exp_lookup[e] = item
            self.int_lookup[i] = item
            self.poly_lookup[tuple(p)] = item
        self.elems = np.array([GFElem(self, i) for i in range(self.field_size())])
        if hasattr(self, 'prim_elems'):
            self.prim_elems = [GFElem(self, x.int) for x in self.prim_elems]
        self.gen_tables()

    def normal_basis_construction(self, alpha=None):
        p, d = self.characteristic(), self.extension_degree()
        # If normal basis is not specified, set it to be the current base primitive element if it
        # can form a normal basis, otherwise pick one from primitive elements

        def pre_check(alpha):
            # Check if field element alpha can form normal basis according to three necessary conditions
            thm7 = not (alpha.trace() == self.elems[0])
            if not thm7:
                return False
            thm8 = d == 1 or any((alpha * alpha ** (2**i)).trace() == self.elems[0] for i in range(d))
            if not thm8:
                return False
            thm9 = not (d % 2 == 0 and alpha.trace() == (alpha ** (2**(d // 2) + 1)).trace())
            return thm9

        def construct_F_matrix(alpha):
            # Construct the F matrix corresponding to alpha, where F_{ij} = Tr( alpha^(2^i) alpha^(2^j) )
            alpha_vec = [alpha**(p**i) for i in range(d)]
            F = zeros(self, (d, d))
            for i in range(d):
                for j in range(d):
                    F[i, j] = (alpha_vec[i] * alpha_vec[j]).trace()
            return F

        def total_check(alpha):
            # Check if alpha can form a normal basis, if so, return corresponding dual basis
            if p == 2 and not pre_check(alpha):
                return None
            F = construct_F_matrix(alpha)
            _, F_inv, rank = rref(F)
            if rank < d:
                return None
            else:
                beta_vec = F_inv @ GFMat([alpha**(p**i) for i in range(d)]).reshape((-1, 1))
                return beta_vec.item(0)
        if alpha is None:
            beta = total_check(self.base_prim_elem)
            if beta is not None:
                self.normal_basis = self.base_prim_elem
                self.dual_basis = beta
            else:
                for x in self.prim_elems:
                    if x == self.base_prim_elem:
                        continue
                    beta = total_check(x)
                    if beta is not None:
                        self.normal_basis, self.dual_basis = x, beta
        else:
            beta = total_check(alpha)
            if beta is None:
                raise GFException('{0} cannot form a normal basis'.format(alpha))
            else:
                self.normal_basis, self.dual_basis = alpha, beta

        # For each linear combination, compute the value and assign it to corresponding field elements
        alpha_vec = [self.normal_basis**(p**i) for i in range(d)]
        beta_vec = [self.dual_basis**(p**i) for i in range(d)]
        for x in range(self.field_size()):
            coeff = de2vec(x, base=p, n=d)
            res = reduce(lambda x, y: x + y, [c * b for c, b in zip(coeff, alpha_vec)])
            self.elems[res.int].normal_repr = coeff
            res = reduce(lambda x, y: x + y, [c * b for c, b in zip(coeff, beta_vec)])
            self.elems[res.int].dual_repr = coeff

    def __repr__(self):
        p, d = self.char, self.ext_deg
        poly, prim_elem = print_poly(self.irreducible_poly), print_poly(self.base_prim_elem.poly)
        return 'GF({0},{1}) generated by {2}, primitive element {3}'.format(p, d, poly, prim_elem)

    def __getitem__(self, sliced):
        return self.elems[sliced]

    def __eq__(self, other):
        if not isinstance(other, GF):
            return False
        if self is other:
            return True
        return self.char == other.char and self.ext_deg == other.ext_deg \
            and tuple(self.irreducible_poly) == tuple(other.irreducible_poly) \
            and self.base_prim_elem.int == other.base_prim_elem.int \
            and self.base_prim_elem.exp == other.base_prim_elem.exp \
            and tuple(self.base_prim_elem.poly) == tuple(other.base_prim_elem.poly)

    def __ne__(self, other):
        return not (self == other)

    def change_base(self, base_elem):
        if base_elem not in self.prim_elems:
            raise GFException('input is not a primitive element')
        elif base_elem != self.base_prim_elem:
            self.base_prim_elem = base_elem
            self.build(base_elem.poly)
        self.normal_basis_construction()

    def characteristic(self):
        return self.char

    def extension_degree(self):
        return self.ext_deg

    def field_size(self):
        return self.char ** self.ext_deg

    def field_irreducible(self):
        return self.irreducible_poly

    def conjugacy_list(self):
        visited, ans = set(), []
        for i in range(self.field_size()):
            if self[i] in visited:
                continue
            ans.append(self[i].conjugacy_class())
            visited |= set(ans[-1])
        return ans


class GFElem:
    def __init__(self, field, val, mode='int'):
        if mode == 'int':
            try:
                item = field.int_lookup[val]
            except Exception:
                raise GFException('Value must be integer between 0 and {0}'.format(field.field_size() - 1))
        elif mode == 'exp':
            try:
                item = field.exp_lookup[val]
            except Exception:
                raise GFException('Value must be -inf or integer between 0 and {0}'.format(field.field_size() - 2))
        elif mode == 'poly':
            try:
                item = field.poly_lookup[tuple(val)]
            except Exception:
                raise GFException('Polynomial must have integer coefficients between 0 and {0}'.format(field.characteristic() - 1))
        else:
            raise GFException('Unrecognized initialization mode')
        self.field = field
        self.exp = item['exp']
        self.int = item['int']
        self.poly = item['poly']
        self.normal_repr = None
        self.dual_repr = None

    def to_str(self, disp_mode=None):
        if disp_mode is None:
            disp_mode = display_mode
        if disp_mode == 'full':
            return 'exp: {0}, int: {1}, poly: {2}'.format(self.exp, self.int, print_poly(self.poly))
        elif disp_mode == 'int':
            return str(self.int)
        elif disp_mode == 'exp':
            return '0' if self.exp == -np.inf else '1' if self.exp == 0 else 'a' if self.exp == 1 else 'a^' + str(self.exp)
        elif disp_mode == 'poly':
            return print_poly(self.poly)
        else:
            return NotImplemented

    def __repr__(self):
        return self.to_str()

    def __pos__(self):
        return self.field.elems[self.int]

    def __neg__(self):
        return self.field.neg_table[self.int]

    def sqrt(self):
        if self.exp == -np.inf:
            return GFElem(self.field, -np.inf, mode='exp')
        else:
            val = (self.exp + self.field.field_size() - 1) // 2 if self.exp % 2 else self.exp // 2
            return GFElem(self.field, val, mode='exp')

    def inv_frob(self):
        return self.field.inv_frob_table[self.int]

    def trace(self):
        p, d = self.field.characteristic(), self.field.extension_degree()
        pow_list = [self ** (p**i) for i in range(d)]
        return reduce(lambda x, y: x + y, pow_list, self.field[0])

    def __add__(self, other):
        if isinstance(other, GFElem):
            if check_field_mismatch:
                if self.field != other.field:
                    raise GFException('field or base primitive element mismatch between two operands')
            return self.field.add_table[self.int, other.int]
        elif isinstance(other, int):
            other %= self.field.characteristic()
            return self.field.add_table[self.int, other]
        else:
            return NotImplemented

    def __radd__(self, other):
        return self + other

    def __iadd__(self, other):
        return self + other

    def __sub__(self, other):
        if isinstance(other, GFElem):
            if check_field_mismatch:
                if self.field != other.field:
                    raise GFException('field or base primitive element mismatch between two operands')
            return self.field.sub_table[self.int, other.int]
        elif isinstance(other, int):
            other %= self.field.characteristic()
            return self.field.sub_table[self.int, other]
        else:
            return NotImplemented

    def __rsub__(self, other):
        return other + (-self)

    def __isub__(self, other):
        return self - other

    def __mul__(self, other):
        if isinstance(other, GFElem):
            if check_field_mismatch:
                if self.field != other.field:
                    raise GFException('field or base primitive element mismatch between two operands')
            return self.field.mul_table[self.int, other.int]
        elif isinstance(other, int):
            other %= self.field.characteristic()
            return self.field.mul_table[self.int, other]
        else:
            return NotImplemented

    def __rmul__(self, other):
        return self * other

    def __imul__(self, other):
        return self * other

    def __truediv__(self, other):
        if isinstance(other, GFElem):
            if check_field_mismatch:
                if self.field != other.field:
                    raise GFException('field or base primitive element mismatch between two operands')
            return self * other**(-1)
        elif isinstance(other, int):
            other %= self.field.characteristic()
            return self * GFElem(self.field, other)**(-1)
        else:
            return NotImplemented

    def __rtruediv__(self, other):
        if isinstance(other, GFElem):
            if check_field_mismatch:
                if self.field != other.field:
                    raise GFException('field or base primitive element mismatch between two operands')
            return other * self**(-1)
        elif isinstance(other, int):
            other %= self.field.characteristic()
            return other * self**(-1)
        else:
            return NotImplemented

    def __itruediv__(self, other):
        return self * other**(-1)

    def __pow__(self, other):
        if isinstance(other, int):
            if self.exp != -np.inf or (self.exp == -np.inf and other >= 0):
                other %= self.field.field_size() - 1
                return self.field.pow_table[self.int][other]
            else:
                raise GFException('divide by zero is invalid')
        else:
            return NotImplemented

    def __ipow__(self, other):
        return self ** other

    def __eq__(self, other):
        if isinstance(other, GFElem):
            if check_field_mismatch:
                if self.field != other.field:
                    raise GFException('field or base primitive element mismatch between two operands')
            return (self.int, self.exp, tuple(self.poly)) == (other.int, other.exp, tuple(other.poly))
        else:
            return NotImplemented

    def __ne__(self, other):
        return not (self == other)

    def __hash__(self):
        return hash((self.int, self.exp, tuple(self.poly)))

    def __int__(self):
        return int(self.int)

    def conjugacy_class(self):
        conj_cls, x = [], self
        while x not in conj_cls:
            conj_cls.append(x)
            x = x ** 2
        return conj_cls

    def minimal_poly(self):
        min_poly = reduce(np.convolve, [[x, self.field[1]] for x in self.conjugacy_class()])
        return np.array([x.int for x in min_poly], dtype=int)

    def vector_repr(self, dual=False):
        vec_repr = self.normal_repr if not dual else self.dual_repr
        return fromint(self.field, vec_repr)

    def matrix_repr(self, dual=False):
        if self.normal_repr is None or self.dual_repr is None:
            return None
        alpha = self.field.normal_basis if not dual else self.field.dual_basis
        p, d = self.field.characteristic(), self.field.extension_degree()
        col_elems = [self * alpha ** (p**i) for i in range(d)]
        cols = [fromint(self.field, x.vector_repr()).reshape((-1, 1)) for x in col_elems]
        return bmat(cols)


class GFMat(np.ndarray):
    def __new__(cls, input_array):
        ans = np.asarray(input_array).view(cls)
        ans.field = ans.item(0).field
        return ans

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.field = getattr(obj, 'field', None)

    def __array_wrap__(self, out_arr, context=None):
        return np.ndarray.__array_wrap__(self, out_arr, context)

    def __matmul__(self, other):
        if isinstance(other, GFMat):
            if check_field_mismatch:
                if self.field != other.field:
                    raise GFException('field or base primitive element mismatch between two operands')
            return np.dot(self, other)
        else:
            return NotImplemented

    def copy(self):
        return deepcopy(self)


def fromint(field, M):
    vfunc = np.vectorize(lambda x: field[int(x)])
    return GFMat(vfunc(M))


def fromexp(field, M):
    vfunc = np.vectorize(lambda x: GFElem(field, int(x) if x != -np.inf else -np.inf, mode='exp'))
    return GFMat(vfunc(M))


def triu(M, k=0):
    return M * fromint(M.field, np.triu(np.ones(shape=M.shape), k=k))


def tril(M, k=0):
    return M * fromint(M.field, np.tril(np.ones(shape=M.shape), k=k))


def diag(M, k=0):
    if len(M) == 2:
        return np.diag(M, k)
    else:
        field = M.item(0).field
        vfunc = np.vectorize(lambda x: field[int(x)] if isinstance(x, int) else x)
        return GFMat(vfunc(np.diag(M, k)))


def random(field, shape, sparsity=1, force_full_rank=False):
    vfunc = np.vectorize(lambda x: field[int(x)])
    while True:
        A = GFMat(vfunc(np.random.randint(low=0, high=field.field_size(), size=shape)))
        A *= (np.random.random(A.shape) < sparsity).astype(int)
        if not force_full_rank or gfrank(A) == min(shape):
            break
    return A


def eye(field, N, M=None, k=0):
    vfunc = np.vectorize(lambda x: field[int(x)])
    return GFMat(vfunc(np.eye(N, M or N, k)))


def ones(field, shape):
    vfunc = np.vectorize(lambda x: field[int(x)])
    return GFMat(vfunc(np.ones(shape)))


def zeros(field, shape):
    vfunc = np.vectorize(lambda x: field[int(x)])
    return GFMat(vfunc(np.zeros(shape)))


def bmat(blks):
    return GFMat(np.bmat(blks))


def zfill(x, n, mode='b'):
    x = np.array(x)
    f0 = x.item(0).field[0]
    if n > x.size:
        if mode == 'b':
            x = np.concatenate((x, [f0] * (n - x.size)))
        elif mode == 'f':
            x = np.concatenate(([f0] * (n - x.size), x))
        else:
            return NotImplemented
    return x


def ref(A):
    (m, n), j = A.shape, 0
    Ar, rank = bmat([A, eye(A.field, m)]), 0
    for i in range(min(m, n)):
        # Find value and index of non-zero element in the remainder of column i.
        while j < n:
            temp = np.where(Ar[i:, j] != A.field[0])[0]
            if len(temp) == 0:
                # If the lower half of j-th row is all-zero, check next column
                j += 1
            else:
                # Swap i-th and k-th rows
                k, rank = temp[0] + i, rank + 1
                if i != k:
                    Ar[[i, k], j:] = Ar[[k, i], j:]
                # Save the right hand side of the pivot row
                pivot = Ar[i, j]
                row = Ar[i, j:].reshape((1, -1)) * pivot**(-1)
                col = bmat([zeros(A.field, i + 1), Ar[i + 1:, j]]).reshape((-1, 1))
                Ar[:, j:] -= col * row
                Ar[i, j:] *= pivot**(-1)
                break
        j += 1
    E, X = GFMat(Ar[:, :n]), GFMat(Ar[:, n:])
    return E, X, rank


def rref(A):
    (m, n), j = A.shape, 0
    Ar, rank = bmat([A, eye(A.field, m)]), 0
    for i in range(min(m, n)):
        # Find value and index of non-zero element in the remainder of column i.
        while j < n:
            temp = np.where(Ar[i:, j] != A.field[0])[0]
            if len(temp) == 0:
                # If the lower half of j-th row is all-zero, check next column
                j += 1
            else:
                # Swap i-th and k-th rows
                k, rank = temp[0] + i, rank + 1
                if i != k:
                    Ar[[i, k], j:] = Ar[[k, i], j:]
                # Save the right hand side of the pivot row
                pivot = Ar[i, j]
                row = Ar[i, j:].reshape((1, -1)) * pivot**(-1)
                col = bmat([Ar[:i, j], [A.field[0]], Ar[i + 1:, j]]).reshape((-1, 1))
                Ar[:, j:] -= col * row
                Ar[i, j:] *= pivot**(-1)
                break
        j += 1
    R, Y = GFMat(Ar[:, :n]), GFMat(Ar[:, n:])
    return R, Y, rank


def gfrank(A):
    E, X, rank = ref(A)
    return rank


def inv(A):
    m, n = A.shape
    if m != n:
        raise GFException('matrix must be square')
    R, Y, rank = rref(A)
    if rank < max(m, n):
        raise GFException('matrix is singular')
    return Y


class GFPoly:
    def __init__(self, poly):
        if len(poly) == 0:
            raise GFException('empty input is invalid')
        self.field, self.coeff = poly[0].field, np.array(poly)
        temp = np.where(self.coeff != self.field[0])[0]
        if len(temp) == 0:
            self.coeff = np.array([self.field[0]])
        else:
            self.coeff = self.coeff[:temp.max() + 1]
        self.alpha = self.coeff[0].field.base_prim_elem

    def to_str(self, disp_mode=None):
        if disp_mode is None:
            disp_mode = display_mode
        if self.coeff.size == 1 and self.coeff[0] == self.field[0]:
            return '0'
        terms = []
        for i, elem in enumerate(self.coeff):
            if elem == self.field[0]:
                continue
            c_str = elem.to_str(disp_mode)
            if disp_mode == 'poly':
                c_str = c_str if '+' not in c_str else '(' + c_str + ')'
            x_str = ('x^' + str(i)) if i > 1 else 'x' if i == 1 else ''
            terms.append(c_str if x_str == '' else x_str if c_str == '1' else c_str + ' ' + x_str)
        return ' + '.join(terms)

    def __repr__(self):
        return self.to_str()

    def __neg__(self):
        return GFPoly(deepcopy(-self.coeff))

    def __pos__(self):
        return GFPoly(deepcopy(self.coeff))

    def __add__(self, other):
        if isinstance(other, GFPoly):
            if check_field_mismatch:
                if self.field != other.field:
                    raise GFException('field or base primitive element mismatch between two operands')
            n = max(self.coeff.size, other.coeff.size) + 1
            return GFPoly(zfill(self.coeff, n) + zfill(other.coeff, n))
        elif isinstance(other, GFElem):
            if check_field_mismatch:
                if self.field != other.field:
                    raise GFException('field or base primitive element mismatch between two operands')
            poly = deepcopy(self.coeff)
            poly[0] += other
            return GFPoly(poly)
        elif isinstance(other, int):
            poly = deepcopy(self.coeff)
            poly[0] += self.field[other % self.field.characteristic()]
            return GFPoly(poly)
        else:
            return NotImplemented

    def __radd__(self, other):
        return self + other

    def __iadd__(self, other):
        return self + other

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __isub__(self, other):
        return self + (-other)

    def __mul__(self, other):
        if isinstance(other, GFPoly):
            if check_field_mismatch:
                if self.field != other.field:
                    raise GFException('field or base primitive element mismatch between two operands')
            return GFPoly(np.convolve(self.coeff, other.coeff))
        elif isinstance(other, GFElem):
            if check_field_mismatch:
                if self.field != other.field:
                    raise GFException('field or base primitive element mismatch between two operands')
            return GFPoly(other * self.coeff)
        elif isinstance(other, int):
            return GFPoly(other * self.coeff)
        else:
            return NotImplemented

    def __rmul__(self, other):
        return self * other

    def __imul__(self, other):
        return self * other

    def __divmod__(self, other):
        if isinstance(other, GFPoly):
            if check_field_mismatch:
                if self.field != other.field:
                    raise GFException('field or base primitive element mismatch between two operands')
            if self.degree() < other.degree():
                return GFPoly([self.field[0]]), GFPoly(deepcopy(self.coeff))
            elif other.degree() == 0:
                return divmod(self, other.coeff[0])
            else:
                q_coeff = [self.field[0] for _ in range(self.degree() - other.degree() + 1)]
                r = GFPoly(deepcopy(self.coeff))
                while r.degree() >= other.degree():
                    d = r.degree() - other.degree()
                    q_coeff[d] = r.coeff[-1] / other.coeff[-1]
                    r -= GFPoly(zfill([q_coeff[d]], d + 1, mode='f')) * other
                return GFPoly(q_coeff), r
        elif isinstance(other, GFElem):
            if check_field_mismatch:
                if self.field != other.field:
                    raise GFException('field or base primitive element mismatch between two operands')
            return self * other**(-1), GFPoly([self.field[0]])
        elif isinstance(other, int):
            other %= self.field.characteristic()
            return self * GFElem(self.field, other)**(-1), GFPoly([self.field[0]])
        else:
            return NotImplemented

    def __floordiv__(self, other):
        q, r = divmod(self, other)
        return q

    def __ifloordiv__(self, other):
        return self // other

    def __mod__(self, other):
        q, r = divmod(self, other)
        return r

    def __imod__(self, other):
        return self % other

    def __pow__(self, other):
        if isinstance(other, int) and other >= 0:
            poly = np.array([self.field[1]])
            for i in range(other):
                poly = np.convolve(poly, self.coeff)
            return GFPoly(poly)
        else:
            return NotImplemented

    def __ipow__(self, other):
        return self ** other

    def __eq__(self, other):
        if isinstance(other, GFPoly):
            if check_field_mismatch:
                if self.field != other.field:
                    raise GFException('field or base primitive element mismatch between two operands')
            return np.all(self.coeff == other.coeff)
        else:
            return NotImplemented

    def __ne__(self, other):
        return not (self == other)

    def __hash__(self):
        return hash(tuple(self.coeff.astype(int)))

    def copy(self):
        return GFPoly(self.coeff.copy())

    def degree(self):
        return self.coeff.size - 1

    def monic(self):
        return self.copy() if self.is_zero() else GFPoly(self.coeff[-1]**(-1) * self.coeff)

    def is_zero(self):
        return self.coeff[-1] == self.field[0]

    def is_one(self):
        return self.coeff[-1] == self.field[1] and self.coeff.size == 1

    def derivative(self):
        coeff = [i * x for i, x in enumerate(self.coeff[1:], 1)]
        if len(coeff) == 0:
            coeff = [self.field[0]]
        return GFPoly(coeff)

    def inv_frob(self):
        if self.is_zero():
            return self.copy()
        p = self.field.characteristic()
        index = np.where(self.coeff != self.field[0])[0]
        if not np.all(np.mod(index, p) == 0):
            raise GFException('This polynomial is not the {0}-th power of some other polynomial'.format(p))
        coeff = [self.field[0]] * (1 + self.degree() // p)
        for i in index:
            coeff[i // p] = self.coeff[i].inv_frob()
        return GFPoly(coeff)

    def eval(self, x):
        scalar_output = False
        if isinstance(x, int) or isinstance(x, GFElem):
            scalar_output = True
        x = np.array(x).reshape(1) if scalar_output else np.array(x)
        V = np.vander(x, N=self.degree() + 1, increasing=True)
        ans = np.dot(V, self.coeff)
        return ans[0] if scalar_output else ans

    def roots(self):
        vals = self.eval(self.field[:])
        return np.array(self.field[:])[vals == self.field[0]]


def poly_gcd(f, g):
    q, r = f.copy(), g.copy()
    while not r.is_zero():
        q, r = r, q % r
    return q.monic()


def SFF(f):
    p = f.field.characteristic()
    i, res, g = 1, [], f.derivative()
    if not g.is_zero():
        c = poly_gcd(f, g)
        w = f // c
        while not w.is_one():
            y = poly_gcd(w, c)
            z = w // y
            if not z.is_one():
                res.append((z, i))
            i, w, c = i + 1, y, c // y
        if not c.is_one():
            res += [(poly, i * p) for poly, i in SFF(c.inv_frob())]
        return res
    else:
        return [(poly, i * p) for poly, i in SFF(f.inv_frob())]


def DDF(f):
    i, S, ff, q = 1, [], f.copy(), f.field.field_size()
    while ff.degree() >= 2 * i:
        big_poly = GFPoly(fromint(f.field, [0, -1] + [0] * (q**i - 2) + [1]))
        g = poly_gcd(ff, big_poly)
        if not g.is_one():
            S.append((g, i))
            ff //= g
        i += 1
    if not ff.is_one():
        S.append((ff, ff.degree()))
    return S if len(S) >= 1 else [(ff, 1)]


def EDF(f, r):
    V, n, m, q = {f}, f.degree(), f.degree() // r, f.field.field_size()
    while len(V) < m:
        h = GFPoly(random(f.field, n))
        g = (h ** ((q**r) // 2) - 1) % f
        for u in [x for x in V if x.degree() > r]:
            gu = poly_gcd(g, u)
            if not gu.is_one() and gu != u:
                V.discard(u)
                V.add(gu)
                V.add(u // gu)
    return list(V)


def factor(f):
    if f.degree() == 0:
        return f.copy()
    const = f.coeff[-1]
    f = f.monic()
    ans, sff_res = [], SFF(f)
    for f_sf, r in sff_res:
        ddf_res = DDF(f_sf)
        for fd, d in ddf_res:
            ans += [(fac, r) for fac in EDF(fd, d)]
    return ans


def GFMat_test(shape=(7, 10)):
    gf = GF(p=7, d=1, poly=[4, 1])
    # gf = GF(p=2,d=4,poly=[1,1,1,1,1])
    print(gf)
    A = random(gf, shape)
    A *= (np.random.random(A.shape) > 0.7).astype(int)
    print('A')
    print(A)

    R, Y, rank = rref(A)
    print('rank A is', rank)
    print('R, Y = rref(A)')
    print('R')
    print(R)
    print('Y')
    print(Y)
    print('np.all(Y @ A == R) is', np.all(Y @ A == R))

    E, X, rank = ref(A)
    print('E, X = ref(A)')
    print('E')
    print(E)
    print('X')
    print(X)
    print('np.all(X @ A == E) is', np.all(X @ A == E))


def GFPoly_test():
    # gf = GF(p=2,d=4,poly=[1,1,1,1,1])
    # gf.change_base(gf[7])
    gf = GF(7, 3, poly=[2, 1, 0, 1])
    print(gf)
    poly1 = GFPoly(random(gf, 3))
    poly2 = GFPoly(random(gf, 4))
    print('poly1 is\n', poly1)
    print('poly2 is\n', poly2)
    print('poly1+1 is\n', poly1 + 1)
    print('poly2+gf[11] is\n', poly2 + gf[11], '\nwhere gf[11] is\n', gf[11])
    print('poly1+poly2 is\n', poly1 + poly2)
    print('poly1*poly2 is\n', poly1 * poly2)
    print('divmod(poly2, poly1) gives \n q = {0}, r = {1}'.format(*divmod(poly2, poly1)))
    print('poly2 // poly1 is\n', poly2 // poly1)
    print('poly2 % poly1 is\n', poly2 % poly1)
    print('eval poly1 at gf[11] is\n', poly1.eval(gf[11]))
    print('roots of poly1+poly2 are\n', (poly1 + poly2).roots())
    print('poly2.derivative() is\n', poly2.derivative())


def GFPoly_factor_test():
    gf = GF(3, 1, poly=[1, 1])
    f = GFPoly([gf[1]])
    factors = defaultdict(int)
    for i in range(30):
        d = np.random.randint(low=1, high=5)
        while True:
            fac = GFPoly(random(gf, d + 1))
            if fac.degree() == d:
                break
        f *= fac
        factors[fac] += 1
    res = factor(f)
    print('f.monic()')
    print(f.monic())
    print('factors')
    print(res)
    result = reduce(lambda x, y: x * y[0]**y[1], res, GFPoly([gf[1]]))
    print('result')
    print(result)
    print('match', result == f.monic())


def list_irreducible_polynomials(K=1000):
    def is_prime(x):
        return all(x % i for i in range(2, x))
    res = {}
    for p in range(2, int(K**0.5) + 1):
        if not is_prime(p):
            continue
        res[p], D = defaultdict(list), 1 + int(np.log(K) / np.log(p))
        V, seen = de2vec(np.arange(p**D), p), set()
        print(p, D)
        gf = GF(p, 1, poly=[1, 1])
        for i, v in enumerate(V):
            f = GFPoly(fromint(gf, v))
            pattern = tuple((f.coeff.astype(int) != 0) + 0)
            if pattern in seen:
                continue
            if f.degree() == 0 or f.monic() != f:
                continue
            sff_res = SFF(f)
            if len(sff_res) > 1 or sff_res[0][1] > 1:
                continue
            ddf_res = DDF(f)
            if len(ddf_res) > 1:
                continue
            edf_res = EDF(f, ddf_res[0][1])
            if len(edf_res) > 1:
                continue
            print(f)
            seen.add(pattern)
            res[p][f.degree()].append(f)
    return res
