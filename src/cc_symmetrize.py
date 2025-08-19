import numpy as np
import copy as cp
import gc

class Symmetrizer:
    def symmetrize(self, occ, virt, R_ijab):
        R_ijab_new = np.zeros((occ, occ, virt, virt))
        for i in range(occ):
            for j in range(occ):
                for a in range(virt):
                    for b in range(virt):
                        R_ijab_new[i, j, a, b] = R_ijab[i, j, a, b] + R_ijab[j, i, b, a]
        R_ijab = cp.deepcopy(R_ijab_new)
        del R_ijab_new
        gc.collect()
        return R_ijab