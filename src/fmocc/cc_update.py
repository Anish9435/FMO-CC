import numpy as np
import copy as cp
import gc

class AmplitudeUpdater:
    def update_t2(self, R_ijab, t2, D2, conv):
        ntmax = np.size(t2)
        eps_t = float(np.sum(abs(R_ijab)) / ntmax)
        if eps_t >= conv:
            delt2 = np.divide(R_ijab, D2, out=np.zeros_like(R_ijab), where=abs(D2) > 1e-10)
            t2 += delt2
        gc.collect()
        return eps_t, t2

    def update_t1t2(self, R_ia, R_ijab, t1, t2, D1, D2):
        ntmax = np.size(t1) + np.size(t2)
        eps = float(np.sum(abs(R_ia)) + np.sum(abs(R_ijab))) / ntmax
        delt1 = np.divide(R_ia, D1, out=np.zeros_like(R_ia), where=abs(D1) > 1e-10)
        delt2 = np.divide(R_ijab, D2, out=np.zeros_like(R_ijab), where=abs(D2) > 1e-10)
        t1 += delt1
        t2 += delt2
        gc.collect()
        return eps, t1, t2

    def update_So(self, R_ijav, So, Do, conv):
        ntmax = np.size(So)
        eps_So = float(np.sum(abs(R_ijav)) / ntmax)
        if eps_So >= conv:
            delSo = np.divide(R_ijav, Do, out=np.zeros_like(R_ijav), where=abs(Do) > 1e-10)
            So += delSo
        gc.collect()
        return eps_So, So

    def update_Sv(self, R_iuab, Sv, Dv, conv):
        ntmax = np.size(Sv)
        eps_Sv = float(np.sum(abs(R_iuab)) / ntmax)
        if eps_Sv >= conv:
            delSv = np.divide(R_iuab, Dv, out=np.zeros_like(R_iuab), where=abs(Dv) > 1e-10)
            Sv += delSv
        gc.collect()
        return eps_Sv, Sv