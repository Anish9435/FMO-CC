from multiprocessing import Pool
import numpy as np
import copy as cp
import time

from sqlalchemy import Tuple
from diagrams import DiagramBuilder
from .utils import Symmetrizer, AmplitudeUpdater, get_logger

class CCParallel:
    def __init__(self):
        self.logger = get_logger(__name__)
        self.diagram_builder = DiagramBuilder()
        self.amplitude_updater = AmplitudeUpdater()
        self.symmetrizer = Symmetrizer()

    def energy_ccd(self, occ, nao, t2, twoelecint_mo):
        if t2.shape != (occ, occ, nao - occ, nao - occ):
            self.logger.error(f"Invalid t2 shape: {t2.shape}")
            raise ValueError(f"Invalid t2 shape")
        E_ccd = 2 * np.einsum('ijab,ijab', t2, twoelecint_mo[:occ, :occ, occ:nao, occ:nao]) - np.einsum('ijab,ijba', t2, twoelecint_mo[:occ, :occ, occ:nao, occ:nao])
        self.logger.info(f"Computed CCD energy: {E_ccd}")
        return E_ccd

    def energy_ccsd(self, occ, nao, t1, t2, twoelecint_mo):
        if t1.shape != (occ, nao - occ) or t2.shape != (occ, occ, nao - occ, nao - occ):
            self.logger.error(f"Invalid shapes: t1={t1.shape}, t2={t2.shape}")
            raise ValueError(f"Invalid input shapes")
        E_ccd = self.energy_ccd(occ, nao, t2, twoelecint_mo)
        E_ccd += 2 * np.einsum('ijab,ia,jb', twoelecint_mo[:occ, :occ, occ:nao, occ:nao], t1, t1) - np.einsum('ijab,ib,ja', twoelecint_mo[:occ, :occ, occ:nao, occ:nao], t1, t1)
        self.logger.info(f"Computed CCSD energy: {E_ccd}")
        return E_ccd

    def convergence_I(self, E_ccd, E_old, eps_t, eps_So, eps_Sv, conv, x):
        del_E = E_ccd - E_old
        if all(abs(v) <= conv for v in [eps_t, eps_So, eps_Sv, del_E]):
            self.logger.info(f"ICCSD converged at cycle {x+1}, correlation energy: {E_ccd}")
            return True, E_ccd
        self.logger.info(f"Cycle {x+1}: t1+t2={eps_t}, So={eps_So}, Sv={eps_Sv}, ΔE={del_E}, E_corr={E_ccd}")
        return False, E_ccd

    def convergence(self, E_ccd, E_old, eps, conv, x):
        del_E = E_ccd - E_old
        if abs(eps) <= conv and abs(del_E) <= conv:
            self.logger.info(f"CCSD converged at cycle {x+1}, correlation energy: {E_ccd}")
            return True, E_ccd
        self.logger.info(f"Cycle {x+1}: t1+t2={eps}, ΔE={del_E}, E_corr={E_ccd}")
        return False, E_ccd

    def cc_calc(self, occ, virt, o_act, v_act, nao, t1, t2, So, Sv, D1, D2, Do, Dv, twoelecint_mo, Fock_mo, calc, n_iter, E_old, conv):
        for x in range(n_iter):
            if calc == 'CCSD':
                pool = Pool(12)
                tau = cp.deepcopy(t2) + np.einsum('ia,jb->ijab', t1, t1)
                result_comb_temp1 = pool.apply_async(self.diagram_builder.update1, args=(occ, nao, t1, t2, tau, Fock_mo, twoelecint_mo))
                result_comb_temp2 = pool.apply_async(self.diagram_builder.update2, args=(occ, nao, t1, tau, twoelecint_mo))
                result_comb_temp3 = pool.apply_async(self.diagram_builder.update11, args=(So, Sv, t2, occ, virt, v_act, nao, twoelecint_mo))
                R_ijab3_temp = pool.apply_async(self.diagram_builder.update3,args=(occ,nao,tau,t1,t2,twoelecint_mo,))
                R_ijab4_temp = pool.apply_async(self.diagram_builder.update4,args=(occ,nao,t1,t2,twoelecint_mo,))
                R_ijab5_temp = pool.apply_async(self.diagram_builder.update5,args=(occ,virt,nao,t1,t2,twoelecint_mo,))
                R_ijab6_temp = pool.apply_async(self.diagram_builder.update6,args=(occ,virt,nao,t1,t2,twoelecint_mo,))
                R_ijab7_temp = pool.apply_async(self.diagram_builder.update7,args=(occ,nao,t1,t2,twoelecint_mo,))
                R_ijab8_temp = pool.apply_async(self.diagram_builder.update8,args=(occ,nao,t1,t2,twoelecint_mo,))
                R_ijab9_temp = pool.apply_async(self.diagram_builder.update9,args=(occ,nao,tau,twoelecint_mo,))
                pool.close()
                pool.join()

                R_ia1, R_ijab1 = result_comb_temp1.get()
                R_ia2, R_ijab2 = result_comb_temp2.get()
                R_ia10, R_ijab10 = result_comb_temp3.get()
                R_ijab3 = R_ijab3_temp.get()
                R_ijab4 = R_ijab4_temp.get()
                R_ijab5 = R_ijab5_temp.get()
                R_ijab6 = R_ijab6_temp.get()
                R_ijab7 = R_ijab7_temp.get()
                R_ijab8 = R_ijab8_temp.get()
                R_ijab9 = R_ijab9_temp.get()

                R_ia = (R_ia1+R_ia2+R_ia10)
                R_ijab = (R_ijab1+R_ijab2+R_ijab3+R_ijab4+R_ijab5+R_ijab6+R_ijab7+R_ijab8+R_ijab9+R_ijab10)

                R_ijab = self.symmetrizer.symmetrize(occ,virt,R_ijab)
                eps_t, t1, t2 = self.amplitude_updater.update_t1t2(R_ia,R_ijab,t1,t2,D1,D2)

                E_ccd = self.energy_ccsd(occ, nao, t1, t2, twoelecint_mo)
                val, E_ccd = self.convergence(E_ccd, E_old, eps_t, conv, x)
                if val:
                    break
                E_old = E_ccd

            if calc == 'ICCSD':
                pool=Pool(12)
                tau = cp.deepcopy(t2)
                tau += np.einsum('ia,jb->ijab',t1,t1)
            
                #II_oo = self.diagram_builder.So_int_diagrams(occ,o_act,nao,So,t2,twoelecint_mo)[1]
                #II_vv = self.diagram_builder.Sv_int_diagrams(occ,virt,v_act,nao,Sv,t2,twoelecint_mo)[1]
            
                result_comb_temp1 = pool.apply_async(self.diagram_builder.update1,args=(occ,nao,t1,t2,tau,Fock_mo,twoelecint_mo,))
                result_comb_temp2 = pool.apply_async(self.diagram_builder.update2,args=(occ,nao,t1,tau,twoelecint_mo,))
                result_comb_temp3 = pool.apply_async(self.diagram_builder.update10,args=(occ,nao,t1,t2,twoelecint_mo,))
                R_ijab3_temp = pool.apply_async(self.diagram_builder.update3,args=(occ,nao,tau,t1,t2,twoelecint_mo,))
                R_ijab4_temp = pool.apply_async(self.diagram_builder.update4,args=(occ,nao,t1,t2,twoelecint_mo,))
                R_ijab5_temp = pool.apply_async(self.diagram_builder.update5,args=(occ,virt,nao,t1,t2,twoelecint_mo,))
                R_ijab6_temp = pool.apply_async(self.diagram_builder.update6,args=(occ,virt,nao,t1,t2,twoelecint_mo,))
                R_ijab7_temp = pool.apply_async(self.diagram_builder.update7,args=(occ,nao,t1,t2,twoelecint_mo,))
                R_ijab8_temp = pool.apply_async(self.diagram_builder.update8,args=(occ,nao,t1,t2,twoelecint_mo,))
                R_ijab9_temp = pool.apply_async(self.diagram_builder.update9,args=(occ,nao,tau,twoelecint_mo,))
                #R_ia11_temp = pool.apply_async(diagrams.update11,args=(So,Sv,t2,))
                R_iuab_temp = pool.apply_async(self.diagram_builder.Sv_diagrams,args=(occ,v_act,nao,Sv,t1,t2,Fock_mo,twoelecint_mo,))
                R_ijav_temp = pool.apply_async(self.diagram_builder.So_diagrams,args=(occ,o_act,nao,So,t1,t2,Fock_mo,twoelecint_mo,))

                pool.close()
                pool.join()
            
                R_ia1, R_ijab1 = result_comb_temp1.get() 
                R_ia2, R_ijab2 = result_comb_temp2.get() 
                R_ia10, R_ijab10 = result_comb_temp3.get() 
                R_ijab3 = R_ijab3_temp.get()
                R_ijab4 = R_ijab4_temp.get()
                R_ijab5 = R_ijab5_temp.get()
                R_ijab6 = R_ijab6_temp.get()
                R_ijab7 = R_ijab7_temp.get()
                R_ijab8 = R_ijab8_temp.get()
                R_ijab9 = R_ijab9_temp.get()
                #R_ia11 = R_ia11_temp.get()
            
                R_ia = (R_ia1+R_ia2+R_ia10)
                R_ijab = (R_ijab1+R_ijab2+R_ijab3+R_ijab4+R_ijab5+R_ijab6+R_ijab7+R_ijab8+R_ijab9+R_ijab10)
                R_ijab += self.diagram_builder.So_int_diagrams(occ,o_act,nao,So,t2,twoelecint_mo)[0]
                R_ijab += self.diagram_builder.Sv_int_diagrams(occ,virt,v_act,nao,Sv,t2,twoelecint_mo)[0]
                R_ijab = self.symmetrizer.symmetrize(occ,virt,R_ijab)
            
                R_iuab = R_iuab_temp.get()
                #R_iuab += diagrams.T1_contribution_Sv(occ,nao,v_act,t1,twoelecint_mo)
                #R_iuab += diagrams.coupling_terms_So(So,t2)[0]
                #R_iuab += diagrams.w2_int_2(So,Sv,t2)

                R_ijav = R_ijav_temp.get() 
                #R_ijav += diagrams.T1_contribution_So(occ,nao,o_act,t1,twoelecint_mo)
                #R_ijav += diagrams.coupling_terms_Sv(Sv,t2)[0]
                #R_ijav += diagrams.w2_int_1(So,Sv,t2)
            
            
                eps_t, t1, t2 = self.amplitude_updater.update_t1t2(R_ia,R_ijab,t1,t2,D1,D2)
                eps_So, So = self.amplitude_updater.update_So(R_ijav,So,Do,conv)
                eps_Sv, Sv = self.amplitude_updater.update_Sv(R_iuab,Sv,Dv,conv)

                E_ccd = self.energy_ccsd(occ,nao,t1,t2,twoelecint_mo)
                val, E_ccd = self.convergence_I(E_ccd,E_old,eps_t,eps_So,eps_Sv,conv,x)
                if val:
                    print(E_ccd)
                    break
                else:
                    E_old = E_ccd
                    
            if calc == 'ICCSD-PT':
                pool=Pool(12)
                tau = cp.deepcopy(t2)
                tau += np.einsum('ia,jb->ijab',t1,t1)
            
                #II_oo = self.diagram_builder.So_int_diagrams(occ,o_act,nao,So,t2,twoelecint_mo)[1]
                #II_vv = self.diagram_builder.Sv_int_diagrams(occ,virt,v_act,nao,Sv,t2,twoelecint_mo)[1]
            
                result_comb_temp1 = pool.apply_async(self.diagram_builder.update1,args=(t1,t2,tau,))
                result_comb_temp2 = pool.apply_async(self.diagram_builder.update2,args=(t1,tau,))
                result_comb_temp3 = pool.apply_async(self.diagram_builder.update10,args=(t1,t2,))
                R_ijab3_temp = pool.apply_async(self.diagram_builder.update3,args=(tau,t1,t2,))
                R_ijab4_temp = pool.apply_async(self.diagram_builder.update4,args=(t1,t2,))
                R_ijab5_temp = pool.apply_async(self.diagram_builder.update5,args=(t1,t2,))
                R_ijab6_temp = pool.apply_async(self.diagram_builder.update6,args=(t1,t2,))
                R_ijab7_temp = pool.apply_async(self.diagram_builder.update7,args=(t1,t2,))
                R_ijab8_temp = pool.apply_async(self.diagram_builder.update8,args=(t1,t2,))
                R_ijab9_temp = pool.apply_async(self.diagram_builder.update9,args=(tau,))
                R_ia11_temp = pool.apply_async(self.diagram_builder.update11,args=(So,Sv,t2,))
            
                pool.close()
                pool.join()
            
                R_ia1, R_ijab1 = result_comb_temp1.get() 
                R_ia2, R_ijab2 = result_comb_temp2.get() 
                R_ia10, R_ijab10 = result_comb_temp3.get() 
                R_ijab3 = R_ijab3_temp.get()
                R_ijab4 = R_ijab4_temp.get()
                R_ijab5 = R_ijab5_temp.get()
                R_ijab6 = R_ijab6_temp.get()
                R_ijab7 = R_ijab7_temp.get()
                R_ijab8 = R_ijab8_temp.get()
                R_ijab9 = R_ijab9_temp.get()
                R_ia11 = R_ia11_temp.get()
            
                R_ia = (R_ia1+R_ia2+R_ia10+R_ia11)
                R_ijab = (R_ijab1+R_ijab2+R_ijab3+R_ijab4+R_ijab5+R_ijab6+R_ijab7+R_ijab8+R_ijab9+R_ijab10)
                R_ijab += self.diagram_builder.So_int_diagrams(So,t2)[0]
                R_ijab += self.diagram_builder.Sv_int_diagrams(Sv,t2)[0]
                R_ijab = self.symmetrizer.symmetrize(R_ijab)
            
                eps_t, t1, t2 = self.amplitude_updater.update_t1t2(R_ia,R_ijab,t1,t2)
            
                E_ccd = self.energy_ccsd(t1,t2)
                val = self.convergence(E_ccd,E_old,eps_t)
                if val:
                    break
                else:
                    E_old = E_ccd
                    
        self.logger.info(f"CC Calculation completed after {x+1} iterations")
        return E_ccd, x