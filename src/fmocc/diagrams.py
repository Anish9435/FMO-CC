"""
Algebraic expressions for the CC theory

This module defines the DiagramBuilder class, responsible for evaluating the
diagrammatic contributions to the CC amplitude equations within the FMO framework.
Each function corresponds to a distinct subset of Goldstone diagrams in the
CCSD or iCCSDn equations.

Key Responsibilities
--------------------
- Assemble Fock-space intermediates and diagrammatic contributions.
- Construct residual tensors R₁ (R_ia) and R₂ (R_ijab) and additional residual tensors for iCCSD/iCCSD-PT.
- Extend support for iCCSD and perturbative corrections.

Dependencies
-------------
- Python standard libraries: copy
- External library: numpy
- Local module: utils (FMOCC_LOGGER)
"""
import copy as cp
import numpy as np
from .utils import FMOCC_LOGGER

class DiagramBuilder:
    """
    Constructs coupled-cluster diagram contributions for FMO-based CC computations.

    Each method corresponds to a specific subset of CCSD Goldstone diagrams,
    implemented as tensor contractions over Fock and two-electron integrals.
    The computed intermediates feed into amplitude update equations in
    `FMOProcessor`.

    Methods
    -------
    update1(...)
        Core CCSD diagrams contributing to R_ia and R_ijab.
    update2(...)
        Secondary singles and doubles corrections.
    update3(...)
        Higher-order diagrams involving four-occupied intermediates.
    update4–update10(...)
        Additional diagram subsets (nonlinear and exchange terms).
    So_diagrams(...), Sv_diagrams(...)
        Fragment-level (occupied/virtual) diagram contributions for iCCSD.
    """
    def __init__(self):
        self.logger = FMOCC_LOGGER

    def update1(self,occ,nao,t1,t2,tau,Fock_mo,twoelecint_mo):
        I_vv = cp.deepcopy(Fock_mo[occ:nao,occ:nao])
        I_vv += -2*np.einsum('cdkl,klad->ca',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],t2) + np.einsum('cdkl,klda->ca',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],t2)
        I_oo = cp.deepcopy(Fock_mo[:occ,:occ])
        I_oo += 2*np.einsum('cdkl,ilcd->ik',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],tau) - np.einsum('dckl,lidc->ik',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],tau)
 
        R_ia1 = cp.deepcopy(Fock_mo[:occ,occ:nao]) 
        R_ia1 += -np.einsum('ik,ka->ia',I_oo,t1)
        R_ia1 += np.einsum('ca,ic->ia',I_vv,t1)
  
        I_vv += 2*np.einsum('bcja,jb->ca',twoelecint_mo[occ:nao,occ:nao,:occ,occ:nao],t1)
        I_vv += -np.einsum('cbja,jb->ca',twoelecint_mo[occ:nao,occ:nao,:occ,occ:nao],t1)
        I_vv += -2*np.einsum('dclk,ld,ka->ca',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],t1,t1)
        I_oo += 2*np.einsum('ibkj,jb->ik',twoelecint_mo[:occ,occ:nao,:occ,:occ],t1)
        I_oo += -np.einsum('ibjk,jb->ik',twoelecint_mo[:occ,occ:nao,:occ,:occ],t1)

        R_ijab1 = 0.5*cp.deepcopy(twoelecint_mo[:occ,:occ,occ:nao,occ:nao])
        R_ijab1 += -np.einsum('ik,kjab->ijab',I_oo,t2)
        R_ijab1 += np.einsum('ca,ijcb->ijab',I_vv,t2)

        return R_ia1,R_ijab1

    def update2(self,occ,nao,t1,tau,twoelecint_mo):
        R_ia2 = -2*np.einsum('ibkj,kjab->ia',twoelecint_mo[:occ,occ:nao,:occ,:occ],tau)
        R_ia2 += np.einsum('ibkj,jkab->ia',twoelecint_mo[:occ,occ:nao,:occ,:occ],tau)
        R_ia2 += 2*np.einsum('cdak,ikcd->ia',twoelecint_mo[occ:nao,occ:nao,occ:nao,:occ],tau)
        R_ia2 += -np.einsum('cdak,ikdc->ia',twoelecint_mo[occ:nao,occ:nao,occ:nao,:occ],tau)
        R_ia2 += 2*np.einsum('icak,kc->ia',twoelecint_mo[:occ,occ:nao,occ:nao,:occ],t1)
        R_ia2 += -np.einsum('icka,kc->ia',twoelecint_mo[:occ,occ:nao,:occ,occ:nao],t1)
  
        R_ijab2 = -np.einsum('ickb,ka,jc->ijab',twoelecint_mo[:occ,occ:nao,:occ,occ:nao],t1,t1)
        R_ijab2 += -np.einsum('icak,jc,kb->ijab',twoelecint_mo[:occ,occ:nao,occ:nao,:occ],t1,t1)
        R_ijab2 += -np.einsum('ijkb,ka->ijab',twoelecint_mo[:occ,:occ,:occ,occ:nao],t1)
        R_ijab2 += np.einsum('cjab,ic->ijab',twoelecint_mo[occ:nao,:occ,occ:nao,occ:nao],t1)

        return R_ia2,R_ijab2

    def update3(self,occ,nao,tau,t1,t2,twoelecint_mo):
        Ioooo = cp.deepcopy(twoelecint_mo[:occ,:occ,:occ,:occ])
        Ioooo += np.einsum('cdkl,ijcd->ijkl',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],t2)
        Ioooo_2 = 0.5*np.einsum('cdkl,ic,jd->ijkl',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],t1,t1)
  
        R_ijab3 = 0.5*np.einsum('ijkl,klab->ijab',Ioooo,tau)
        R_ijab3 += np.einsum('ijkl,klab->ijab',Ioooo_2,t2)

        return R_ijab3

    def update4(self,occ,nao,t1,t2,twoelecint_mo):
        Iovov = cp.deepcopy(twoelecint_mo[:occ,occ:nao,:occ,occ:nao])
        Iovov += -0.5*np.einsum('dckl,ildb->ickb',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],t2)
        Iovov_2 = cp.deepcopy(twoelecint_mo[:occ,occ:nao,:occ,occ:nao])
        Iovov_3 = -np.einsum('dckl,ildb->ickb',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],t2)
  
        R_ijab4 = - np.einsum('ickb,kjac->ijab',Iovov,t2)    
        R_ijab4 += -np.einsum('icka,kjcb->ijab',Iovov_2,t2)   
        R_ijab4 += -np.einsum('ickb,jc,ka->ijab',Iovov_3,t1,t1)

        return R_ijab4


    def update5(self,occ,virt,nao,t1,t2,twoelecint_mo):
        I_oovo = np.zeros((occ,occ,virt,occ))
        I_oovo += -np.einsum('cikl,jlca->ijak',twoelecint_mo[occ:nao,:occ,:occ,:occ],t2)
        I_oovo += np.einsum('cdka,jicd->ijak',twoelecint_mo[occ:nao,occ:nao,:occ,occ:nao],t2)
        I_oovo += -np.einsum('jclk,lica->ijak',twoelecint_mo[:occ,occ:nao,:occ,:occ],t2)
        I_oovo += 2*np.einsum('jckl,ilac->ijak',twoelecint_mo[:occ,occ:nao,:occ,:occ],t2)
        I_oovo += -np.einsum('jckl,ilca->ijak',twoelecint_mo[:occ,occ:nao,:occ,:occ],t2)
   
        R_ijab5 = -np.einsum('ijak,kb->ijab',I_oovo,t1)

        return R_ijab5

    def update6(self,occ,virt,nao,t1,t2,twoelecint_mo):
        I_vovv = np.zeros((virt,occ,virt,virt))
        I_vovv += np.einsum('cjkl,klab->cjab',twoelecint_mo[occ:nao,:occ,:occ,:occ],t2)
        I_vovv += -np.einsum('cdlb,ljad->cjab',twoelecint_mo[occ:nao,occ:nao,:occ,occ:nao],t2)
        I_vovv += -np.einsum('cdka,kjdb->cjab',twoelecint_mo[occ:nao,occ:nao,:occ,occ:nao],t2)
        I_vovv += 2*np.einsum('cdal,ljdb->cjab',twoelecint_mo[occ:nao,occ:nao,occ:nao,:occ],t2)
        I_vovv += -np.einsum('cdal,jldb->cjab',twoelecint_mo[occ:nao,occ:nao,occ:nao,:occ],t2)

        R_ijab6 = np.einsum('cjab,ic->ijab',I_vovv,t1)

        return R_ijab6

    def update7(self,occ,nao,t1,t2,twoelecint_mo):
        Iovvo = cp.deepcopy(twoelecint_mo[:occ,occ:nao,occ:nao,:occ])
        Iovvo += np.einsum('dclk,jlbd->jcbk',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],t2) - np.einsum('dclk,jldb->jcbk',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],t2) - 0.5*np.einsum('cdlk,jlbd->jcbk',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],t2)

        Iovvo_2 = cp.deepcopy(twoelecint_mo[:occ,occ:nao,occ:nao,:occ])
        Iovvo_2 += -0.5*np.einsum('dclk,jldb->jcbk',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],t2)  - np.einsum('dckl,ljdb->jcbk',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],t2)
  
        Iovvo_3 = 2*np.einsum('dclk,jlbd->jcbk',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],t2) - np.einsum('dclk,jldb->jcbk',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],t2) + np.einsum('cdak,ic->idak',twoelecint_mo[occ:nao,occ:nao,occ:nao,:occ],t1)
        Iovvo_3 += -np.einsum('iclk,la->icak',twoelecint_mo[:occ,occ:nao,:occ,:occ],t1)
  
        R_ijab7 = 2*np.einsum('jcbk,kica->ijab',Iovvo,t2) 
        R_ijab7 += - np.einsum('jcbk,ikca->ijab',Iovvo_2,t2)   
        R_ijab7 += -np.einsum('jcbk,ic,ka->ijab',Iovvo_3,t1,t1)

        return R_ijab7  

    def update8(self,occ,nao,t1,t2,twoelecint_mo):
        I_voov = -np.einsum('cdkl,kjdb->cjlb',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],t2)
        R_ijab8 = -np.einsum('cjlb,ic,la->ijab',I_voov,t1,t1)

        return R_ijab8

    def update9(self,occ,nao,tau,twoelecint_mo):
        Ivvvv = cp.deepcopy(twoelecint_mo[occ:nao,occ:nao,occ:nao,occ:nao])
        R_ijab9 = 0.5*np.einsum('cdab,ijcd->ijab',Ivvvv,tau)

        return R_ijab9

    def update10(self,occ,nao,t1,t2,twoelecint_mo):
        I1 = 2*np.einsum('cbkj,kc->bj',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],t1)
        I2 = -np.einsum('cbjk,kc->bj',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],t1)
        I3 = -np.einsum('cdkl,ic,ka->idal',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],t1,t1)
        Iooov = np.einsum('dl,ijdb->ijlb',I2,t2)
 
        R_ia10 = 2*np.einsum('bj,ijab->ia',I1,t2) - np.einsum('bj,ijba->ia',I1,t2)
        R_ia10 += 2*np.einsum('bj,ijab->ia',I2,t2) - np.einsum('bj,ijba->ia',I2,t2)

        R_ijab10 = -np.einsum('ijlb,la->ijab',Iooov,t1)
        R_ijab10 += -0.5*np.einsum('idal,jd,lb->ijab',I3,t1,t1)

        return R_ia10,R_ijab10

    def update11(self,So,Sv,t2,occ,v_act,o_act,nao,twoelecint_mo):
        I4_v = 2*np.einsum('cdkj,kbcd->bj',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],Sv)
        I4_o = -2*np.einsum('cbkl,klcj->bj',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],So)
        I5_v = -np.einsum('cdkj,kbdc->bj',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],Sv)
        I5_o = np.einsum('dblk,kldj->bj',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],So)
  
        R_ia11 = 2*np.einsum('bj,jiba->ia',I4_o,t2[occ-o_act:occ,:,:,:],optimize=True)
        R_ia11 += -np.einsum('bj,jiab->ia',I4_o,t2[occ-o_act:occ,:,:,:],optimize=True)
        R_ia11 += 2*np.einsum('bj,jiba->ia',I4_v,t2[:,:,:v_act,:],optimize=True)
        R_ia11 += -np.einsum('bj,ijba->ia',I4_v,t2[:,:,:v_act,:],optimize=True)
        R_ia11 += 2*np.einsum('bj,jiba->ia',I5_o,t2[occ-o_act:occ,:,:,:],optimize=True)
        R_ia11 += -np.einsum('bj,jiab->ia',I5_o,t2[occ-o_act:occ,:,:,:],optimize=True)
        R_ia11 += 2*np.einsum('bj,jiba->ia',I5_v,t2[:,:,:v_act,:],optimize=True)
        R_ia11 += np.einsum('bj,ijba->ia',I5_v,t2[:,:,:v_act,:],optimize=True)

        return R_ia11

    def So_int_diagrams(self,occ,o_act,nao,So,t2,twoelecint_mo):
        II_oo = np.zeros((occ,occ)) 
        II_oo[:,occ-o_act:occ] += -2*0.25*np.einsum('ciml,mlcv->iv',twoelecint_mo[occ:nao,:occ,:occ,:occ],So) + 0.25*np.einsum('diml,lmdv->iv',twoelecint_mo[occ:nao,:occ,:occ,:occ],So)
  
        R_ijab = -np.einsum('ik,kjab->ijab',II_oo,t2)   
        return R_ijab,II_oo

    def Sv_int_diagrams(self,occ,virt,v_act,nao,Sv,t2,twoelecint_mo):
        II_vv = np.zeros((virt,virt))
        II_vv[:v_act,:] += 2*0.25*np.einsum('dema,mude->ua',twoelecint_mo[occ:nao,occ:nao,:occ,occ:nao],Sv) - 0.25*np.einsum('dema,mued->ua',twoelecint_mo[occ:nao,occ:nao,:occ,occ:nao],Sv)
  
        R_ijab = np.einsum('ca,ijcb->ijab',II_vv,t2)   
        return R_ijab,II_vv
    
    def Sv_diagrams(self,occ,v_act,nao,Sv,t2,Fock_mo,twoelecint_mo):
        R_iuab = cp.deepcopy(twoelecint_mo[:occ,occ:occ+v_act,occ:nao,occ:nao])
        R_iuab += -np.einsum('ik,kuab->iuab',Fock_mo[:occ,:occ],Sv)
        R_iuab += np.einsum('da,iudb->iuab',Fock_mo[occ:nao,occ:nao],Sv)
        R_iuab += np.einsum('db,iuad->iuab',Fock_mo[occ:nao,occ:nao],Sv)
        R_iuab += np.einsum('edab,iued->iuab',twoelecint_mo[occ:nao,occ:nao,occ:nao,occ:nao],Sv)
        R_iuab += 2*np.einsum('dukb,kida->iuab',twoelecint_mo[occ:nao,occ:occ+v_act,:occ,occ:nao],t2)
        R_iuab += 2*np.einsum('idak,kudb->iuab',twoelecint_mo[:occ,occ:nao,occ:nao,:occ],Sv)
        R_iuab += -np.einsum('idka,kudb->iuab',twoelecint_mo[:occ,occ:nao,:occ,occ:nao],Sv)
        R_iuab += -np.einsum('udkb,kida->iuab',twoelecint_mo[occ:occ+v_act,occ:nao,:occ,occ:nao],t2)
        R_iuab += -np.einsum('dukb,kiad->iuab',twoelecint_mo[occ:nao,occ:occ+v_act,:occ,occ:nao],t2)
        R_iuab += -np.einsum('dika,kubd->iuab',twoelecint_mo[occ:nao,:occ,:occ,occ:nao],Sv)
        R_iuab += np.einsum('uikl,klba->iuab',twoelecint_mo[occ:occ+v_act,:occ,:occ,:occ],t2)
        R_iuab += -np.einsum('udka,kibd->iuab',twoelecint_mo[occ:occ+v_act,occ:nao,:occ,occ:nao],t2)
        R_iuab += -np.einsum('idkb,kuad->iuab',twoelecint_mo[:occ,occ:nao,:occ,occ:nao],Sv)

        return R_iuab

    def So_diagrams(self,occ,o_act,nao,So,t2,Fock_mo,twoelecint_mo):
        R_ijav = cp.deepcopy(twoelecint_mo[:occ,:occ,occ:nao,occ-o_act:occ])
        R_ijav += np.einsum('da,ijdv->ijav',Fock_mo[occ:nao,occ:nao],So)
        R_ijav += -np.einsum('jl,ilav->ijav',Fock_mo[:occ,:occ],So)
        R_ijav += -np.einsum('il,ljav->ijav',Fock_mo[:occ,:occ],So)
        R_ijav += 2*np.einsum('djlv,lida->ijav',twoelecint_mo[occ:nao,:occ,:occ,occ-o_act:occ],t2)
        R_ijav += 2*np.einsum('dila,ljdv->ijav',twoelecint_mo[occ:nao,:occ,:occ,occ:nao],So)
        R_ijav += -np.einsum('djlv,liad->ijav',twoelecint_mo[occ:nao,:occ,:occ,occ-o_act:occ],t2)
        R_ijav += -np.einsum('dila,jldv->ijav',twoelecint_mo[occ:nao,:occ,:occ,occ:nao],So)
        R_ijav += -np.einsum('dial,ljdv->ijav',twoelecint_mo[occ:nao,:occ,occ:nao,:occ],So)
        R_ijav += -np.einsum('djvl,lida->ijav',twoelecint_mo[occ:nao,:occ,occ-o_act:occ,:occ],t2)
        R_ijav += np.einsum('ijlm,lmav->ijav',twoelecint_mo[:occ,:occ,:occ,:occ],So)
        R_ijav += np.einsum('cdva,jicd->ijav',twoelecint_mo[occ:nao,occ:nao,occ-o_act:occ,occ:nao],t2)
        R_ijav += -np.einsum('jdla,ildv->ijav',twoelecint_mo[:occ,occ:nao,:occ,occ:nao],So)
        R_ijav += -np.einsum('idlv,ljad->ijav',twoelecint_mo[:occ,occ:nao,:occ,occ-o_act:occ],t2)

        return R_ijav
    
    def T1_contribution_Sv(self,occ,nao,v_act,t1,twoelecint_mo):
        R_iuab = -np.einsum('uika,kb->iuab',twoelecint_mo[occ:occ+v_act,:occ,:occ,occ:nao],t1)
        R_iuab += np.einsum('duab,id->iuab',twoelecint_mo[occ:nao,occ:occ+v_act,occ:nao,occ:nao],t1)
        R_iuab += -np.einsum('iukb,ka->iuab',twoelecint_mo[:occ,occ:occ+v_act,:occ,occ:nao],t1)
        return R_iuab

    def T1_contribution_So(occ,nao,o_act,t1,twoelecint_mo):
        R_ijav = np.einsum('diva,jd->ijav',twoelecint_mo[occ:nao,:occ,occ-o_act:occ,occ:nao],t1)
        R_ijav += np.einsum('djav,id->ijav',twoelecint_mo[occ:nao,:occ,occ:nao,occ-o_act:occ],t1)
        R_ijav += -np.einsum('ijkv,ka->ijav',twoelecint_mo[:occ,:occ,:occ,occ-o_act:occ],t1)
        return R_ijav