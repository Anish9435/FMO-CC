import numpy as np
import subprocess
import copy as cp
import glob
import os
from .utils import get_logger

class FMOExtractor:
    def __init__(self, gamess_out, gamess_2eint, nmer, outfile1, outfile2, outfile3, tempfile, tempfile2):
        self.logger = get_logger(__name__)
        if not os.path.exists(gamess_out):
            raise FileNotFoundError(f"GAMESS output file {gamess_out} not found")
        if not os.path.exists(gamess_2eint):
            raise FileNotFoundError(f"GAMESS 2e integral file {gamess_2eint} not found")
        self.outfile1 = outfile1
        self.outfile2 = outfile2
        self.outfile3 = outfile3
        self.gamess_out = gamess_out
        self.gamess_2eint = gamess_2eint
        self.tempfile = tempfile
        self.tempfile2 = tempfile2
        self.nmer = nmer
        self.dimer_key = ['iFrag', 'jFrag']
        self.mono_key = ['iFrag', 'Iter']
        self.string_list = ['II,JST,KST,LST', 'SCHWARZ INEQUALITY']

    def get_enuc(self):
        with open(self.gamess_out, 'r') as outfile:
            for line in outfile:
                if "Nuclear repulsion energy:" in line.strip():
                    return float(line.split()[-1])
        self.logger.error("Nuclear repulsion energy not found in GAMESS output")
        return None
    
    def get_nfrags(self):
        try:
            with open(self.gamess_out, 'r') as outfile:
                for line in outfile:
                    if "Number of fragments:" in line.strip():
                        return int(line.split()[-1])
            self.logger.error("Number of fragments not found in GAMESS output")
        except Exception as e:
            raise RuntimeError(f"Error parsing nfrag from {self.gamess_out}: {e}")
        return 0

    def get_nbasis(self):
        try:
            with open(self.gamess_2eint, 'r') as outfile:
                for line in outfile:
                    if "Total number of basis functions:" in line.strip():
                        return int(line.split()[-1])
            self.logger.error("Total number of basis functions not found in GAMESS 2e integral file")
        except Exception as e:
            raise RuntimeError(f"Error parsing nbasis from {self.gamess_2eint}: {e}")
        return 0

    def get_tot_rhf(self):
        try:
            with open(self.gamess_out, 'r') as outfile:
                for line in outfile:
                    if "Total energy of the molecule: Euncorr(2)=" in line.strip():
                        return float(line.split()[-1])
            self.logger.error("Total RHF energy not found in GAMESS output")
        except Exception as e:
            raise RuntimeError(f"Error parsing RHF energy from {self.gamess_out}: {e}")
        return 0.0
    
    def get_frag_naos_atoms(self, lnum):
        with open(self.gamess_out, 'r') as outfile:
            outlines = outfile.readlines()
        nao1 = []
        natoms = []
        ini_idx = 0
        tot = lnum
        for i,line in enumerate(reversed(outlines[:lnum])):
            if line.strip() == "I  NAME     Q NAT0 NATB NA  NAO LAY MUL SCFTYP        NOP     MOL    CONV":
                self.lnum = tot-i
                ini_idx = tot-i+1
                break
        for i,line in enumerate(outlines[ini_idx:]):
            if line.strip() and "locfmo:" not in line.strip():
                val1 = int(line.split()[6])
                val2 = int(line.split()[3])
                nao1.append(val1)
                natoms.append(val2)
            else:
                break
        nao1.sort()
        natoms.sort()
        self.logger.info(f"Extracted nao1: {nao1}, natoms: {natoms}")
        return nao1, natoms
    
    def get_frag_nmos(self, lnum, nfrag):
        with open(self.gamess_out, 'r') as infile:
            inlines = infile.readlines()
        nmo_mono = []
        for i, line in enumerate(reversed(inlines[:lnum])):
            if "TOTAL NUMBER OF MOS IN VARIATION SPACE=" in line.strip():
                val = int(line.split()[-1])
                nmo_mono.append(val)
                if len(nmo_mono) == nfrag:
                    break
        self.logger.info(f"Extracted fragment MOs: {nmo_mono}")
        return nmo_mono
    
    def get_nelec(self):
        try:
            with open(self.gamess_out, 'r') as outfile:
                for line in outfile:
                    if "Total number of electrons:" in line.strip():
                        return int(line.split()[-1])
            self.logger.error("Total number of electrons not found in GAMESS output")
        except Exception as e:
            raise RuntimeError(f"Error parsing nelec from {self.gamess_out}: {e}")
        return 0
    
    def coeff(self, lnum):
        with open(self.gamess_out, 'r') as infile:
            inlines = infile.readlines()
        ini_idx = 0
        content = []
        count = 0
        total_lines = lnum
        for i, line in enumerate(reversed(inlines[:lnum])):
            if self.nmer == 2:
                if all(word in line for word in self.dimer_key):
                    self.ifrag = int(line.split()[1])
                    self.jfrag = int(line.split()[3])
                    self.Erhf = float(line.split()[5])
                    count += 1
                if count > 0 and line.strip() == "EIGENVECTORS":
                    self.lnum = total_lines - i
                    ini_idx = total_lines - i + 1
                    break
            elif self.nmer == 1:
                if all(word in line for word in self.mono_key):
                    x1, x2, x3, ifrag, x4, Erhf = line.split()
                    self.ifrag = int(ifrag)
                    self.jfrag = 0
                    self.Erhf = float(Erhf)
                    count += 1
                if count > 0 and line.strip() == "EIGENVECTORS":
                    lnum = total_lines - i
                    ini_idx = total_lines - i + 1
                    break
        for line in inlines[ini_idx:]:
            if "...... END OF RHF CALCULATION ......" not in line.strip():
                content.append(line)
            else:
                break
        with open(self.outfile1, 'w') as outf:
            outf.writelines(content)
        self.logger.info(f"Extracted coefficients to {self.outfile1}")
        return self.ifrag, self.jfrag, self.Erhf, lnum
    
    def bare_hamiltonian(self, lnum):
        with open(self.gamess_out, 'r') as infile:
            inlines = infile.readlines()
        ini_idx = 0
        content = []
        count = 0
        tot = lnum
        for i,line in enumerate(reversed(inlines[:lnum])):
            if self.nmer == 2:
                if all(word in line for word in self.dimer_key):
                    self.ifrag = int(line.split()[1])
                    self.jfrag = int(line.split()[3])
                    self.Erhf = float(line.split()[5])
                    count += 1
                if count>0:
                    if line.strip() == "BARE NUCLEUS HAMILTONIAN INTEGRALS (H=T+V)":
                        self.lnum = tot-i
                        ini_idx = tot-i+1
                        break
            if self.nmer == 1:
                if all(word in line for word in self.mono_key):
                    x1, x2, x3, ifrag, x4 , Erhf = line.split()
                    self.ifrag = int(ifrag)
                    self.jfrag = 0
                    self.Erhf = float(Erhf)
                    count += 1
                if count>0 and line.strip() == "BARE NUCLEUS HAMILTONIAN INTEGRALS (H=T+V)":
                    lnum = tot-i
                    ini_idx = tot-i+1
                    break
        for line in inlines[ini_idx:]:
            if not line.strip() == "KINETIC ENERGY INTEGRALS":
                content.append(line)
            else:
                break
        with open(self.outfile2, 'w') as outf:
            outf.writelines(content)
        self.logger.info(f"Extracted bare Hamiltonian to {self.outfile2}")
        return self.ifrag, self.jfrag, self.Erhf
    
    def twoelecint(self, lnum):                                                                                         
        with open(self.gamess_2eint, 'r') as infile:
            inlines = infile.readlines()
        ini_idx = 0
        content = []                                                                                                                          
        count = 0
        tot = lnum
        for i,line in enumerate(reversed(inlines[:lnum])):
            if self.nmer == 2:
                if all(word in line for word in self.dimer_key):
                    self.ifrag = int(line.split()[1])
                    self.jfrag = int(line.split()[3])
                    self.Erhf = float(line.split()[5])
                    count+=1
                if count>0:
                    if line.strip() == "II,JST,KST,LST =  2  1  1  1 NREC =         1 INTLOC =    2":
                        lnum = tot-i
                        ini_idx = tot-i
                        break
            if self.nmer == 1:
                if all(word in line for word in self.mono_key):
                    x1, x2, x3, ifrag, x4 , Erhf = line.split()
                    self.ifrag = int(ifrag)
                    self.jfrag = 0
                    self.Erhf = float(Erhf)
                    count+=1
                if count>0:
                    if line.strip() == "II,JST,KST,LST =  2  1  1  1 NREC =         1 INTLOC =    2":
                        lnum = tot-i
                        ini_idx = tot-i
                        break
        for i,line in enumerate(inlines[ini_idx:]):
            if "TOTAL NUMBER OF NONZERO TWO-ELECTRON INTEGRALS" not in line:
                content.append(line)
            else:
                break
        with open(self.outfile3, 'w') as outf:
            outf.writelines(content)
        self.logger.info(f"Extracted two-electron integrals to {self.outfile3}")
        return self.ifrag, self.jfrag, self.Erhf, lnum

    def get_coeff(self, nmo, nao):
        with open(self.outfile1, 'r') as infile:
            inlines = infile.readlines()
        elements = np.zeros((nmo,nao))
        m = 4
        idx1 = 0
        if nmo == nao:
            r = nmo % 5
            q = nao // 5
            for k in range(q):
                for i in range(5):
                    for j in range(m, m + nao):
                        idx2 = int(inlines[j].split()[0]) - 1
                        try:
                            val = inlines[j].split()[i + 4]
                            elements[idx1, idx2] = float(val)
                        except IndexError:
                            continue
                    idx1 += 1
                m = j + 5
            for i in range(r):
                idx1 = 5 * q + i
                for j in range(m, m + nao):
                    idx2 = int(inlines[j].split()[0]) - 1
                    try:
                        val = inlines[j].split()[i + 4]
                        elements[idx1, idx2] = float(val)
                    except IndexError:
                        continue
        else:
            r = nmo % 5
            q = nmo // 5
            for k in range(q):
                for i in range(5):
                    for j in range(m, m + nao):
                        idx2 = int(inlines[j].split()[0]) - 1
                        try:
                            val = inlines[j].split()[i + 4]
                            elements[idx1, idx2] = float(val)
                        except IndexError:
                            continue
                    idx1 += 1
                m = j + 5
            for i in range(r):
                idx1 = 5 * q + i
                for j in range(m, m + nao):
                    idx2 = int(inlines[j].split()[0]) - 1
                    try:
                        val = inlines[j].split()[i + 4]
                        elements[idx1, idx2] = float(val)
                    except IndexError:
                        continue
        elements = np.transpose(elements)
        self.logger.info(f"Extracted coefficients with shape {elements.shape} from {self.outfile1}")
        return elements
    
    def get_orb_energy(self, nao, nmo):
        with open(self.outfile1, 'r') as infile:
            inlines = infile.readlines()
        orb_energy = []
        m = 2
        if nmo==nao:
            r = nmo%5
            q = int(nao/5)
            for k in range(q):
                for i in range(5):
                    for j in range(m,m+1):
                        try:
                           val = float(inlines[j].split()[i])
                           orb_energy.append(val)
                        except IndexError:
                           continue
                m = j+nao+4
            for i in range(r):
                idx1 = 5*q+i
                for j in range(m,m+1):
                    try:
                        val = float(inlines[j].split()[i])
                        orb_energy.append(val)
                    except IndexError:
                        continue
        else:
            r = nmo%5
            q = int(nmo/5)
            for k in range(q):
                for i in range(5):
                    for j in range(m,m+1):
                        try:
                           val = float(inlines[j].split()[i])
                           orb_energy.append(val)
                        except IndexError:
                           continue
                m = j+nao+4
            for i in range(r):
                idx1 = 5*q+i
                for j in range(m,m+1):
                    try:
                        val = float(inlines[j].split()[i])
                        orb_energy.append(val)
                    except IndexError:
                        continue
        orb_energy = np.array(orb_energy)
        self.logger.info(f"Extracted orbital energies with shape {orb_energy.shape} from {self.outfile1}")
        return orb_energy

    def get_1e_parameter(self, nao):
        with open(self.outfile2, 'r') as infile:
            inlines = infile.readlines()     
        elements=np.zeros((nao,nao))
        r = nao%5
        q = int(nao/5)
        m = 2                                                                                                                               
        idx1 = 0         
        for k in range(q):                                                                                                                  
            for i in range(5):                                                                                                              
                for j in range(m,m+nao-5*k):                                                                                                
                   idx2 = int(inlines[j].split()[0])-1                                                                                      
                   try:
                       val = inlines[j].split()[i+4]  
                       elements[idx1,idx2] = float(val)
                       elements[idx2,idx1] = float(val)
                   except IndexError:
                       continue 
                idx1 = idx1+1                                                                                                               
            m = j+4                                                                                                                  
        for i in range(r):
            idx1 = 5*q+i
            for j in range(m,m+nao-5*q):
                idx2 = int(inlines[j].split()[0])-1
                try:  
                    val = inlines[j].split()[i+4]                                                                                           
                    elements[idx1,idx2] = float(val)                                                                                        
                    elements[idx2,idx1] = float(val)       
                except IndexError:                                                                                                          
                    continue
        self.logger.info(f"Extracted one-electron parameters with shape {elements.shape} from {self.outfile2}")
        return elements

    def twoelecint_process(self):
        try:
            with open(self.outfile3, 'r') as infile:
                lines = infile.readlines()
            with open(self.tempfile, 'w') as outf:
                for line in lines:
                    if not any(s in line for s in self.string_list):
                        outf.write(line)
            self.logger.info(f"Processed two-electron integrals and saved to {self.tempfile}")
            return 0
        except Exception as e:
            raise RuntimeError(f"Error processing two-electron integrals: {e}")

    def bash_run(self):
        try:
            subprocess.run(['bash', 'scripts/twoeint_process.sh'], check=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Error running twoeint_process.sh: {e}")

    def read_2eint(self, nao):
        try:
            with open(self.tempfile2, 'r') as infile:
                lines = infile.readlines()
            twoeint = np.zeros((nao, nao, nao, nao))
            length = len(lines)
            for i in range(length):
                idx1, idx2, idx3, idx4, x, val = lines[i].split()
                twoeint[int(idx1)-1,int(idx2)-1,int(idx3)-1,int(idx4)-1] = cp.deepcopy(float(val))
                twoeint[int(idx3)-1,int(idx4)-1,int(idx1)-1,int(idx2)-1] = cp.deepcopy(float(val))
                twoeint[int(idx2)-1,int(idx1)-1,int(idx4)-1,int(idx3)-1] = cp.deepcopy(float(val))
                twoeint[int(idx4)-1,int(idx3)-1,int(idx2)-1,int(idx1)-1] = cp.deepcopy(float(val))
                twoeint[int(idx2)-1,int(idx1)-1,int(idx3)-1,int(idx4)-1] = cp.deepcopy(float(val))
                twoeint[int(idx4)-1,int(idx3)-1,int(idx1)-1,int(idx2)-1] = cp.deepcopy(float(val))
                twoeint[int(idx1)-1,int(idx2)-1,int(idx4)-1,int(idx3)-1] = cp.deepcopy(float(val))
                twoeint[int(idx3)-1,int(idx4)-1,int(idx2)-1,int(idx1)-1] = cp.deepcopy(float(val))
            self.logger.info(f"Read two-electron integrals with shape {twoeint.shape} from {self.tempfile2}")
            return twoeint
        except Exception as e:
            raise RuntimeError(f"Error reading 2e integrals from {self.tempfile2}: {e}")

    def cleanup(self):
        files = glob.glob('*.txt')
        for f in files:
            if os.path.exists(f):
                try:
                    os.remove(f)
                    self.logger.info(f"Deleted temporary file {f}")
                except OSError as e:
                    self.logger.warning(f"Warning: Could not delete {f}: {e}")