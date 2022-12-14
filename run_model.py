# Copyright 2022 Hyun-Yong Lee

import numpy as np
import model
from tenpy.networks.mps import MPS
from tenpy.algorithms import dmrg
from tenpy.algorithms import tebd
import os
import os.path
import sys
import matplotlib.pyplot as plt
import pickle

def ensure_dir(f):
    d=os.path.dirname(f)
    if not os.path.exists(d):
        os.makedirs(d)
    return d

import logging.config
conf = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {'custom': {'format': '%(levelname)-8s: %(message)s'}},
    'handlers': {'to_file': {'class': 'logging.FileHandler',
                             'filename': 'log',
                             'formatter': 'custom',
                             'level': 'INFO',
                             'mode': 'a'},
                'to_stdout': {'class': 'logging.StreamHandler',
                              'formatter': 'custom',
                              'level': 'INFO',
                              'stream': 'ext://sys.stdout'}},
    'root': {'handlers': ['to_stdout', 'to_file'], 'level': 'DEBUG'},
}
logging.config.dictConfig(conf)

# os.environ["OMP_NUM_THREADS"] = "68"

Lx = int(sys.argv[1])
Ly = int(sys.argv[2])
t = float(sys.argv[3])
U = float(sys.argv[4])
mu = float(sys.argv[5])
CHI = int(sys.argv[6])
RM = sys.argv[7]
QN = sys.argv[8]
PATH = sys.argv[9]
BC_MPS = sys.argv[10]
IS = sys.argv[11]

if BC_MPS == 'infinite':
    BC = 'periodic'
else:
    BC = ['periodic','open']

model_params = {
    "Lx": Lx,
    "Ly": Ly,
    "t": t,
    "U": U,
    "mu": mu,
    "bc_MPS": BC_MPS,
    "bc": BC,
    "QN": QN
}

print("\n\n\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
M = model.FERMIONIC_FRACTON(model_params)

L = Lx * Ly
# initial state
if IS == 'checkerboard':
    product_state = ( ['empty', 'full'] * int(Ly/2) + ['full', 'empty'] * int(Ly/2) ) * int(Lx/2)
else:
    product_state = [IS] * M.lat.N_sites
    
psi = MPS.from_product_state(M.lat.mps_sites(), product_state, bc=M.lat.bc_MPS)

if RM == 'random':
    TEBD_params = {'N_steps': 20, 'trunc_params':{'chi_max': 32}, 'verbose': 0}
    eng = tebd.RandomUnitaryEvolution(psi, TEBD_params)
    eng.run()
    psi.canonical_form() 


chi_list = {0: 4, 4: 8, 8: 16, 12: 32, 16: 64, 20: CHI}

if BC_MPS == 'infinite':
    max_sweep = 1000
    disable_after = 50
    S_err = 1.0e-6
else:
    max_sweep = 1000
    disable_after = 50
    S_err = 1.0e-8

dmrg_params = {
    'mixer' : dmrg.SubspaceExpansion,
    'mixer_params': {
        'amplitude': 1.e-3,
        'decay': 1.5,
        'disable_after': disable_after
    },
    'trunc_params': {
        'chi_max': CHI,
        'svd_min': 1.e-9
    },
    'chi_list': chi_list,
    'max_E_err': 1.0e-8,
    'max_S_err': S_err,
    'max_sweeps': max_sweep,
    'combine' : True
}

ensure_dir(PATH + "observables/")
ensure_dir(PATH + "entanglement/")
ensure_dir(PATH + "logs/")
ensure_dir(PATH + "mps/")

# ground state
eng = dmrg.TwoSiteDMRGEngine(psi, M, dmrg_params)
E, psi = eng.run()  # equivalent to dmrg.run() up to the return parameters.

N = psi.expectation_value("N")
C = np.abs( psi.expectation_value("C") )
EE = psi.entanglement_entropy()
ES = psi.entanglement_spectrum()

if BC_MPS == 'finite':
    lx = Lx-1
    xi = 0.
else:
    lx = Lx
    xi = psi.correlation_length()


file1 = open( PATH + "observables/energy.txt","a")
file1.write(repr(t) + " " + repr(U) + " " + repr(mu) + " " + repr(E) + " " + repr( np.mean(N) ) + " " + repr( np.mean(C) ) + " " + repr(xi) + " " + "\n")

file2 = open( PATH + "observables/numbers.txt","a")
file2.write(repr(t) + " " + repr(U) + " " + repr(mu) + " " + "  ".join(map(str, N)) + " " + "\n")

file3 = open( PATH + "observables/condensation.txt","a")
file3.write(repr(t) + " " + repr(U) + " " + repr(mu) + " " + "  ".join(map(str, C)) + " " + "\n")

file_EE = open( PATH + "entanglement/ee_t_%.2f_U_%.2f_mu_%.2f.txt" % (t,U,mu),"a")
file_EE.write("  ".join(map(str, EE)) + " " + "\n")

file_STAT = open( PATH + "logs/stat_t_%.2f_U_%.2f_mu_%.2f.txt" % (t,U,mu),"a")
file_STAT.write("  ".join(map(str,eng.sweep_stats['E'])) + " " + "\n")
file_STAT.write("  ".join(map(str,eng.sweep_stats['S'])) + " " + "\n")
file_STAT.write("  ".join(map(str,eng.sweep_stats['max_trunc_err'])) + " " + "\n")
file_STAT.write("  ".join(map(str,eng.sweep_stats['norm_err'])) + " " + "\n")

with open( PATH + 'mps/gs_t_%.2f_U%.2f_mu%.2f.pkl' % (t,U,mu), 'wb') as f:
    pickle.dump(psi, f)



print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n\n\n")
