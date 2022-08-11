# Copyright 2022 Hyun-Yong Lee

import numpy as np
from tenpy.models.lattice import Site, Chain, Square
from tenpy.models.model import CouplingModel, NearestNeighborModel, MPOModel, CouplingMPOModel
from tenpy.linalg import np_conserved as npc
from tenpy.tools.params import Config
from tenpy.networks.site import FermionSite  # if you want to use the predefined site
import matplotlib.pyplot as plt
__all__ = ['FERMIONIC_FRACTON']


class FERMIONIC_FRACTON(CouplingModel,MPOModel):
    
    def __init__(self, model_params):
        
        # 0) read out/set default parameters
        if not isinstance(model_params, Config):
            model_params = Config(model_params, "FERMIONIC_FRACTON")
        Lx = model_params.get('Lx', 1)
        Ly = model_params.get('Ly', 1)
        t = model_params.get('t', 1.)
        U = model_params.get('U', 1.)
        mu = model_params.get('mu', 0.)
        bc_MPS = model_params.get('bc_MPS', 'infinite')
        bc = model_params.get('bc', 'periodic')
        QN = model_params.get('QN', 'N')

        site = FermionSite( conserve=QN, filling=0.5 )
        
        lat = Square(Lx=Lx, Ly=Ly, site=site, bc=bc, bc_MPS=bc_MPS)
        CouplingModel.__init__(self, lat)
        
        # 4-site hopping
        self.add_multi_coupling( U, [('Cd', [1,1], 0), ('Cd', [0,0], 0), ('C', [1,0], 0), ('C', [0,1], 0)])
        self.add_multi_coupling( U, [('Cd', [0,1], 0), ('Cd', [1,0], 0), ('C', [0,0], 0) ,('C', [1,1], 0)])
        
        # chemical potential
        self.add_onsite( -mu, 0, 'N')

        # hopping
        for u1, u2, dx in self.lat.pairs['nearest_neighbors']:
            self.add_coupling( -t, u1, 'Cd', u2, 'C', dx, plus_hc=True)
          
        MPOModel.__init__(self, lat, self.calc_H_MPO())

        

        