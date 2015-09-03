from dolfin import *
from cr_tools import *
from constants import *
from hs_solver import *
from phi_solver import *

class GladsModel():

  def __init__(self, model_inputs, in_dir = None):

    ### Initialize model inputs

    self.mesh = model_inputs['mesh']
    self.V_cg = FunctionSpace(self.mesh, "CG", 1)
    self.V_cr = FunctionSpace(self.mesh, "CR", 1)
    self.model_inputs = model_inputs
    
    # If an input directory is specified, load model inputs from there. 
    # Otherwise use the specified model inputs dictionary.
    if in_dir:
      model_inputs = self.load_inputs(in_dir)
      
    # Ice thickness    
    self.H = self.model_inputs['H']
    # Bed elevation
    self.B = self.model_inputs['B']
    # Basal sliding speed
    self.u_b = self.model_inputs['u_b']
    # Melt rate
    self.m = self.model_inputs['m']
    # Cavity gap height
    self.h = self.model_inputs['h_init']
    # Potential
    self.phi_prev = self.model_inputs['phi_init']
    # Potential at 0 pressure
    self.phi_m = self.model_inputs['phi_m']
    # Ice overburden pressure
    self.p_i = self.model_inputs['p_i']
    # Potential at overburden pressure
    self.phi_0 = self.model_inputs['phi_0']
    # Dirichlet boundary conditions
    self.d_bcs = self.model_inputs['d_bcs']
    # Channel areas
    self.S = self.model_inputs['S_init']
    # A cr function mask that is 1 on interior edges and 0 on exterior edges. 
    # Used to prevent opening on exterior edges.
    self.mask = self.model_inputs['mask']
    # Newton solver parameters
    self.newton_params = self.model_inputs['newton_params']
    # Directory storing maps that are used to deal with CR functions in parallel
    self.maps_dir = self.model_inputs['maps_dir']
    # Facet function marking boundaries
    self.boundaries = self.model_inputs['boundaries']
    
    # Neumann (flux) boundary conditions
    self.n_bcs = []
    if 'n_bcs' in self.model_inputs:
      self.n_bcs = model_inputs['n_bcs']
    
    # If there is a dictionary of physical constants specified, use it. 
    # Otherwise use the defaults. 
    if 'constants' in self.model_inputs :
      self.constants = self.model_inputs['constants']
    else :
      self.constants = physical_constants
    

    ### Set up a few more things we'll need

    self.V_cg = FunctionSpace(self.mesh, "CG", 1)
    self.V_cr = FunctionSpace(self.mesh, "CR", 1)

    # Potential
    self.phi = Function(self.V_cg)
    # Derivative of potential over channel edges
    self.dphi_ds_cr = Function(self.V_cr)
    # Effective pressure
    self.N = Function(self.V_cg)
    # Effective pressure on edges
    self.N_cr = Function(self.V_cr)
    # Sheet height on edges
    self.h_cr = Function(self.V_cr)
    self.update_h_cr()
    # Water pressure
    self.p_w = Function(self.V_cg)
    # Pressure as a fraction of overburden
    self.pfo = Function(self.V_cg)
    # Current time
    self.t = 0.0
    # Object for dealing with CR functions in parallel
    self.cr_tools = CRTools(self.mesh, self.V_cg, self.V_cr, self.maps_dir)


    ### Create the solver objects

    # Potential solver    
    self.phi_solver = PhiSolver(self)
    # h and S ODE solver
    self.hs_solver = HSSolver(self)
    

  # Steps the potential, gap height, and water height forward by dt  
  def solve(self, dt):
    # Step the potential forward by dt with h fixed
    self.phi_solver.solve()
    # Step h forward by dt with phi fixed
    self.h_solver.solve(dt)
    
    
  # Load all model inputs from a directory except for the mesh and initial 
  # conditions on h, h_w, and phi
  def load_inputs(self, in_dir):
    # Bed
    B = Function(self.V_cg)
    File(in_dir + "B.xml") >> B
    # Ice thickness
    H = Function(self.V_cg)
    File(in_dir + "H.xml") >> H
    # Melt
    m = Function(self.V_cg)
    File(in_dir + "m.xml") >> m
    # Sliding speed
    u_b = Function(self.V_cg)
    File(in_dir + "u_b.xml") >> u_b
    # Potential at 0 pressure
    phi_m = Function(self.V_cg)
    File(in_dir + "phi_m.xml") >> phi_m
    # Potential at overburden pressure
    phi_0 = Function(self.V_cg)
    File(in_dir + "phi_0.xml") >> phi_0
    # Ice overburden pressure
    p_i = Function(self.V_cg)
    File(in_dir + "p_i.xml") >> p_i
    # CR function mask 
    mask = Function(self.V_cr)
    File(in_dir + "mask.xml") >> mask
    # Boundary facet function
    boundaries = FacetFunction('size_t', self.mesh)
    File(in_dir + "boundaries.xml") >> boundaries
  
    self.model_inputs['B'] = B
    self.model_inputs['H'] = H
    self.model_inputs['m'] = m
    self.model_inputs['u_b'] = u_b
    self.model_inputs['phi_m'] = phi_m
    self.model_inputs['phi_0'] = phi_0
    self.model_inputs['p_i'] = p_i
    self.model_inputs['mask'] = mask
    self.model_inputs['boundaries'] = mask
    
  
  # Update the effective pressure to reflect current value of phi
  def update_N(self):
    self.N.vector().set_local(self.phi_0.vector().array() - self.phi.vector().array())
    self.N.vector().apply("insert")
    
  
  # Update the water pressure to reflect current value of phi
  def update_pw(self):
    self.p_w.vector().set_local(self.phi.vector().array() - self.phi_m.vector().array())
    self.p_w.vector().apply("insert")
    
  
  # Update the pressure as a fraction of overburden to reflect the current 
  # value of phi
  def update_pfo(self):
    # Update water pressure
    self.update_pw()
    # Compute overburden pressure
    self.pfo.vector().set_local(self.p_w.vector().array() / self.p_i.vector().array())
    self.pfo.apply("insert")

    
  # Update the edge derivatives of the potential to reflect current value of phi
  def update_dphi_ds_cr(self):
    self.cr_tools.ds_assemble(self.phi, self.dphi_ds_cr)
    
  
  # Update effective pressure on edge midpoints to reflect current value of phi
  def update_N_cr(self):
    self.update_N()
    self.cr_tools.midpoint(self.N, self.N_cr)
    
  
  # Updates all fields derived from phi
  def update_phi(self):
    self.phi_prev.assign(self.phi)
    self.update_N_cr()
    self.update_dphi_ds_cr()
    self.update_pfo()
    
  
  # Update the edge midpoint values h_cr to reflect the current value of h
  def update_h_cr(self):
    self.cr_tools.midpoint(self.h, self.h_cr)
    