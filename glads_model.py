from dolfin import *
from cr_tools import *
from constants import *
from hs_solver import *
from phi_solver import *

""" Wrapper class for Mauro's Glads model"""

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
    # Directory storing maps that are used to deal with CR functions in parallel
    self.maps_dir = self.model_inputs['maps_dir']
    # Facet function marking boundaries
    self.boundaries = self.model_inputs['boundaries']
    # Output directory
    self.out_dir = self.model_inputs['out_dir']

    # If the Newton parameters are specified use them. Otherwise use some
    # defaults
    if 'newton_params' in self.model_inputs:
      self.newton_params = self.model_inputs['newton_params']
    else :
      prm = NonlinearVariationalSolver.default_parameters()
      prm['newton_solver']['relaxation_parameter'] = 0.95
      prm['newton_solver']['relative_tolerance'] = 5e-6
      prm['newton_solver']['absolute_tolerance'] = 5e-3
      prm['newton_solver']['error_on_nonconvergence'] = False
      prm['newton_solver']['maximum_iterations'] = 50
      prm['newton_solver']['linear_solver'] = 'mumps'
      
      self.newton_params = prm
    
    # Neumann (flux) boundary conditions
    self.n_bcs = []
    if 'n_bcs' in self.model_inputs:
      self.n_bcs = model_inputs['n_bcs']
    
    # If there is a dictionary of physical constants specified, use it. 
    # Otherwise use the defaults. 
    if 'constants' in self.model_inputs :
      self.pcs = self.model_inputs['constants']
    else :
      self.pcs = pcs
    

    ### Set up a few more things we'll need

    # Function spaces
    self.V_cg = FunctionSpace(self.mesh, "CG", 1)
    self.V_cr = FunctionSpace(self.mesh, "CR", 1)
    
    # Object for dealing with CR functions in parallel
    self.cr_tools = CRTools(self.mesh, self.V_cg, self.V_cr, self.maps_dir)

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
    # Stores the value of S**alpha. A workaround for a bug in Fenics that
    # causes problems when exponentiating a CR function
    self.S_alpha = Function(self.V_cr)
    self.update_S_alpha()
    # Water pressure
    self.p_w = Function(self.V_cg)
    # Pressure as a fraction of overburden
    self.pfo = Function(self.V_cg)
    # Current time
    self.t = 0.0
    
    
    ### Output files
    
    # Facet function for writing cr functions to pvd files
    self.ff_out = FacetFunctionDouble(self.mesh)
    self.S_out = File(self.out_dir + "S.pvd")
    self.h_out = File(self.out_dir + "h.pvd")
    self.phi_out = File(self.out_dir + "phi.pvd")
    self.pfo_out = File(self.out_dir + "pfo.pvd")


    ### Create the solver objects

    # Potential solver    
    self.phi_solver = PhiSolver(self)
    # h and S ODE solver
    self.hs_solver = HSSolver(self)
    

  # Steps phi, h, and S forwardt by dt
  def step(self, dt):
    # Step the potential forward by dt with h and S fixes
    self.phi_solver.step(dt)
    # Step h and S forward with phi fixed
    self.hs_solver.step(dt)
    
    
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
    self.pfo.vector().apply("insert")

    
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
    
  
  # Update S**alpha to reflect current value of S
  def update_S_alpha(self):
    alpha = self.pcs['alpha']
    self.S_alpha.vector().set_local(self.S.vector().array()**alpha)
    
  
  # Write h, S, pfo, and phi to pvd files
  def write_pvds(self):
    self.cr_tools.copy_cr_to_facet(self.S, self.ff_out)
    self.S_out << self.ff_out
    self.h_out << self.h
    self.phi_out << self.phi
    self.pfo_out << self.pfo
  
  
    