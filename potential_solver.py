from dolfin import *
from bdf import *
from cr_tools import *
from constants import *

parameters['form_compiler']['precision'] = 30

class PotentialSolver(object):
  
  # Takes in a dictionary of model inputs including
  # H : Ice thickness (m)
  # B : Bed elevation (m)
  # u_b : Sliding speed (m/s)
  # m : Melt rate (m/s)
  # h0 : Initial sheet height (m)
  # S0 : Initial channel areas (m^2)
  # phi_init : Initial potential (Pa)
  # boundaries: A facet function marking the margin 
  def __init__(self, mesh, model_inputs, newton_params):
    
    # Typical CG function space for phi and h
    V_cg = FunctionSpace(mesh, "CG", 1)
    # CR function space defined on edges for S
    V_cr = FunctionSpace(mesh, "CR", 1)
    
    # Get the bed elevation
    B = model_inputs['B']
    # Get the ice thickness
    H = model_inputs['H']
    # Get basal sliding speed
    u_b = model_inputs['u_b']
    # Get melt rate
    m = model_inputs['m']
    # Get boundary facet function
    boundaries = model_inputs['boundaries']

    ### Set up the sheet model 
    
    # Unknown sheet thickness defined on continuous domain
    h = model_inputs['h0']
    # Unknown potential
    phi = Function(V_cg)
    # Phi at the previous time step
    phi_prev = model_inputs['phi_init']
    # Ice overburden pressure
    p_i = project(rho_i * g * H, V_cg)
    # Potential due to bed slope
    phi_m = project(rho_w * g * B, V_cg)
    # Driving potential
    phi_0 = project(p_i + phi_m, V_cg)
    # Effective pressure
    N = phi_0 - phi
    # Flux vector
    q = -Constant(k) * h**alpha * (dot(grad(phi), grad(phi)) + Constant(phi_reg))**(delta / 2.0) * grad(phi)
    # Opening term (m / s) 
    w = conditional(gt(h_r - h, 0.0), u_b * (h_r - h) / Constant(l_r), 0.0)
    # Closing term
    v = Constant(A) * h * N**3
    # Time step
    dt = Constant(1.0)


    ### Set up the channel model 
    
    # Channel cross sectional area defined on edges
    S = model_inputs['S0']
    # S**alpha. This is a work around for a weird bug which causes exponentiation
    # on S to fail for no apparent reason.
    S_exp = Function(V_cr)
    # Normal and tangent vectors 
    n = FacetNormal(mesh)
    t = as_vector([n[1], -n[0]])
    # Derivative of phi along channel 
    dphi_ds = dot(grad(phi), t)
    # Discharge through channels
    Q = -Constant(k_c) * S_exp * abs(dphi_ds + Constant(phi_reg))**delta * dphi_ds
    # Approximate discharge of sheet in direction of channel
    q_c = -Constant(k) * h**alpha * abs(dphi_ds + Constant(phi_reg))**delta * dphi_ds
    # Energy dissipation 
    Xi = abs(Q * dphi_ds) + abs(Constant(l_c) * q_c * dphi_ds)
    # Another channel source term
    w_c = (Xi / Constant(L)) * Constant((1. / rho_i) - (1. / rho_w))
    # Channel creep closure rate
    v_c = Constant(A) * S * N**3


    ### Set up the PDE for the potential ###
    
    # Measure for integrals o
    ds = Measure("ds")[boundaries]
    theta = TestFunction(V_cg)
    
    # Constant in front of storage term
    C = Constant(e_v / (rho_w * g))
    
    # Storage term
    F_s = C * (phi - phi_prev) * theta * dx
    # Sheet contribution to PDE
    F_s += dt * (-dot(grad(theta), q) + (w - v - m) * theta) * dx 
    # Robin type boundary conditon 
    F_s += dt * Constant(0.00000005) * (phi - phi_m) * theta * ds(1)
    
    # Channel contribution to PDE
    F_c = dt * (-dot(grad(theta), t) * Q + (w_c - v_c) * theta)('+') * dS
    # Variational form
    F = F_s + F_c
    
    # Get the Jacobian
    dphi = TrialFunction(V_cg)
    J = derivative(F, phi, dphi) 
    
    
    ### Make some of stuff accessible to the outside
    self.V_cr = V_cr
    self.V_cg = V_cg
    self.S = S
    self.h = h
    self.phi = phi
    self.phi_prev = phi_prev
    self.dt = dt
    self.F = F
    self.J = J
    self.bcs = []
    self.newton_params = newton_params
    self.S_exp = S_exp

  # Steps the potential forward by dt
  def step(self, dt):
    # Solve for potential
    self.dt.assign(dt)
    solve(self.F == 0, self.phi, self.bcs, J = self.J, solver_parameters = self.newton_params)
    # Update previous solution
    self.phi_prev.assign(self.phi)
  