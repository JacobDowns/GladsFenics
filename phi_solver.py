from dolfin import *
from constants import *

class PhiSolver(object):
  
  def __init__(self, model):
    
    # Get melt rate
    m = model.m
    # Sheet height
    h = model.h
    # Channel areas
    S = model.S
    # Basal sliding speed
    u_b = model.u_b
    # Potential
    phi = model.phi
    # Potential at previous time step
    phi_prev = model.phi_prev
    # Potential at overburden pressure
    phi_0 = model.phi_0
    # Density of ice
    rho_i = model.constants['rho_i']
    # Density of water
    rho_w = model.constants['rho_w']
    # Rate factor
    A = model.constants['A']
    # Sheet conductivity
    k = model.constants['k']
    # Channel conductivity
    k_c = model.constants['k_c']
    # Bump height
    h_r = model.constants['h_r']
    # Distance between bumps
    l_r = model.constants['l_r']
    # Sheet width under channel
    l_c = model.constants['l_c']
    # Latent heat
    L = model.constants['L']
    # Void storage ratio
    e_v = model.constants['e_v']
    # Gravitational acceleration
    g = model.constants['g']
    # Exponents
    alpha = model.constants['alpha']
    delta = model.constants['delta']
    # Constants in front of storage term
    c1 = e_v / (rho_w * g)
    # Regularization parameter
    phi_reg = 1e-16
    
  
    ### Set up the sheet model 

    # Expression for effective pressure in terms of potential
    N = phi_0 - phi
    # Flux vector
    q = -Constant(k) * h**alpha * (dot(grad(phi), grad(phi)) + phi_reg)**(delta / 2.0) * grad(phi)
    # Opening term 
    w = conditional(gt(h_r - h, 0.0), u_b * (h_r - h) / Constant(l_r), 0.0)
    # Closing term
    v = Constant(A) * h * N**3
    # Time step
    dt = Constant(1.0)


    ### Set up the channel model 
    
    # This function stores the value of S**alpha. It's necessary due to a bug
    # in Fenics that causes problems exponentiating a CR function
    S_alpha = Function(V_cr)
    # Normal and tangent vectors 
    n = FacetNormal(mesh)
    t = as_vector([n[1], -n[0]])
    # Derivative of phi along channel 
    dphi_ds = dot(grad(phi), t)
    # Discharge through channels
    Q = -Constant(k_c) * S_alpha * abs(dphi_ds + Constant(phi_reg))**delta * dphi_ds
    # Approximate discharge of sheet in direction of channel
    q_c = -Constant(k) * h**alpha * abs(dphi_ds + Constant(phi_reg))**delta * dphi_ds
    # Energy dissipation 
    Xi = abs(Q * dphi_ds) + abs(Constant(l_c) * q_c * dphi_ds)
    # Channel creep closure rate
    v_c = Constant(A) * S * N**3
    # Another channel source term
    w_c = (Xi / Constant(L)) * Constant((1. / rho_i) - (1. / rho_w))
    

    ### Set up the PDE for the potential ###
    
    # Measure for integrals over mesh boundaries
    ds = Measure("ds")[boundaries]
    theta = TestFunction(V_cg)
    
    # Constant in front of storage term
    C1 = Constant(c1)
    # Storage term
    F_s = C1 * (phi - phi_prev) * theta * dx
    # Sheet contribution to PDE
    F_s += dt * (-dot(grad(theta), q) + (w - v - m) * theta) * dx 
    
    # Add any non-zero Neumann boundary conditions
    for (m, c) in model.n_bcs: 
      F_s += dt * Constant(c) * theta * ds(m)
    
    # Channel contribution to PDE
    F_c = dt * (-dot(grad(theta), t) * Q + (w_c - v_c) * theta)('+') * dS
    # Variational form
    F = F_s + F_c
    
    # Get the Jacobian
    dphi = TrialFunction(V_cg)
    J = derivative(F, phi, dphi) 
    
    
    ### Assign local variables

    self.F = F
    self.J = J
    self.model = model


  # Steps the potential forward by dt. Returns true if the Newton solver converged or false if it
  # had to use a smaller relaxation parameter.
  def step(self, dt):
    self.dt.assign(dt)
    
    try :
      # Solve for potential
      solve(self.F == 0, model.phi, model.d_bcs, J = self.J, solver_parameters = model.newton_params)
      
      # Update previous solution
      self.phi_prev.assign(self.phi)
      # Derive values from the new potential 
      self.update_phi()
    except :
      # Remember the relaxation parameter
      r = model.newton_params['newton_solver']['relaxation_parameter']

      # Try the solve again with a lower relaxation param
      model.newton_params['newton_solver']['relaxation_parameter'] = 0.7
      model.newton_params['newton_solver']['error_on_nonconvergence'] = False
      solve(self.F == 0, self.phi, self.d_bcs, J = self.J, solver_parameters = self.newton_params)

      # Update previous solution
      self.phi_prev.assign(self.phi)
      # Derive values from potential
      self.update_phi()
      
      # Set the Newton parameters back 
      self.newton_params['newton_solver']['relaxation_parameter'] = r
      self.newton_params['newton_solver']['error_on_nonconvergence'] = True
      
      # Didn't converge with standard params
      return False

    # Did converge with standard params
    return True
  