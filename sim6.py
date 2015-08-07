from numpy import * 
from dolfin import *
from scipy.integrate import ode
from bdf import *
from channel_tools import *

### Define the mesh and ice sheet geometry ###

# Input directory
in_dir = "model_inputs/"

mesh = Mesh(in_dir + "mesh_300.xml")
# Standard continuous function space for the sheet model
V = FunctionSpace(mesh, "CG", 1)
# CR function space on edges
V_edge = FunctionSpace(mesh, "CR", 1)
# Vector function space for displaying the flux 
Vv = VectorFunctionSpace(mesh, "CG", 1)

# Bed
B = Function(V)
File(in_dir + "B.xml") >> B

# Thickness
H = Function(V)
File(in_dir + "H.xml") >> H

#H = project(conditional(lt(H, 5.0), 50.0, H), V)

# Basal velocity
u_b = Function(V)
File(in_dir + "u_b.xml") >> u_b

# Melt rate
m = Function(V)
File(in_dir + "m.xml") >> m

# Facet function for marking the margin boundary
boundaries = FacetFunction("size_t", mesh)
File(in_dir + "boundaries.xml") >> boundaries



#### Constants ###          
# Seconds per day
spd = 60.0 * 60.0 * 24.0
# Seconds in a year
spy = spd * 365.0        
m.interpolate(Constant(1.0 / spy))                    
# Density of water (kg / m^3)
rho_w = 1000.0  
# Density of ice (kg / m^3)
rho_i = 910.0
# Gravitational acceleration (m / s^2)
g = 9.81 
# Flow rate factor of ice (1 / Pa^3 * s) 
A = 5.0e-25
# Average bump height (m)
h_r = 0.1
# Typical spacing between bumps (m)
l_r = 2.0
# Sheet width under channel (m)
l_c = 2.0          
# Sheet conductivity (m^(7/4) / kg^(1/2))
k = 1e-2
# Channel conductivity (m^(7/4) / kg^(1/2))
k_c = 1e-1
# Specific heat capacity of ice (J / (kg * K))
c_w = 4.22e3
# Pressure melting coefficient (J / (kg * K))
c_t = 7.5e-8
# Latent heat (J / kg)
L = 3.34e5
# Void storage ratio
e_v = 1e-2
# Exponents 
alpha = 5. / 4.
beta = 3. / 2.
delta = beta - 2.0
# Regularization parameters for Newton solver
phi_reg = 1e-15
# Constant in front of time derivative
c1 = Constant(e_v / (rho_w * g))



### Set up the sheet model ###

# Unknown sheet thickness defined on continuous domain
h = Function(V)
h.interpolate(Constant(0.05))
# Sheet thickness on the channel edges
h_e = Function(V_edge)
# Unknown potential
phi = Function(V)
# Phi at the previous time step
phi_prev = Function(V)
# Potential at the previous time step
phi_prev = Function(V)
# Ice overburden pressure
p_i = project(rho_i * g * H, V)
# Potential due to bed slope
phi_m = project(rho_w * g * B, V)
# Driving potential
phi_0 = project(p_i + phi_m, V)
phi_prev.assign(phi_0)
# Effective pressure
N = phi_0 - phi
# Flux vector
q = -Constant(k) * h**alpha * (dot(grad(phi), grad(phi)) + Constant(phi_reg))**(delta / 2.0) * grad(phi)
# Opening term (m / s) 
w = conditional(gt(h_r - h, 0.0), u_b * (h_r - h) / Constant(l_r), 0.0)
# Closing term
v = Constant(A) * h * N**3
# Water pressure as a fraction of overburden
pfo = Function(V)
# Effective pressure as a function
N_n = Function(V)
# Effective pressure on edges
N_e = Function(V_edge)



### Set up the channel model ###

# Channel cross sectional area defined on edges
S = Function(V_edge)
# S**alpha. This is a work around for a weird bug which causes exponentiation
# on S to fail for no apparent reason.
S_exp = Function(V_edge)
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
# Derivative of phi along channel defined on channel edges
dphi_ds_e = Function(V_edge)



### Set up the PDE for the potential ###

# Set up a measure on the margin outside of the trough
ds = Measure("ds")[boundaries]

theta = TestFunction(V)
dphi = TrialFunction(V)



C = Constant((rho_w * g) / e_v)
print C([0])

# Channel contribution to the RHS of the PDE
RHS_s = C * (dot(grad(theta), q) - (w - v - m) * theta) * dx - C * Constant(0.00000005) * (phi - phi_m) * theta * ds(1)
# Channel contribution to the RHS of the PDE
RHS_c = C * (dot(grad(theta), t) * Q - (w_c - v_c) * theta)('+') * dS
# Variational form
RHS = RHS_s + RHS_c

# Apply 0 pressure at a single point in the trough

class TroughCenter(SubDomain):
  def inside(self, x, on_boundary):
    return near(x[0], 0.0) and abs(x[1] - 10001.0) < 149.5

#bc = DirichletBC(V, phi_m, TroughCenter(), "pointwise")

# Make the initial solution consistent by setting the pressure at the margin
# to 0
#bc.apply(phi_prev.vector())


### Set up the sheet / channel size ODE ### 

# Object for dealing with CR functions
edge_project = EdgeProjector(V, V_edge, mesh)

# Mask that is 0 on mesh edges and 1 on interior edges. This is used to prevent
# opening on exterior edges
theta_e = TestFunction(V_edge)
mask = assemble(theta_e('+') * dS)
mask[mask.array() > 0.0] = 1.0
mask = mask.array()

# Get some of the fields as arrays for use in the ODE solver
# Length of the vector for a CG function
vec_len = V.dim()
# Vector form of phi_m
phi_m_n = phi_m.vector().array()
# Vector form of phi_0
phi_0_n = phi_0.vector().array()
# Vector form of sliding speed
ub_n = u_b.vector().array()
# Vector form of overburden pressure
pi_n = p_i.vector().array()

# This function derives several useful values from the potential including
# values necessary for solving the ODE
def derive_values():
  # Get potential as an array
  phi_n = phi.vector().array()
  # Correct underpressure if necessary
  #under_indexes = phi_n < phi_m_n
  #print ("num under", sum(under_indexes))
  #phi_n[under_indexes] = phi_m_n[under_indexes]
  #phi.vector()[:] = phi_n
  # Derive effective pressure
  N_n.vector()[:] = phi_0_n - phi_n
  # Set the downstream direction based on phi
  edge_project.midpoint(N_n, N_e)
  # Compute derivative of potential along channels
  edge_project.ds(phi, dphi_ds_e)
  # Derive the water pressure
  pw_n = phi_n - phi_m_n
  # Water pressure as a fraction of overburden
  pfo.vector()[:] = pw_n / pi_n
  # Sheet thickness on channel edges
  edge_project.midpoint(h, h_e)

# Slope function for the sheet
def f_h(t, h_n) :
  # Ensure that the sheet height is positive
  h_n[h_n < 0.0] = 0.0
  # Sheet opening term
  w_n = ub_n * (h_r - h_n) / l_r
  # Ensure that the opening term is non-negative
  w_n[w_n < 0.0] = 0.0
  # Sheet closure term
  v_n = A * h_n * N_n.vector().array()**3
  # Return the time rate of change of the sheet
  dhdt = w_n - v_n
  return dhdt

# Slope function for the channel
def f_S(t, h_n, S_n):
  # Ensure that the channel area is positive
  S_n[S_n < 0.0] = 0.0
  
   # Get effective pressures, sheet thickness on edges.
  N_n = N_e.vector().array()
  # Get midpoint values of sheet thickness
  h.vector()[:] = h_n
  edge_project.midpoint(h, h_e)
  h_n =  h_e.vector().array()
  
  # Array form of the derivative of the potential 
  phi_s = dphi_ds_e.vector().array()  
  # Along channel flux
  Q_n = -k_c * S_n**alpha * abs(phi_s + phi_reg)**delta * phi_s
  # Flux of sheet under channel
  q_n = -k * h_n**alpha * abs(phi_s + phi_reg)**delta * phi_s
  # Dissipation melting due to turbulent flux
  Xi_n = abs(Q_n * phi_s) + abs(l_c * q_n * phi_s)
  # Creep closure
  v_c_n = A * S_n * N_n**3
  # Total opening rate
  opening = Xi_n / (rho_i * L)
 
  # Dissalow negative opening rate where the channel area is 0
  opening[opening[S_n == 0.0] < 0.0] = 0.0
  # Calculate rate of channel size change
  dsdt = mask * (opening - v_c_n)
  return dsdt

h_len = len(h.vector())

def rhs(t, Y):
  Ys = np.split(Y, [h_len])
  h_n = Ys[0]
  S_n = Ys[1]
  
  dhdt = f_h(t, h_n)
  dsdt = f_S(t, h_n, S_n)
  
  return hstack((dhdt, dsdt))



### Set up the simulation ###

# Simulation end time
T = 250.0 * spd
# Initial time step
dt = 60.0 * 60.0

# Set some parameters for the Newton solver
prm = NonlinearVariationalSolver.default_parameters()

prm['newton_solver']['relaxation_parameter'] = 1.0
prm['newton_solver']['relative_tolerance'] = 5e-11
prm['newton_solver']['absolute_tolerance'] = 1e-2
prm['newton_solver']['error_on_nonconvergence'] = True
prm['newton_solver']['maximum_iterations'] = 50
prm['newton_solver']['linear_solver'] = 'umfpack'

# Create the time stepping object for the PDE
bdf_solver = BDFSolver(RHS, phi, phi_prev, [], theta, dt, prm)

# Initial condition
Y0 = hstack((h.vector().array(), S.vector().array()))

# Set up integrator
ode_solver = ode(rhs).set_integrator('vode', method='adams', max_step = 60.0 * 5.0)
ode_solver.set_initial_value(Y0, 0.0)

# Create output files 
out_dir = "results4/"
out_h = File(out_dir + "h.pvd")
out_phi = File(out_dir + "phi.pvd")
out_pfo = File(out_dir + "pfo.pvd")
out_N = File(out_dir + "N.pvd")
out_dphi_ds = File(out_dir + "dphi_ds.pvd")
out_pfo = File(out_dir + "pfo.pvd")
out_S = File(out_dir + "S.pvd")
out_q = File(out_dir + "q.pvd")
out_qn = File(out_dir + "qn.pvd") 

# Output some of the static functions as well
File(out_dir + "B.pvd") << B
File(out_dir + "H.pvd") << H
File(out_dir + "p_i.pvd") << p_i
File(out_dir + "phi_m.pvd") << phi_m
File(out_dir + "phi_0.pvd") << phi_0
File(out_dir + "m.pvd") << m
File(out_dir + "phi_init.pvd") << phi_prev


# Create some facet functions to display functions defined on channel edges
S_f = FacetFunction('double',mesh)

# Iteration count
i = 0



### Run the simulation ###

while ode_solver.successful() and ode_solver.t <= T :     
  print ("Current Time:", ode_solver.t / spd)
  
  # Reset the solution in an attempt to help convergence
  phi.assign(phi_m)
  
  bdf_solver.step()

  # Derive some values from the potential   
  derive_values()
  
  #plot(pfo, interactive = True)
    
  # Update the sheet thickness and channel size
  ode_solver.integrate(ode_solver.t + dt)
  Ys = np.split(ode_solver.y, [h_len])
  
  # Retrieve values from the ODE solver
  h.vector()[:] = Ys[0]
  S.vector()[:] = Ys[1]
  
  print ("S bounds", S.vector().min(), S.vector().max())
  print ("h bounds", h.vector().min(), h.vector().max())

  # Compute S**alpha
  S_exp.vector()[:] = S.vector().array()**alpha
  
  if i % 1 == 0:
    # Output a bunch of stuff
    out_h << h
    out_phi << phi
    out_pfo << pfo
    #out_N << N_n
    out_q << project(q, Vv)

    # Copy some functions to facet functions for display purposes
    edge_project.copy_to_facet(S, S_f)
    out_S << S_f
    
  # Checkpoint
  if i % 20 == 0:
    File(out_dir + "h_" + str(i) + ".xml") << h
    File(out_dir + "S_" + str(i) + ".xml") << S
    File(out_dir + "phi_" + str(i) + ".xml") << phi
  
  i += 1
  
  print

  