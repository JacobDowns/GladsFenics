from dolfin import *
from bdf import *
from cr_tools import *
from constants import *
from adams_solver import *
from dolfin import MPI, mpi_comm_world

parameters['form_compiler']['precision'] = 30

### Load model inputs

# Model input directory
in_dir = "model_inputs/"
# Directory of parallel maps
maps_dir = "model_inputs/maps"
# Output directory
out_dir = "output/"
# Checkpoint directory
check_dir = "output/checkpoint/"

mesh = Mesh(in_dir + "mesh_300.xml")

V_cg = FunctionSpace(mesh, "CG", 1)
# CR function space defined on edges
V_cr = FunctionSpace(mesh, "CR", 1)
# Vector function space for displaying the flux 
V_v = VectorFunctionSpace(mesh, "CG", 1)

# Bed
B = Function(V_cg)
File(in_dir + "B.xml") >> B

# Thickness
H = Function(V_cg)
File(in_dir + "H.xml") >> H

# Basal velocity
u_b = Function(V_cg)
File(in_dir + "u_b.xml") >> u_b

# Melt rate
m = Function(V_cg)
File(in_dir + "m.xml") >> m

# Facet function for marking the margin boundary
boundaries = FacetFunction("size_t", mesh)
File(in_dir + "boundaries.xml") >> boundaries

# Mask that's 0 on exterior edges and 1 on interior edges
mask = Function(V_cr)
File(in_dir + "mask.xml") >> mask

# Create an object for dealing with CR functions 
cr_tools = CRTools(mesh, V_cg, V_cr, maps_dir)


### Set up the sheet model ###

# Unknown sheet thickness defined on continuous domain
h = Function(V_cg)
h.interpolate(Constant(0.05))
# Sheet thickness on the channel edges
h_e = Function(V_cr)
# Unknown potential
phi = Function(V_cg)
# Phi at the previous time step
phi_prev = Function(V_cg)
# Ice overburden pressure
p_i = project(rho_i * g * H, V_cg)
# Potential due to bed slope
phi_m = project(rho_w * g * B, V_cg)
# Driving potential
phi_0 = project(p_i + phi_m, V_cg)
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
pfo = Function(V_cg)
# Effective pressure as a function
N_n = Function(V_cg)
# Effective pressure on edges
N_e = Function(V_cr)


### Set up the channel model 

# Channel cross sectional area defined on edges
S = Function(V_cr)
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
# Derivative of phi along channel defined on channel edges
dphi_ds_e = Function(V_cr)


### Set up the PDE for the potential ###

# Measure for integrals on margin
ds = Measure("ds")[boundaries]
theta = TestFunction(V_cg)

C = Constant((rho_w * g) / e_v)
# Channel contribution to the RHS of the PDE
RHS_s = C * (dot(grad(theta), q) - (w - v - m) * theta) * dx - C * Constant(0.00000005) * (phi - phi_m) * theta * ds(1)
# Channel contribution to the RHS of the PDE
RHS_c = C * (dot(grad(theta), t) * Q - (w_c - v_c) * theta)('+') * dS
# Variational form
RHS = RHS_s + RHS_c


### Set up the sheet and channel size ODEs 

# This array is 0 on mesh exterior edges and 1 on interior edges. It's used
# to prevent opening of channels on exterior mesh edges
local_mask = mask.vector().array()

# Length of the local vector for a CG function
vec_len = len(h.vector().array())
# Vector form of phi_m
phi_m_n = phi_m.vector().array()
# Vector form of phi_0
phi_0_n = phi_0.vector().array()
# Vector form of sliding speed
ub_n = u_b.vector().array()
# Vector form of overburden pressure
pi_n = p_i.vector().array()

# This function derives some fields from the potential for use in the oDe
def derive_values():
  # Get potential as an array
  phi_n = phi.vector().array()
  # Derive effective pressure
  N_n.vector()[:] = phi_0_n - phi_n
  # Set the downstream direction based on phi
  cr_tools.midpoint(N_n, N_e)
  # Compute derivative of potential along channels
  cr_tools.ds(phi, dphi_ds_e)
  # Derive the water pressure
  pw_n = phi_n - phi_m_n
  # Water pressure as a fraction of overburden
  pfo.vector()[:] = pw_n / pi_n

# Slope function for the sheet ODE
def f_h(t, Xs) :
  h_n = Xs[0]
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

# Slope function for the channel ODE
def f_S(t, Xs):
  S_n = Xs[1]
  
  # Ensure that the channel area is positive
  S_n[S_n < 0.0] = 0.0
  
   # Get effective pressures, sheet thickness on edges.
  N_n = N_e.vector().array()
  
  # Get midpoint values of sheet thickness
  cr_tools.midpoint(h, h_e)
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
  v_o_n = Xi_n / (rho_i * L)
 
  # Dissalow negative opening rate where the channel area is 0
  v_o_n[v_o_n[S_n == 0.0] < 0.0] = 0.0
  
  # Calculate rate of channel size change
  dsdt = local_mask * (v_o_n - v_c_n)
  return dsdt


### Set up the simulation

# Simulation end time
T = 250.0 * spd
# Initial time step
dt = 60.0 * 30.0
# Maximum tmie step for ODE solver
dt_max = 5.0 * 60.0
# Iteration count
i = 0

# Set some parameters for the Newton solver
prm = NonlinearVariationalSolver.default_parameters()

prm['newton_solver']['relaxation_parameter'] = 1.0
prm['newton_solver']['relative_tolerance'] = 1e-11
prm['newton_solver']['absolute_tolerance'] = 5e-3
prm['newton_solver']['error_on_nonconvergence'] = True
prm['newton_solver']['maximum_iterations'] = 50
prm['newton_solver']['linear_solver'] = 'mumps'
#prm['newton_solver']['preconditioner'] = 'ilu'

# Create the time stepping object for the PDE
bdf_solver = BDFSolver(RHS, phi, phi_prev, [], theta, dt, prm)
# Create the ODE solver
ode_solver = AdamsSolver([h, S], [f_h, f_S], init_t = 0.0, init_dt = dt, dt_max = dt_max, tol = 1e-7, verbose = True)  

# Create output files 
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


### Run the simulation

while ode_solver.t <= T :     
  print ("Current Time:", ode_solver.t / spd)
  
  # Step the potential forward
  bdf_solver.step()
  # Derive some values from the potential for the ODE 
  derive_values()
  # Step the sheet and channel ODEs forward
  ode_solver.step(dt) 
  
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
    out_q << project(q, V_v)

    # Copy some functions to facet functions for display purposes
    cr_tools.copy_cr_to_facet(S, S_f)
    out_S << S_f
    
  # Checkpoint
  if i % 20 == 0:
    File(check_dir + "h_" + str(i) + ".xml") << h
    File(check_dir + "S_" + str(i) + ".xml") << S
    File(check_dir + "phi_" + str(i) + ".xml") << phi
  
  i += 1
  
  print

  