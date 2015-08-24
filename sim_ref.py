from dolfin import *
from cr_tools import *
from constants import *
from scipy.integrate import ode
from potential_solver import *
import numpy as np
from dolfin import MPI, mpi_comm_world

# Model input directory
in_dir = "inputs_ref/"
# Directory of parallel maps
maps_dir = in_dir + "maps"
# Output directory
out_dir = "out_ref/"
# Checkpoint directory
check_dir = out_dir + "checkpoint/"
# Process
MPI_rank = MPI.rank(mpi_comm_world())

### Load model inputs

mesh = Mesh(in_dir + "ref_mesh.xml")

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




### Setup an object to solve for the potential 

# Initial sheet height
h0 = Function(V_cg)
h0.interpolate(Constant(0.05))
# Initial channel areas
S0 = Function(V_cr)
# Potential at 0 pressure
phi_m = project(rho_w * g * B, V_cg)
# Overburden pressure
p_i = project(rho_i * g * H, V_cg)
# Potential at overburden pressure
phi_0 = project(phi_m + p_i, V_cg)
# Initial value for the potential 
phi_init = Function(V_cg)
phi_init.assign(phi_0)

# 0 pressure on margin
bc = DirichletBC(V_cg, phi_m, boundaries, 1)

# Set some parameters for the Newton solver
prm = NonlinearVariationalSolver.default_parameters()
prm['newton_solver']['relaxation_parameter'] = 1.0
prm['newton_solver']['relative_tolerance'] = 5e-6
prm['newton_solver']['absolute_tolerance'] = 5e-3
prm['newton_solver']['error_on_nonconvergence'] = True
prm['newton_solver']['maximum_iterations'] = 50
prm['newton_solver']['linear_solver'] = 'mumps'

model_inputs = {}
model_inputs['B'] = B
model_inputs['H'] = H
model_inputs['u_b'] = u_b
model_inputs['m'] = m
model_inputs['h0'] = h0
model_inputs['S0'] = S0
model_inputs['phi_init'] = phi_init
model_inputs['boundaries'] = boundaries
model_inputs['bcs'] = [bc]

# Create an object that solves for the potential 
phi_solver = PotentialSolver(mesh, model_inputs, prm)

# Potential
phi = phi_solver.phi
# Channel area
S = phi_solver.S
# S**alpha
S_exp = phi_solver.S_exp
# Sheet height
h = phi_solver.h
# Pressure as a fraction of overburden
pfo = Function(V_cg)
# Effective pressure
N = Function(V_cg)

### Set up the ODEs for h and S

# Create an object for dealing with CR functions 
cr_tools = CRTools(mesh, V_cg, V_cr, maps_dir)

# Sheet thickness on the channel edges
h_e = Function(V_cr)
# Effective pressure on edges
N_e = Function(V_cr)
# Derivative of phi along channel defined on channel edges
dphi_ds_e = Function(V_cr)
# Water pressure
pw = Function(V_cg)
# Derivative of water pressure on channel edges
dpw_ds_e = Function(V_cr)

# This array is 0 on mesh exterior edges and 1 on interior edges. It's used
# to prevent opening of channels on exterior mesh edges
local_mask = mask.vector().array()

# Vector for phi_m
phi_m_n = phi_m.vector().array()
# Vector for phi_0
phi_0_n = phi_0.vector().array()
# Vector for sliding speed
ub_n = u_b.vector().array()
# Vector for overburden pressure
pi_n = p_i.vector().array()
# Length of h vector
h_len = len(h.vector().array())

# This function derives some fields from the potential for use in the oDe
def derive_values():
  # Get potential as an array
  phi_n = phi.vector().array()
  # Derive effective pressure
  N.vector().set_local(phi_0_n - phi_n)
  N.vector().apply("insert")
  # Effective pressure on midpoints
  cr_tools.midpoint(N, N_e)
  # Compute derivative of potential along channels
  cr_tools.ds_assemble(phi, dphi_ds_e)
  # Derive the water pressure
  pw.vector().set_local(phi_n - phi_m_n)
  pw.vector().apply("insert")
  # Compute the derivative of pressure along channels
  #cr_tools.ds(pw, dpw_ds_e)
  # Water pressure as a fraction of overburden
  pw_n = pw.vector().array()
  pfo.vector().set_local(pw_n / pi_n)
  pfo.vector().apply("insert")
  # Get h on midpoitns
  cr_tools.midpoint(h, h_e)

# Slope function for the sheet
def f_h(t, h_n) :
  # Ensure that the sheet height is positive
  h_n[h_n < 0.0] = 0.0
  # Sheet opening term
  w_n = ub_n * (h_r - h_n) / l_r
  # Ensure that the opening term is non-negative
  w_n[w_n < 0.0] = 0.0
  # Sheet closure term
  v_n = A * h_n * N.vector().array()**3
  # Return the time rate of change of the sheet
  dhdt = w_n - v_n
  return dhdt

#f_out = FacetFunctionDouble(mesh)
#cr_out = Function(V_cr)

# Slope function for the channel
def f_S(t, S_n):
  # Ensure that the channel area is positive
  S_n[S_n < 0.0] = 0.0
  # Get effective pressures, sheet thickness on edges.
  N_n = N_e.vector().array()
  # Get midpoint values of sheet thickness
  h_n = h_e.vector().array()
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
  # Disallow negative opening rate where the channel area is 0
  v_o_n[v_o_n[S_n == 0.0] < 0.0] = 0.0
  # Calculate rate of channel size change
  dsdt = local_mask * (v_o_n - v_c_n)
  return dsdt

# Combined slope function for h and S
def rhs(t, Y):
  Ys = np.split(Y, [h_len])
  h_n = Ys[0]
  S_n = Ys[1]
  
  dhdt = f_h(t, h_n)
  dsdt = f_S(t, S_n)
  
  return np.hstack((dhdt, dsdt))


### Set up the simulation

# Simulation end time
T = 500.0 * spd
# Initial time step
dt = 60.0 * 30.0
# Maximum tmie step for ODE solver
dt_max = 5.0 * 60.0
# Iteration count
i = 0

# ODE solver initial condition
Y0 = np.hstack((h.vector().array(), S.vector().array()))
# Set up integrator
ode_solver = ode(rhs).set_integrator('vode', method='adams', max_step = 60.0 * 5.0)
ode_solver.set_initial_value(Y0, 0.0)

# Create output files 
out_h = File(out_dir + "h.pvd")
out_phi = File(out_dir + "phi.pvd")
out_dphi_ds = File(out_dir + "dphi_ds.pvd")
out_pfo = File(out_dir + "pfo.pvd")
out_S = File(out_dir + "S.pvd")
out_N = File(out_dir + "N.pvd")
out_N_e = File(out_dir + "N_e.pvd")

# Output some of the static functions as well
File(out_dir + "B.pvd") << B
File(out_dir + "H.pvd") << H
File(out_dir + "phi_m.pvd") << phi_m
File(out_dir + "phi_0.pvd") << phi_0
File(out_dir + "m.pvd") << m
File(out_dir + "phi_init.pvd") << phi_init

# Create some facet functions to display functions defined on channel edges
S_f = FacetFunction('double',mesh)
dphi_ds_f = FacetFunction('double', mesh)
N_f = FacetFunction('double', mesh)

### Run the simulation

while ode_solver.t <= T :  
  if MPI_rank == 0:   
    print ("Current Time:", ode_solver.t / spd)
  
  # Step the potential forward
  phi_solver.step(dt)
  
  # Derive values for the ODE
  derive_values()  

  # Update the sheet thickness and channel size
  ode_solver.integrate(ode_solver.t + dt)
  Ys = np.split(ode_solver.y, [h_len])
  
  # Retrieve values from the ODE solver
  h.vector().set_local(Ys[0])
  h.vector().apply("insert")
  S.vector().set_local(Ys[1])
  S.vector().apply("insert")

  # Compute S**alpha
  S_exp_n = S.vector().array()**alpha
  S_exp.vector().set_local(S_exp_n)
  S_exp.vector().apply("insert")
  
  if i % 16 == 0:
    # Output a bunch of stuff
    out_h << h
    out_phi << phi
    out_pfo << pfo
    out_N_e << N_e
  
    cr_tools.copy_cr_to_facet(S, S_f)
    out_S << S_f
    
  # Checkpoint
  if i % 96 == 0:
    File(check_dir + "h_" + str(i) + ".xml") << h
    File(check_dir + "S_" + str(i) + ".xml") << S
    File(check_dir + "phi_" + str(i) + ".xml") << phi
  
  i += 1
  
  if MPI_rank == 0:
    print

  