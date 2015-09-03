from dolfin import *
from cr_tools import *
from constants import *
from scipy.integrate import ode
from potential_solver import *
import numpy as np
from dolfin import MPI, mpi_comm_world

# Model input directory
in_dir = "inputs_incline_up/"
# Directory of parallel maps
maps_dir = in_dir + "maps"
# Output directory
out_dir = "out_incline_up/"
# Checkpoint directory
check_dir = out_dir + "checkpoint/"
# Process
MPI_rank = MPI.rank(mpi_comm_world())

### Load model inputs

mesh = Mesh(in_dir + "mesh_300.xml")

V_cg = FunctionSpace(mesh, "CG", 1)
# CR function space defined on edges
V_cr = FunctionSpace(mesh, "CR", 1)

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
prm['newton_solver']['relative_tolerance'] = 1e-5
prm['newton_solver']['absolute_tolerance'] = 1e-2
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


  