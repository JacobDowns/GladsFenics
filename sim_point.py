from dolfin import *
from glads_model import *
from constants import *
from dolfin import MPI, mpi_comm_world

np.set_printoptions(suppress = True)

# Model input directory
in_dir = "inputs_slope/"
# Output directory
out_dir = "out_point/"
# Process number
MPI_rank = MPI.rank(mpi_comm_world())

# Load mesh and create function spaces
mesh = Mesh(in_dir + "mesh.xml")
V_cg = FunctionSpace(mesh, "CG", 1)
V_cr = FunctionSpace(mesh, "CR", 1)


### Model inputs

# Initial sheet height
h_init = Function(V_cg)
h_init.interpolate(Constant(0.05))

# Initial channel areas
S_init = Function(V_cr)

# Initial potential
phi_init = Function(V_cg)
File(in_dir + "phi_0.xml") >> phi_init

# Load the boundary facet function
boundaries = FacetFunction('size_t', mesh)
File(in_dir + "boundaries.xml") >> boundaries

# Load potential at 0 pressure
phi_m = Function(V_cg)
File(in_dir + "phi_m.xml") >> phi_m

# Mark one point on the margin to be the outlet
def point_boundary(x, on_boundary):
  return near(x[0], 0.0) and abs(x[1] - 10100.0) < 200.0

# Enforce 0 pressure bc at that point
bc = DirichletBC(V_cg, phi_m, point_boundary, "pointwise")

pcs['k'] = 5e-4
pcs['k_c'] = 0.05

model_inputs = {}
model_inputs['mesh'] = mesh
model_inputs['h_init'] = h_init
model_inputs['S_init'] = S_init
model_inputs['phi_init'] = phi_init
model_inputs['d_bcs'] = [bc]
model_inputs['maps_dir'] = in_dir + "maps/"
model_inputs['out_dir'] = out_dir
model_inputs['constants'] = pcs

# Create the Glads model
model = GladsModel(model_inputs, in_dir)


### Run the simulation

# Seconds per day
spd = pcs['spd']
# End time
T = 50.0 * spd
# Time step
dt = 60.0 * 30.0
# Irataion count
i = 0

while model.t < T:
  if MPI_rank == 0: 
    current_time = model.t / spd
    print "Current Time: " + str(current_time)
  
  model.step(dt)
  
  if i % 2 == 0:
    model.write_pvds()
  
  if MPI_rank == 0: 
    print
