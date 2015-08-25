"""
This script generates all fields such as ice thickness, bed elevation, and melt
needed to run the model. 
"""

from dolfin import *
from parallel_map_gen import *
from constants import *

# Directory to write model inputs
out_dir = "inputs_thin_margin/"
# Directory to write parallel maps
maps_dir = out_dir + "maps"

mesh = Mesh("inputs_thin_margin/ref_mesh.xml")
V_cg = FunctionSpace(mesh, "CG", 1)
V_cr = FunctionSpace(mesh, "CR", 1)

### Create the parallel maps
# Create the parallel map generator
map_gen = ParallelMapGen(mesh)
# Write maps to a folder
map_gen.write_maps(maps_dir)


# Maximum ice thickness
H_max = 1500.0
# Length of ice sheet
L = 60e3
# Length of flat region
F = 5e3


# Bed topography
class Bed(Expression):
  def eval(self,value,x):
    x = max(x[0] - F, 1.0)
    value[0] = sin(0.5 * pi / 180) * x
  
B = project(Bed(), V_cg)

# Surface topography 
class Surface(Expression):
  def eval(self,value,x):
    x = max(x[0] - F, 1.0)
    value[0] = x              
    value[0] = max( sqrt(x * H_max**2 / L) , 1.0)
    
S = project(Surface(), V_cg)

# Thickness
H = project(S - B, V_cg)


### Create a melt function (m/s)
m = project(Expression("(1.0 + 1.5 * (50000.0 - x[0]) / 50000.0) / 31536000.0"), V_cg)

File(out_dir + "m.xml") << m
File(out_dir + "m.pvd") << m

File(out_dir + "B.xml") << B
File(out_dir + "B.pvd") << B

File(out_dir + "H.xml") << H
File(out_dir + "H.pvd") << H

File(out_dir + "m.xml") << m
File(out_dir + "m.pvd") << m


#### Create a sliding sliding speed (m/s)
u_b = project(Expression("(10.0 + 240.0 * (50000.0 - x[0]) / 50000.0) / 31536000.0"), V_cg)

File(out_dir + "u_b.xml") << u_b
File(out_dir + "u_b.pvd") << u_b


### Create a facet function with marked boundaries

# Margin
class MarginSub(SubDomain):
  def inside(self, x, on_boundary):
    return on_boundary and near(x[0], 0.0)
    
# Divide 
class DivideSub(SubDomain):
  def inside(self, x, on_boundary):
    return on_boundary and near(x[0], 50000.0)
    
ms = MarginSub()
ds = DivideSub()

boundaries = FacetFunction("size_t", mesh)
boundaries.set_all(0)
ms.mark(boundaries, 1)
ds.mark(boundaries, 2)

File(out_dir + "boundaries.xml") << boundaries
File(out_dir + "boundaries.pvd") << boundaries


### Create a mask for the ODE solver

# Create a mask array that is 0 on mesh edges and 1 on interior edges. This 
# is used to prevent opening on exterior edges
v_cr = TestFunction(V_cr)
mask = assemble(v_cr('+') * dS)
mask[mask.array() > 0.0] = 1.0

mask_cr = Function(V_cr)
mask_cr.vector()[:] = mask.array()

File(out_dir + "mask.xml") << mask_cr
File(out_dir + "mask.pvd") << mask_cr
