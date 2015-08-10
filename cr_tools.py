from dolfin import *
import numpy as np

# Computes directional of CG functions along edges as well as midpoints of CG
# functions along edges in parallel
class CRTools(object):
  
  def __init__(self, mesh, V_cg, V_cr, maps_dir) :
    self.mesh = mesh
    # DG function Space
    self.V_cg = V_cg
    # CR function space
    self.V_cr = V_cr
    # Parallel maps input directory
    self.maps_dir = maps_dir
    # Process
    self.MPI_rank = MPI.rank(mpi_comm_world())
    
    # Load in some functions that help do stuff in parallel
    self.load_maps()
    # Create a map from local facets to global edges    
    self.compute_lf_ge_map()
    # Create maps from local edges to global vertex dofs
    self.compute_le_gv_maps()
    
    f = Function(self.V_cr)
    self.ds(self.f_cg, f)
    
  # Copies a CR function to a facet function
  def copy_cr_to_facet(self, cr, ff) :
    # Gather all edge values from each of the local arrays on each process
    cr_vals = Vector()
    cr.vector().gather(cr_vals, np.array(range(self.V_cr.dim()), dtype = 'intc'))
    # Get the edge values corresponding to each local facet
    local_vals = cr_vals[self.lf_ge]    
    ff.array()[:] = local_vals
     
  
  # Compute a map from local facets to indexes in a global array of edge values
  def compute_lf_ge_map(self):
    #Gather an array of global edge values
    x = Vector()
    self.f_cr.vector().gather(x, np.array(range(self.V_cr.dim()), dtype = 'intc'))
    x = x.array()
    
    # Sort the array
    indexes = x.argsort()

    # Create the map    
    self.lf_ge = indexes[self.f_e.array()]
  
  # Create  maps from local edges to indexes in a global array of vertex values
  def compute_le_gv_maps(self):
    # Gather an array of global vertex values
    x = Vector()
    self.f_cg.vector().gather(x, np.array(range(self.V_cg.dim()), dtype = 'intc'))
    x = x.array()
    
    # Sort the array
    indexes = x.argsort()

    # Create the maps
    self.le_gv0 = indexes[np.array(self.e_v0.vector().array(), dtype = 'int')]
    self.le_gv1 = indexes[np.array(self.e_v1.vector().array(), dtype = 'int')]

  # Computes the directional derivatives of a CG function along each edge and
  def ds(self, cg, cr):
    cr.vector().set_local(self.ds_array(cg))
    cr.vector().apply("insert")

  # Computes the directional derivatives of a CG function along each edge and
  # returns an array
  def ds_array(self, cg):
    # Gather all edge values from each of the local arrays on each process
    cg_vals = Vector()
    cg.vector().gather(cg_vals, np.array(range(self.V_cg.dim()), dtype = 'intc'))  
    
    # Get the two vertex values on each local edge
    local_vals0 = cg_vals[self.le_gv0] 
    local_vals1 = cg_vals[self.le_gv1]
    
    print (self.MPI_rank, "lv0", max(local_vals0 - self.e_v0.vector().array()))
    print (self.MPI_rank, "lv1", max(local_vals1 - self.e_v1.vector().array()))
    #print (self.MPI_rank, "ev0", self.e_v0.vector().array())
    #print (self.MPI_rank, "ev1", self.e_v1.vector().array())
    
    return abs(local_vals0 - local_vals1) / self.e_lens.vector().array()
  
  # Computes the value of a CG functions at the midpoint of edges and copies
  # the result to a CR function
  def midpoint(self, cg, cr):
    cr.vector().set_local(self.midpoint_array(cg))
    cr.vector().apply("insert")
  
  # Computes the value of a CG functions at the midpoint of edges and returns
  # an array
  def midpoint_array(self, cg):
    cg_vals = Vector()
    cg.vector().gather(cg_vals, np.array(range(self.V_cg.dim()), dtype = 'intc'))
    
    # Get the two vertex values on each local edge
    local_vals0 = cg_vals[self.le_gv0] 
    local_vals1 = cg_vals[self.le_gv1]
    
    return (local_vals0 + local_vals1) / 2.0
  
  # Loads all parallel maps from the directory self.maps_dir
  def load_maps(self):
    # Facets to edges
    self.f_e = FacetFunction('size_t', self.mesh)
    File(self.maps_dir + "/f_e.xml") >> self.f_e
    
    # Global edge order                                          
    self.f_cr = Function(self.V_cr)
    File(self.maps_dir + "/f_cr.xml") >> self.f_cr
    
    # Edges to first global vertex dof
    self.e_v0 = Function(self.V_cr)
    File(self.maps_dir + "/e_v0.xml") >> self.e_v0
    
    # Edges to second global vertex dof
    self.e_v1 = Function(self.V_cr)
    File(self.maps_dir + "/e_v1.xml") >> self.e_v1
    
    # Global vertex order
    self.f_cg = Function(self.V_cg)
    File(self.maps_dir + "/f_cg.xml") >> self.f_cg
    
    # Edge lengths
    self.e_lens = Function(self.V_cr)
    File(self.maps_dir + "/e_lens.xml") >> self.e_lens
