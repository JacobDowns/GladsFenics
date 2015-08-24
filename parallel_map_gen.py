# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 15:07:37 2015

@author: jake
"""
from dolfin import *
import numpy as np
from cr_tools import *

# Writes out several functions that are needed in parallel 
class ParallelMapGen():
  
  def __init__(self, mesh) :
    self.mesh = mesh
    # CR function space
    self.V_cr = FunctionSpace(mesh, "CR", 1)
    # CG function space
    self.V_cg = FunctionSpace(mesh, "CG", 1)
    
    # Create a map from edges to facets
    self.e2f = self.calculate_edge_to_facet_map()    
    # Create two maps that map edges to to the two associated vertex dofs
    self.e2d0, self.e2d1 = self.calculate_edge_to_dof_maps()
    
  # This calculates the mapping from facet dof indices to facets.  It is
  # analogous to the V.dofmap().dof_to_vertex_map(mesh) method.
  def calculate_edge_to_facet_map(self):
    mesh = self.V_cr.mesh()
    n_V = self.V_cr.dim()

    # Find coordinates of dofs and put into array with index
    coords_V = np.hstack((np.reshape(self.V_cr.dofmap().tabulate_all_coordinates(mesh),(n_V,2)), np.zeros((n_V,1))))
    coords_V[:,2] = range(n_V)

    # Find coordinates of facets and put into array with index
    coords_f = np.zeros((n_V,3))
    for f in dolfin.facets(mesh):
        coords_f[f.index(),0] = f.midpoint().x()
        coords_f[f.index(),1] = f.midpoint().y()
        coords_f[f.index(),2] = f.index() 

    # Sort these the same way
    coords_V = np.array(sorted(coords_V,key=tuple))
    coords_f = np.array(sorted(coords_f,key=tuple))

    # the order of the indices becomes the map
    V2fmapping = np.zeros((n_V,2))
    V2fmapping[:,0] = coords_V[:,2]
    V2fmapping[:,1] = coords_f[:,2]

    return (V2fmapping[V2fmapping[:,0].argsort()][:,1]).astype('int')
    
  # Computes maps from each edge in a CR function space to the associated
  # vertex dofs on the edge    
  def calculate_edge_to_dof_maps(self):
    # First vertex index associated with each facet
    f_0 = dolfin.FacetFunction('uint', self.mesh)    
    # Second vertex index associated with each facet
    f_1 = dolfin.FacetFunction('uint', self.mesh)

    # Map from vertex index to degree of freedom
    v2d = vertex_to_dof_map(self.V_cg)
    
    # Get the two dof indexes associated with each facet
    for f in dolfin.facets(self.mesh):
      # Vertexes associated with this facet
      v0, v1 = f.entities(0)
      
      # The first dof associated with this facet
      f_0[f] = v2d[v0]
      # The second dof associated with this facet
      f_1[f] = v2d[v1]
      
    edge_to_dof0 = f_0.array()[self.e2f]
    edge_to_dof1 = f_1.array()[self.e2f]
    
    return (edge_to_dof0, edge_to_dof1)
  
  # Writes out several functions to xml that will be used in parallel
  def write_maps(self, out_dir):
    
    # Create a CR function that will be used in parallel to determine the
    # global edge order
    f_cr = Function(self.V_cr)
    f_cr.vector()[:] = np.array(range(self.V_cr.dim()))
    File(out_dir + "/f_cr.xml") << f_cr

    # Copy it to a facet function. This will be used in parallel to map
    # indexes in local facet arrays to edges in a global edge vector
    f_e = FacetFunction('size_t', self.mesh)
    f_e.array()[self.e2f] = f_cr.vector().array()
    File(out_dir + "/f_e.xml") << f_e

    # Create some function that will be used in parallel to map indexes
    # in local edge arrays to vertex dofs in a global vertex dof vector
    e_v0 = Function(self.V_cr)
    e_v1 = Function(self.V_cr)
    
    e_v0.vector()[:] = self.e2d0
    e_v1.vector()[:] = self.e2d1
    
    File(out_dir + "/e_v0.xml") << e_v0
    File(out_dir + "/e_v1.xml") << e_v1
    
    # Create a CG function that will be used in parallel to determine the 
    # global vertex dof order
    f_cg = Function(self.V_cg)
    f_cg.vector()[:] = np.array(range(self.V_cg.dim()))
    File(out_dir + "/f_cg.xml") << f_cg
    File(out_dir + "/f_cg.pvd") << f_cg
    
    # Finally, compute the length of each edge, which will be used to compute
    # derivatives of CG functions over edges
    v_cr = TestFunction(self.V_cr)
    
    L = assemble(v_cr('+') * dS + v_cr * ds).array()
    e_lens = Function(self.V_cr)
    e_lens.vector()[:] = L
    File(out_dir + "/e_lens.xml") << e_lens
