from dolfin import *
import numpy as np
 
class BDFSolver :
   
  def __init__(self, RHS, U, U0, bcs, v, dt, params) :
    
    # Right hand side
    self.F = RHS
    # Unknown 
    self.U = U
    # Last time step for backward Euler
    self.U0 = U0
    # Dirichlet boundary conditions
    self.bcs = bcs
    # Time step for each backward Euler solve
    self.dt = dt
    # Test function
    self.v = v
    # Function space
    self.V = U.function_space()
    # Stores the last two solutions
    self.Us = [Function(self.V) for i in range(2)]
    # Set the first solution
    self.Us[0].assign(U0)
    # Number of steps we've taken
    self.i = 1
    # Constant form of dt
    self.DT = Constant(self.dt)
    # Parameters for the Newton solver
    self.params = params
    # Coefficients for BDF
    c1 = Constant(-4. / 3.)
    c0 = Constant(1. / 3.)
    
    ### Backward Euler variational form
    
    self.F1 = (self.U - self.U0) * v * dx - self.DT * RHS 
    
    # Compute Jacobian of bacward Euler problem
    dU = TrialFunction(self.V)
    self.J1 = derivative(self.F1, self.U, dU) 

    
    ### BDF variational form
    
    dU1 = TrialFunction(self.V)
    
    U0 = self.Us[0]
    U1 = self.Us[1]
    
    self.F2 = (self.U + c1 * U1 + c0 * U0) * v * dx - self.DT * Constant(2. / 3.) * RHS 
    
    # Compute Jacobian of BDF problem
    self.J2 = derivative(self.F2, self.U, dU1) 
    
  def step(self, restart = False):
    
    print ("i", self.i)
    
    # The restart flag indicates that we should take a step with backward Euler
    # so that we can change the time step
    if self.i == 1 or restart:
      # Take one step with Backward Euler
    
      # Solve for the unknown
      solve(self.F1 == 0, self.U, self.bcs, J = self.J1, solver_parameters = self.params)
    
      # Add the solution to a list of the last four solutions
      self.Us[self.i].assign(self.U)
      
      # Store the last solution for backward Euler
      self.U0.assign(self.U)
    else :
      # On subsequent iterations, use a second order BDF method
      
      try:
        # Solve for the unknown
        solve(self.F2 == 0, self.U, self.bcs, J = self.J2, solver_parameters = self.params)
      except:
        # If the initial solve fails retry it with a lower relaxation parameter
        print "Solve failed: Reducing relaxation parameter."
        
        self.params['newton_solver']['relaxation_parameter'] = 0.5
        self.params['newton_solver']['error_on_nonconvergence'] = False
        
        # Reset the solution
        #self.U.interpolate(Constant(0.0))
        
        # Solve for the unknown
        solve(self.F2 == 0, self.U, self.bcs, J = self.J2, solver_parameters = self.params)
          
      # Update the previous solutions
      U1 = self.Us[1]
      U0 = self.Us[0]
      
      U0.assign(U1)
      U1.assign(self.U)
      
      # We will also set this in case we need to change the time step and
      # bootstrap again with backward Euler
      self.U0.assign(self.U)
    
    self.i += 1
    
    

      
  
      
      