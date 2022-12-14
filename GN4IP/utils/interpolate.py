# Create a class for interpolation

import numpy as np

class interpolator(object):
    '''The interpolator object contains the mesh_points and grid_points for 
    interpolation between the two.
    '''
    
    def __init__(self, mesh_points, grid_size):
        '''Initialize the object using the mesh points and grid_size. 
        Immediately make the grid points using the dimension of the mesh_points
        to determine the dimension of the grid
        '''
        
        self.mesh_points = self.scalePoints(mesh_points)
        self.grid_size = grid_size
        self.dim = mesh_points.shape[0]
        self.makeGridPoints()
        self.inds_mesh2grid = None
        self.inds_grid2mesh = None
    
    def scalePoints(self, points):
        '''Scale the points so that they fit in [-1,1] in each dimension.
        Scale each dimension by the same factor though.
        '''
        
        # Move so minimum dimension is 0
        points = points - points.min(1, keepdims=True)
        
        # Find the maximum
        max_length = points.max()
        
        # Divide all dimensions by half the max
        points = points / (max_length/2)
        
        # Subtract half the max in each dimension to center at [0,0]
        points = points - (points.max(1, keepdims=True) / 2)
        
        # Return these points
        return points
        
    def makeGridPoints(self):
        '''Make the grid points using the dimension and grid_size in self. The
        grid points fit in [-1,1] in each direction. For example, if the grid
        size is 4, the points will be at [-0.75, -0.25, 0.25, 0.75]. Flatten 
        the points into a [dim, grid_size^dim] np.array and store in self.
        '''
        
        # Get the grid points in 1D
        x = np.linspace(-1, 1, self.grid_size, endpoint=False)
        x = x + (x[1]-x[0]) / 2

        # Make the grid in 2D
        if self.dim == 2:
            x_grid, y_grid = np.meshgrid(x, x)
            x_grid = x_grid.flatten()
            y_grid = y_grid.flatten()
            self.grid_points = np.stack((x_grid, y_grid), 0)

        # Make the grid in 3D
        elif self.dim == 3:
            x_grid, y_grid, z_grid = np.meshgrid(x, x, x)
            x_grid = x_grid.flatten()
            y_grid = y_grid.flatten()
            z_grid = z_grid.flatten()
            self.grid_points = np.stack((x_grid, y_grid, z_grid), 0)

    def findNearestPoints(self, points_in, points_out):
        '''Find the inds of the points_in that are closest to the points_out. 
        The output is a np.array of the same length as points_out where the 
        maximum value is the length of points_in.
        '''
        
        # How many points in and out
        n_in = points_in.shape[1]
        n_out = points_out.shape[1]

        # Duplicate the arrays by copying cols and whole arrays
        points_in = np.tile(points_in, n_out)
        points_out = np.repeat(points_out, n_in, axis=1)

        # Compute distances between all points_in and points_out
        d = np.sqrt(np.sum(np.square(points_in-points_out), 0))
        d = np.reshape(d, (n_out, n_in))

        # Return the indicies for the minimum distances
        return np.argmin(d, 1)
    
    def interpMesh2Grid(self, mesh_data):
        '''Interpolate the data from the mesh_points to the grid_points. Use 
        the self.inds_mesh2grid if already computed, otherwise compute those
        first.
        '''
        
        # Compute the indicies
        if self.inds_mesh2grid is None:
            self.inds_mesh2grid = self.findNearestPoints(self.mesh_points, self.grid_points)

        # If output is 2D
        if self.dim == 2:
            
            # Initialize output array
            grid_data = np.zeros((mesh_data.shape[0], mesh_data.shape[1], self.grid_size, self.grid_size))
            
            # Loop through data and do interpolation
            for i in range(mesh_data.shape[0]):
                for j in range(mesh_data.shape[1]):
                    grid_data[i,j,:,:] = np.reshape(mesh_data[i,j,self.inds_mesh2grid], (self.grid_size, self.grid_size))
        
        # If output is 3D
        elif self.dim == 3:
            
            # Initialize output array
            grid_data = np.zeros((mesh_data.shape[0], mesh_data.shape[1], self.grid_size, self.grid_size, self.grid_size))
            
            # Loop through data and do interpolation
            for i in range(mesh_data.shape[0]):
                for j in range(mesh_data.shape[1]):
                    grid_data[i,j,:,:] = np.reshape(mesh_data[i,j,self.inds_mesh2grid], (self.grid_size, self.grid_size, self.grid_size))
        
        # Return the output data
        return grid_data
    
    def interpGrid2Mesh(self, grid_data):
        '''Interpolate the data from the grid_points to the mesh_points. Use
        the self.inds_grid2mesh if already computed, otherwise compute those
        first.
        '''
        
        # Compute the indicies
        if self.inds_grid2mesh is None:
            self.inds_grid2mesh = self.findNearestPoints(self.grid_points, self.mesh_points)
        
        # Initialize the output data
        mesh_data = np.zeros((grid_data.shape[0], grid_data.shape[1], self.mesh_points.shape[1]))
        
        # Loop through data and do interpolation
        for i in range(grid_data.shape[0]):
            for j in range(grid_data.shape[1]):
                mesh_data_ij = grid_data[i,j,:].flatten()
                mesh_data[i,j,:] = mesh_data_ij[self.inds_grid2mesh]
        
        # Return the output data
        return mesh_data
    