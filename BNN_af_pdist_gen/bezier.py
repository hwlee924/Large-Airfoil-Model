""" Defines Class Bezier for bezier fits on airfoil and pressure data"""
import numpy as np
import scipy 


def bernstein_poly(i, n, t):
    """
     The Bernstein polynomial of n, i as a function of t
    """
    return scipy.special.comb(n, i) * ( t**i ) * (1 - t)**(n-i)

def bezier_curve(points, interp_loc):
    """
       Given a set of control points, return the
       bezier curve defined by the control points.
    """

    nPoints = len(points)
    xPoints = np.array([p[0] for p in points])
    yPoints = np.array([p[1] for p in points])

    t = interp_loc 
    polynomial_array = np.array([bernstein_poly(i, nPoints-1, t) for i in range(0, nPoints)])

    xvals = np.dot(xPoints, polynomial_array)
    yvals = np.dot(yPoints, polynomial_array)

    return xvals, yvals

def control_points(X, Y, y, BCtype = 'airfoil'):
    """ 
    Calculate control points based on the type of fitting 
    """
    num_ctrl_pts = y.shape[0]+2
    if BCtype == 'airfoil':
            """ 
            Boundary conditions to draw airfoils
            Point 1: [0.0, 0.0] at LE, 
            Point 2: [0.0, y1 ] d
            Point N: [1.0, 0.0] at TE 
            """
            ctrl_x = np.linspace(0.0, X[-1], num_ctrl_pts-1)[:-1] # Control points to be optimized 
            ctrl_x = np.hstack((0.0, ctrl_x, X[-1]))
            ctrl_y = np.hstack((0.0, y, Y[-1])) # 0.0, 
            ctrl_pts = np.hstack((ctrl_x[:,None], ctrl_y[:,None]))    
    elif BCtype == 'pressure':  
            """ 
            Boundary conditions to draw pressure
            Point 1: [0.0, first element of Cp] at LE, 
            Point 2: [0.0, y1 ] d
            Point N: [1.0, last element of Cp] at TE 
            """
            ctrl_x = np.linspace(0.0, X[-1], y.shape[0]+1)[:-1] * np.pi/2 # Control points to be optimized 
            ctrl_x = np.hstack((X[0], 1-np.cos(ctrl_x), X[-1]))
            ctrl_y = np.hstack((Y[0], y, Y[-1])) # 0.0, 
            ctrl_pts = np.hstack((ctrl_x[:,None], ctrl_y[:,None]))    
    else: 
            raise ValueError('Incorrect operation type')
    return ctrl_pts


class Bezier:
    def __init__(self, X, Y, y, BCtype = 'airfoil'): 
        self.X = X 
        self.Y = Y 
        self.BCtype = BCtype
        self.ctrl_pts = control_points(self.X, self.Y, y, self.BCtype)
        self.curve = bezier_curve(self.ctrl_pts, self.X)

    def update(self, new_input):
        """
        update the object w/ the new control points and calculate the loss
        """
        # Conditions, start point, [0, y_1], ..., end point
        self.ctrl_pts = control_points(self.X, self.Y, new_input, self.BCtype)
        self.curve = bezier_curve(self.ctrl_pts, self.X)
        interp_Y = np.interp(self.curve[0], self.X, self.Y)
        self.loss = np.sqrt(np.mean((self.curve[1]-interp_Y)**2)) # Fix this to be more robust?
        # loss.jpg
        return self.loss
