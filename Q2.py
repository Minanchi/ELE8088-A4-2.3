from scipy.linalg import solve_continuous_lyapunov as solve_lyapunov
import numpy as np
import matplotlib.pyplot as plt

# Define the system matrix A
A = np.array([[0.8, 0.6], [-0.7, 0.3]])

# Choose a positive definite matrix Q
Q = np.eye(2)

# Solve the Lyapunov equation A'PA - P = -Q to find P
P = solve_lyapunov(A.T, -Q)

# Now, let's plot the ellipsoidal set defined by P
# Generate points on a unit circle
theta = np.linspace(0, 2 * np.pi, 100)
circle_points = np.array([np.cos(theta), np.sin(theta)])

# Transform the unit circle points by P to get the ellipsoid
ellipsoid_points = np.dot(np.linalg.cholesky(P).T, circle_points)

# Plot the ellipsoid
plt.figure(figsize=(6, 6))
plt.plot(ellipsoid_points[0, :], ellipsoid_points[1, :], label='Ellipsoidal Invariant Set')
plt.scatter(0, 0, color='red', label='Origin')  # Mark the origin for reference
plt.title('Ellipsoidal Invariant Set for Given System')
plt.xlabel('x1')
plt.ylabel('x2')
plt.legend()
plt.grid(True)
plt.axis('equal')  # Equal scaling for both axes

# Save the plot to a file
plot_filename = '/mnt/data/ellipsoidal_invariant_set.png'
plt.savefig(plot_filename)

