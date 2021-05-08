import numpy as np

n1 = 30     # Size of input
n2 = 6      # Size of output
ns = int(1e4)    # Number of samples
no = 130    # Number of samples needed to get each output

X = np.random.rand(n1, ns)
Y = np.zeros((n2, (ns - no)))

# Parameter matrices
A1 = np.random.rand(n2, n1)
A2 = np.random.rand(no, 1)
b = np.random.rand(n2, 1)

Y[:, 0] = np.squeeze(A1 @ (X[:, :no] @ A2))

for i in range(no + 1, ns):
    Y[:, i - no] = np.squeeze(A1 @ (X[:, (i - no):i] @ A2)) + np.multiply(b.T, Y[:, i - no])