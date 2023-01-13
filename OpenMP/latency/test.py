import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(2**7, 2**20, 14)
print(x)
plt.plot(x, x, "*")
plt.show()