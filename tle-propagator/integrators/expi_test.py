import matplotlib.pyplot as plt
import numpy as np
from expint import expi #type: ignore

xv = np.logspace(-10, 2, 1000, dtype=np.float64)
ei_expint = np.empty(xv.shape, dtype=np.float64)
ei_mpmath = np.empty(xv.shape, dtype=np.float64)

for i, x in enumerate(xv):
    ei_expint[i] = expi(x)

fig, ax = plt.subplots(figsize=(7, 4))
plot1 = ax.plot(xv, ei_expint, label='expint', color='C0')
ax.set_xticks(np.logspace(-10, 2, 5))
ax.set_yticks(np.logspace(-16, -13, 4))
ax.set_xlabel('x')
ax.set_ylabel(r'$ \frac{|Ei_{expint} - Ei_{mpmath}|}{|Ei_{mpmath}|} $')
ax.set_title('Relative error')
plt.savefig('expi_error.svg')
