import matplotlib.pyplot as plt
import numpy as np

# Fixing random state for reproducibility
np.random.seed(19680801)


plt.rcdefaults()
fig, ax = plt.subplots()

# Example data
people = ('batch_size', 'epoch', 'time_step', 'vecteur_h')
y_pos = np.arange(len(people))
#performance = 3 + 10 * np.random.rand(len(people))
performance = [24, 16, 10, 180]

error = np.random.rand(len(people))

ax.barh(y_pos, performance, xerr=error, align='center')
ax.set_yticks(y_pos, labels=people)
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel('Performance')
ax.set_title('How fast do you want to go today?')
ax.legend()

plt.show()