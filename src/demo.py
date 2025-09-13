import numpy as np
import matplotlib.pyplot as plt
color_tab = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
x = np.arange(0, 9, 1)
data = {6: 1825, 2: 553, 5: 390, 1: 156, 7: 134, 0: 133, 8: 28, 3: 21, 4: 6}
y = np.array([data[xx] for xx in x])
plt.pie(y, labels=x, autopct='%1.1f%%', explode=[0.01] * len(y), startangle=90)
plt.title("The composition of different category in data")
plt.show()
