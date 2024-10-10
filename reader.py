import icf
import matplotlib.pyplot as plt


xdata, ydata = icf.load_2col("Values.txt")

plt.plot(xdata, ydata)
plt.show()

