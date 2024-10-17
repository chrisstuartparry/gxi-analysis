import matplotlib.pyplot as plt

import icf

xdata, ydata = icf.load_2col("ICF_labs_GXI_data/line1.csv")

plt.plot(xdata, ydata)
plt.show()
