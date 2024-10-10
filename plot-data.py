import icf
import matplotlib.pyplot as plt


# Load in the data from Google Sheets
# Note this load_4col function specifically assumes 
# the data has non-numeric data in column 1, and 
# numeric data in columns 2,3 and 4
stamp, xdata, ydata, yerr = icf.load_4col("capdata.csv")


# Plot a figure with error bars
plt.figure()
plt.errorbar(xdata, ydata,yerr=yerr, fmt='--o', capsize=4)
plt.xlabel("Time / ns")
plt.ylabel("Capsule Radius / um")
plt.show()
