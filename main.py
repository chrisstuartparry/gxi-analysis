import math

import numpy as np
from scipy.optimize import curve_fit

import icf

xdata, ydata = icf.load_2col("ICF_labs_GXI_data/line1.csv")

xdata *= 60.0
xdata /= 3.5


def gaussian(x, *params):

    A = params[0]
    x0 = params[1]
    c = params[2]
    y0 = params[3]

    return y0 + A * np.exp(-((x - x0) ** 2) / (2 * c * c))


n = 6


def super_gaussian(x, *params):

    A = params[0]
    x0 = params[1]
    c = params[2]
    y0 = params[3]
    # n = params[4]

    return y0 + A * np.exp(-(((x - x0) / (np.sqrt(2) * c)) ** n))


# First guess for fitting to then mask
initial_guess = [1, 1, 500, 2]
print("Our initial guess is", initial_guess)
initial_popt, initial_pcov = curve_fit(super_gaussian, xdata, ydata, p0=initial_guess)

initial_yfit = super_gaussian(xdata, *initial_popt)

midpoint_yvalue = len(initial_yfit) // 2

initial_background_average = initial_popt[3]

for yvalue in initial_yfit[midpoint_yvalue::-1]:
    if math.isclose(yvalue, initial_background_average, rel_tol=0.0001):
        lefthand_yvalue = yvalue
        lefthand_yvalue_index = initial_yfit.tolist().index(yvalue) - 2
        break

for yvalue in initial_yfit[midpoint_yvalue:]:
    if math.isclose(yvalue, initial_background_average, rel_tol=0.0001):
        righthand_yvalue = yvalue
        righthand_yvalue_index = initial_yfit.tolist().index(yvalue) + 2
        break

print("Lefthand yvalue is", lefthand_yvalue, "at index", lefthand_yvalue_index)
print("Righthand yvalue is", righthand_yvalue, "at index", righthand_yvalue_index)

xdata_trimmed = xdata[lefthand_yvalue_index:righthand_yvalue_index]
ydata_trimmed = ydata[lefthand_yvalue_index:righthand_yvalue_index]


# This does the fit, and returns the fit parameters and the covariances
guess = [1, 1, 500, 2]
print("Our initial guess is", guess)
popt, pcov = curve_fit(super_gaussian, xdata_trimmed, ydata_trimmed, p0=guess)

for i in range(len(popt)):
    print("Parameter", i, ":", popt[i], "+/-", np.sqrt(pcov[i][i]))

print("Fit parameters : ", popt)
print("Fit standard deviations : ", np.sqrt(np.diag(pcov)))

# This generates a new list with a Gaussian using the identified fit parameters
# This data is therefore the best fit curve
yfit_trimmed = super_gaussian(xdata_trimmed, *popt)

print("R^2 = ", icf.r_squared(ydata_trimmed, yfit_trimmed))


# This will plot the output, both the original data and the best fit, as well as a residual
# Note this is a special plotting routine written for the icf labs, hence the 'icf' prefix
# The source code can be found in icf.py if you want to copy/alter it
icf.fit_plot(
    xdata_trimmed,
    ydata_trimmed,
    yfit_trimmed,
    xl="Position (um)",
    yl="Intensity (arb. units)",
)
