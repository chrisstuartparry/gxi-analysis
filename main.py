import math

import matplotlib.pyplot as plt
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


def super_gaussian(x, *params):

    A = params[0]
    x0 = params[1]
    c = params[2]
    y0 = params[3]
    # n = params[4]

    return y0 + A * np.exp(-((((x - x0) / (np.sqrt(2) * c))) ** 6))


# First guess for fitting to then mask
initial_guess = [1, 1, 500, 1]
print("Our initial guess is", initial_guess)
initial_popt, initial_pcov = curve_fit(super_gaussian, xdata, ydata, p0=initial_guess)

initial_yfit = super_gaussian(xdata, *initial_popt)

midpoint_yvalue = len(initial_yfit) // 2

initial_background_average = initial_popt[3]

for yvalue in initial_yfit[midpoint_yvalue::-1]:
    if math.isclose(yvalue, initial_background_average, rel_tol=0.0001):
        lefthand_yvalue = yvalue
        lefthand_yvalue_index = initial_yfit.tolist().index(yvalue)
        break

for yvalue in initial_yfit[midpoint_yvalue:]:
    if math.isclose(yvalue, initial_background_average, rel_tol=0.0001):
        righthand_yvalue = yvalue
        righthand_yvalue_index = initial_yfit.tolist().index(yvalue)
        break

print("Lefthand yvalue is", lefthand_yvalue, "at index", lefthand_yvalue_index)
print("Righthand yvalue is", righthand_yvalue, "at index", righthand_yvalue_index)


def sequential_replace(source_arr, target_arr, skip_value=1):
    """
    Sequentially replaces values in target_arr with values from source_arr,
    skipping when source_arr value equals skip_value.

    Parameters:
    -----------
    source_arr : array-like
        Array containing values to replace with
    target_arr : array-like
        Array to be modified
    skip_value : scalar, optional (default=1)
        Value in source_arr to skip during replacement

    Returns:
    --------
    numpy.ndarray
        Modified copy of target_arr
    """
    # Convert inputs to numpy arrays
    source = np.array(source_arr)
    result = np.array(target_arr).copy()

    # Find indices where source array is not equal to skip_value
    valid_indices = np.where(source != skip_value)[0]

    # Perform replacements only at valid indices
    result[valid_indices] = source[valid_indices]

    return result  # Returns: [0, 2, 0, 4, 5]


# Example usage:
source = np.array([1, 2, 1, 4, 5])
target = np.array([0, 0, 0, 0, 0])
result = sequential_replace(source, target)

mask = np.ones_like(xdata)
inverse_mask_min = lefthand_yvalue_index
inverse_mask_max = righthand_yvalue_index
mask[:inverse_mask_min] = initial_background_average
mask[inverse_mask_max:] = initial_background_average
print("Masking from", inverse_mask_min, "to", inverse_mask_max)
print("Giving mask", mask)
plt.plot(xdata, ydata)
plt.show()
plt.plot(xdata, mask)

replaced_ydata = sequential_replace(ydata, mask)

# This does the fit, and returns the fit parameters and the covariances
guess = [1, 1, 500, 1]
print("Our initial guess is", guess)
popt, pcov = curve_fit(super_gaussian, xdata, replaced_ydata, p0=guess)

for i in range(len(popt)):
    print("Parameter", i, ":", popt[i], "+/-", np.sqrt(pcov[i][i]))

print("Fit parameters : ", popt)
print("Fit standard deviations : ", np.sqrt(np.diag(pcov)))

# This generates a new list with a Gaussian using the identified fit parameters
# This data is therefore the best fit curve
yfit = super_gaussian(xdata, *popt)

print("R^2 = ", icf.r_squared(ydata, yfit))


# This will plot the output, both the original data and the best fit, as well as a residual
# Note this is a special plotting routine written for the icf labs, hence the 'icf' prefix
# The source code can be found in icf.py if you want to copy/alter it
icf.fit_plot(xdata, ydata, yfit, xl="Position (um)", yl="Intensity (arb. units)")
