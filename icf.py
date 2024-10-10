import numpy as np
import csv
import re
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


	
# This function will calculate an R^2 values for your fit	
def r_squared(data, fit):
	""" Calculate the R^2 value of a fit, f, to a data set, x """
	
	if len(data) != len(fit):
		print("Data and fit lists must be of equal length")
		return None

	m = np.sum(data)/len(data)
	
	# Sum of squares between data and mean
	ss = np.sum((data-m)**2)
	
	# Sum of residuals between fit and data 
	res = np.sum((data-fit)**2)
		
	# This is the definition of R^2
	return 	1 - res/ss


def find_closest(data, v):
	""" Find the value in data closest to v """
		
	return (np.abs(data-v)).argmin()	



def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False	




def load_2col(filename):
	""" load a 2 column file to return to equally sized data arrays """
	
	xdata = []
	ydata = []
	
	with open(filename, 'r') as inputfile:
	
		lines = inputfile.readlines()
		
		for line in lines:
			data = [_f for _f in re.split('[, \t]', line) if _f]
			if is_number(data[0]) and is_number(data[1]):
				xdata.append(float(data[0]))
				ydata.append(float(data[1]))
			
			
	return np.array(xdata), np.array(ydata)


def load_3col(filename):
	""" load a 3 column file to return to equally sized numpy arrays """
	
	x1data = []
	x2data = []
	x3data = []
	
	with open(filename, 'r') as inputfile:
	
		lines = inputfile.readlines()
		
		for line in lines:
			data = [_f for _f in re.split('[,]', line) if _f]
			if is_number(data[0]) and is_number(data[1]) and is_number(data[2]):
				x1data.append(float(data[0]))
				x2data.append(float(data[1]))
				x3data.append(float(data[2]))			

			
	return np.array(x1data), np.array(x2data), np.array(x3data)


def load_4col(filename):
	""" load a 2 column file to return to equally sized lists """
	
	x1data = []
	x2data = []
	x3data = []
	x4data = []
	
	with open(filename, 'r') as inputfile:
	
		lines = inputfile.readlines()
		
		for line in lines:
			data = [_f for _f in re.split('[,]', line) if _f]
			if is_number(data[1]) and is_number(data[2]) and is_number(data[3]):
				x1data.append(data[0])
				x2data.append(float(data[1]))
				x3data.append(float(data[2]))			
				x4data.append(float(data[3]))
			
	return np.array(x1data), np.array(x2data), np.array(x3data), np.array(x4data)



def fixed_parameter_function(function, xdata, params, guesses, fixes):

	if len(guesses)!=len(fixes):
		print("Guesses and fixes arrays must be of equal length!  No fitting done")
		return
		
	fixed_params = []

	j=0
	
	for i in range(len(guesses)):
		if fixes[i] == 1:
			fixed_params.append(guesses[i])
		else:
			fixed_params.append(params[j])
			j+=1
			
	return function(xdata, *fixed_params)
	
		
			
	
	

def fixed_param_curve_fit(func, xdata, ydata, guesses, fixes):


	params = []
	for i in range(len(guesses)):
		if fixes[i] != 1:
			params.append(guesses[i])
						
	popt, pcov = curve_fit(lambda x,*p: fixed_parameter_function(func, x, p, guesses, fixes), xdata, ydata, p0=params)
	
	totpopt = []
	
	j=0
	
	for i in range(len(guesses)):
		if fixes[i] == 1:
			totpopt.append(guesses[i])
		else:
			totpopt.append(popt[j])
			j+=1
	
	#for i in range(len(popt)):
		#for j in range(len(popt)):
			#if fixes[i] == 1 or fixes[j] == 1:
				#pcov[i][j]=0.0

	
	return totpopt, pcov
	
			
def fit_plot(xdata, ydata, fitdata, xl="", yl=""):

	f, axarr = plt.subplots(2, sharex=True)
	axarr[1].plot(xdata, ydata, lw=2, label="Data")
	axarr[1].plot(xdata, fitdata, lw=2, label="Fit")
	axarr[1].legend()
	axarr[1].set_xlabel(xl)
	axarr[1].set_ylabel(yl)	
	axarr[0].plot(xdata, ydata-fitdata, lw=2, label="Residual") 				
	axarr[0].yaxis.tick_right()
	axarr[0].set_ylabel("Residual")


	f.subplots_adjust(hspace=0)
	plt.show()

