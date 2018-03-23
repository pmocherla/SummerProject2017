# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 10:32:18 2016

@author: Priyanka Mocherla
@version: 2.7

Code created to extract the data from sdt files and sort the data into a more understandable format. The functions here scale the data after transforming them into eV and then there is an example code that uses the functions to plot the fits of the data and the extracted values.
Interval fitting is used to optimise the fit of the Gaussian.

Functions in this module:
    - tempRamp
    - openData
    - data_xy
    - PLjacobian
    - scale_PL
    - find_nearest
    - peaks
    - norm_PL
    - contourplot
    - Gauss
    - guess
    - interval_fitting
    - interval_minimisation
    - FWHM
    - peaks_FWHM_plot
    - errsFromCov
    - FWHMErr
    - peakEnergyError


"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from collections import defaultdict
from lmfit import  Model
import datetime
import glob
import re
import math

def tempRamp(data):
    plotting = []

    for line in data:
        for i in range(len(line)):
            if line[i] == 'T':
                split = line[i+1::].split()
                plotting.append((split[0],float(split[1]),float(split[2])))
            
    times = [datetime.datetime.strptime(i, "%H:%M:%S.%f") for i,j,k in plotting]
    ramp_temp = [j for i,j,k in plotting]
    true_temp = [k for i,j,k in plotting]
    
    return times, ramp_temp, true_temp
    


def openData(sample, temp_data):
    file_data = glob.glob('./*.txt')
    file_data = [i for i in file_data if sample in i]
    times_index = []
    
    for T in range(len(file_data)):
        file_time = re.findall( r'_\w+-\w+-\w+-\w+',file_data[T])
        if len(file_time) != 0:
            file_time = re.split('-|_| ',file_time[0])
            file_time = "-".join(file_time[-4::])
            file_time = datetime.datetime.strptime(file_time, "%H-%M-%S-%f")
            times_index.append((file_time,file_data[T]))
            
    ref_temps = tempRamp(temp_data)
    start = min(ref_temps[0])
    end = max(ref_temps[0])
    
    temperatures = []
   
    for i in range(len(times_index)):

        if times_index[i][0] < end and times_index[i][0] > start:
            closest = sorted(ref_temps[0], key=lambda d: abs(times_index[i][0]-d))[0]
            temperature = ref_temps[1][ref_temps[0].index(closest)]
            temperatures.append((temperature,times_index[i][0]))
            
    matched_data = []
    
    for i in range(len(temperatures)):
        for j in range(len(times_index)):
            if temperatures[i][1] == times_index[j][0]:
                matched_data.append((temperatures[i][0],times_index[j][1]))
                
                
    return matched_data


def data_xy(data):
    """ Method to turn dictionary to x (wavelength) and y (PL) lists

            Parameters
            ----------
            data: dictionary 
                measured data indexed by temperature

            Returns
            -------
            Numpy arrays of the x and y split data
    """
    x = []
    y = []
    for temp in temps:
            unsplit = data['temp_%02d' % temp]
            xi = np.array([i for i,j,k in unsplit])
            yi = np.array([j for i,j,k in unsplit])
            x.append(xi)
            if np.isnan(yi).any() != True:
                y.append(yi)
            else:
                y.append(np.random.random(len(yi,)))
    return np.array(x),np.array(y)

def wavetoeV(wavelengths):
    """ Method to convert a list of wavelengths to electron volts

            Parameters
            ----------
            wavelengths: list
                list of wavelengths in nm

            Returns
            -------
            Numpy array containing the converted electron volts
    """
    
    h = 4.136e-15  # plancks constant in eVs
    c = 3e8*1000000000 # speed of light in nm/s
    
    eV = [h*c/i for i in wavelengths]
    
    return np.array(eV)
    
    
    
def PLjacobian(PL_yi, wavelengths):
    """ Method to scale the PL data according to the jacobian transform

            Parameters
            ----------
            PL_y: list
                list of a single set of PL measurements 
                
            wavelengths: list
                list of wavelengths in nm

            Returns
            -------
            Numpy array containing the PL measurements after a jacobian transformation
    """
    PL_yi = np.array(PL_yi)
    h = 4.136e-15 #  nano scaling thousand missing??
    c = 3e8 #speed of light
    
    eV = wavetoeV(wavelengths)
    PL_scaled = np.zeros(PL_yi.size)
    
    for i in range(eV.size): 
        PL_scaled[i] = PL_yi[i]*4.14e-15*3e8/(eV[i]*eV[i]) #jacobian
        
    return PL_yi
    
            
def scale_PL(x,y,temperature_range):
    """ Method to convert the wavelengths and PLs into their transformed eV state

            Parameters
            ----------
            x: list of 1D arrays
                list of the wavelengths for each PL measurement
                
            y: list of 1D arrays
                list of the PL measurements at each temperature
                
            temperature_range: list
                list of temperatures measured

            Returns
            -------
            Numpy array of the scaled x and y data
    """    
    for i in range(len(temperature_range)):
        x[i] = wavetoeV(x[i])
        y[i] = PLjacobian(y[i],x[i])
    
    return x,y
    


def find_nearest(existing_values,input_value):
    """ Method to find nearest number in a list to an input

            Parameters
            ----------
            existing_values:list
                list of values that are to be scanned
                
            input_value: float
                the value that you would like to find the nearest value to

            Returns
            -------
            the index of the nearest value
    """  
    existing_values = np.array(existing_values)
    
    lst =  existing_values - [input_value]*existing_values.size 
    nearest = min(lst, key=abs) + input_value #finding the smallest difference between input and list
    
    #insert condition for equidistance?
    
    return np.where(existing_values == nearest)[0][0]
    
    

def peaks(PL_y):
    """ Method to find the peaks of multiple data sets

            Parameters
            ----------
            PL_y: list 
                list of arrays to find peaks of each array

            Returns
            -------
            the peak of each array in the list
    """  

    peak_list = np.zeros(len(PL_y))
    
    for i in range(len(PL_y)):
        peak_list[i] = max(PL_y[i])
        
    return peak_list
    

def norm_PL(PL_y):
    """ Method to normalise datasets using their peak values

            Parameters
            ----------
            PL_y:list
                list of values that are to be scanned


            Returns
            -------
            the index of the nearest value
    """
    peak_list = peaks(PL_y)
    PL_ynorm = []
    
    for i in range(len(PL_y)):
        PL_ynorm.append(np.array(PL_y[i]/peak_list[i]))
        
    return PL_ynorm
    

       
def contourplot(data, temperature_range):
    """ Method to produce a contour plot of the normalised data

            Parameters
            ----------
            data: dictionary
                dictionary of PL measurement results at each temperature
                
            temperature_range: list
                list of temperatures measured


            Returns
            -------
            the contour plot coordinates of the data and contour plot
    """
    
    xi,zi = data_xy(data)
    x,z = scale_PL(xi,zi,temperature_range)

    
    Z = np.array(norm_PL(z))
    X,Y = np.meshgrid(x[0],temperature_range)
    plt.contour(X,Y,Z, levels = np.arange(0,1.000,0.0001) )
    
    clb = plt.colorbar()
    clb.ax.set_title('PL')
    plt.title('Normalised PL')
    plt.xlabel('Wavelength/ nm', fontsize = 18)
    plt.ylabel('Temperature/ K', fontsize = 18)
    plt.show()  
    return X,Y,Z

        
def Gauss(x, a, x0, wid):
    """ Method to return a gaussian function with specified parameters

            Parameters
            ----------
            x: 1D array
                array of eV values
            
            a: float
                stretch factor of the gaussian
                
            x0: float
                mean of the gaussian
                
            wid: float
                width (standard deviation) of the gaussian


            Returns
            -------
            a gaussian function
    """
    return a * np.exp(-(x - x0)**2 / (2 * wid**2))
    
def guess(x,y):     
    """ Method to estimate the first and second moments and peak of gaussians

            Parameters
            ----------
            x: 1D array
                array of eV values
            
            y: 1D array
                array of PL values

            Returns
            -------
            the estimate mean, standard deviation, and peak of the gaussian
    """
    mean = sum(x*y)/sum(y)
    sigma = np.sqrt(sum(y * (x - mean)**2) / sum(y))

    
    return mean, sigma
    
def interval_fitting(data, temperatures, interval, trial_temp, tolerance, fix_amplitude = True, normalise = True, show_fit_report = False):
    """ Method to fit a single, full data set - setting an interval to fit a gaussian

            Parameters
            ----------
            data: dictionary
                dictionary of PL measurements for indexed at each temperature
            
            temperatures: list 
                list of temperatures measured
                
            interval: list (length 2)
                start and end of points to be sampled
            
            trial_temp: integer
                the temperature that is to be fitted
                
            tolerance: integer
                minimum number of points allowed in interval
                
            fix_amplitude: Boolean
                initilised to be True, fixes the maximum amplitude of the fitting
                
            normalise: Boolean
                initilised to be True, normalises the data
                
            show_fit_report: Boolean
                initilised to be False, returns a full fit report from the lmfit module
                
    
            Returns
            -------
            the x,y data, parameters for gaussian fit, details of the fitting conditions
    """
    if trial_temp not in temperatures:
        return "Please enter a valid temperature in the measured range"
    
    index = temperatures.index(trial_temp) #temperature index for dataset
    classification = []
    
    plotting = data_xy(data)
    x,y = scale_PL(plotting[0],plotting[1], temperatures) 
    
    if normalise == True:
        y = norm_PL(y)
        classification.append('Normalised')
       
             
    final = find_nearest(x[index],interval[0])
    initial = find_nearest(x[index],interval[1])
    
    if final - initial < tolerance:
        return "Please increase interval or decrease tolerace"
 
    xi = x[index][initial:final]
    yi = y[index][initial:final]
    
    intvals = x[index][initial],x[index][final]
    
    parameter_guesses = guess(xi,yi)
    peak = max(yi)
    gmodel = Model(Gauss)
    
    
    params = gmodel.make_params(a=peak,x0=parameter_guesses[0],wid=parameter_guesses[1])
    
    
    result = gmodel.fit(yi,x=xi,a=peak,x0=parameter_guesses[0],wid=parameter_guesses[1])
    
    if show_fit_report == True:
        print(result.fit_report())
        
    if fix_amplitude == True:
        amp_new = peak
        classification.append('Amplitude Fixed')
    else:
        amp_new = result.best_values['a']
    
    sigma_new = result.best_values['wid']
    mean_new = result.best_values['x0']
    new_params = amp_new, mean_new, sigma_new
    
    return x[index], y[index], new_params, classification, intvals, result.chisqr,  result.covar, result.fit_report(min_correl=0.0)


def interval_minimisation(data, temperatures, interval, trial_temperature, tolerance, scan_step, fix_amplitude = True, normalise = True):
    """ Method to fit a single, full data set - by varying the interval range between specified limits and finding the best fit

            Parameters
            ----------
            data: dictionary
                dictionary of PL measurements for indexed at each temperature
            
            temperatures: list 
                list of temperatures measured
            
            interval: list (length 2)
                start and end of points to be sampled
            
            trial_temperatures: list
                a list of temperatures to be fitted. Initialised to 'All' temps in temperatures
                
            tolerance: integer
                minimum number of points allowed in interval
                
            scan_step: float
                the interval scanning step size (typically 0.01)
                
            fix_amplitude: Boolean
                initilised to be True, fixes the maximum amplitude of the fitting
                
            normalise: Boolean
                initilised to be True, normalises the data
                
            show_fit_report: Boolean
                initilised to be False, returns a full fit report from the lmfit module
                
    
            Returns
            -------
            the x,y data, parameters for gaussian fit and details of the fitting conditions and chi_squared value.
            
    """
    tr_x = np.arange(interval[0],interval[1],scan_step)
    tr_y = tr_x
    chisquare =[]
    
    test = interval_fitting(data, temperatures, interval, trial_temperature, tolerance, fix_amplitude, normalise)
    mean = test[2][1]
    
    try:
        for i in tr_x:
            for j in tr_y:
                if i < j and i < mean and j > mean:
                    inter = [round(i,2),round(j,2)]
                    test = interval_fitting(data, temperatures, inter, trial_temperature, tolerance, fix_amplitude, normalise)
                    mean = test[2][1]
                    chisquare.append([test[5],(i,j)])
                    """
                    plt.figure()
                    plt.plot(test[0],test[1],'b+:', label = str(inter) + str(test[3]))
                    plt.plot(test[0],Gauss(test[0],test[2][0],test[2][1],test[2][2]),'r')
                    plt.legend()
            
                    plt.show()
                    """
                    
    except IndexError:
        pass
    
    chis = [i for i,j in chisquare]
    try:
        bestfit = min(chis)
        int_index = chis.index(bestfit)

    
        bestint = chisquare[int_index][1]
        final = interval_fitting(data, temperatures, bestint, trial_temperature, tolerance, fix_amplitude, normalise)
        
        return final
    
    except ValueError:
        return "Fit Failed"
        pass
        

def FWHM(standard_dev):
    """ Method convert the standard deviation of a gaussian to the full width at half maximum
    
            Parameters
            ----------
            standard_dev: float
                standard deviation of a gaussian distribution
                
            
            Returns
            -------
            the FWHM of the distribution
    """ 
            
    return 2*np.sqrt(2*np.log(2))*standard_dev
    
    
def peaks_FWHM_plot(data,temperature_range,interval,trial_temperature,tolerance,scan_step, fix_amplitude = True, normalise = True):
    """ Method to produce plots for the peak enery and the FWHM with respect to temperature

            Parameters
            ----------
            data: dictionary
                dictionary of PL measurements for indexed at each temperature
            
            temperatures: list 
                list of temperatures measured
            
            interval: list (length 2)
                start and end of points to be sampled
            
            trial_temperatures: list
                a list of temperatures to be fitted. Initialised to 'All' temps in temperatures
                
            tolerance: integer
                minimum number of points allowed in interval
                
            scan_step: float
                the interval scanning step size (typically 0.01)
                
            fix_amplitude: Boolean
                initilised to be True, fixes the maximum amplitude of the fitting
                
            normalise: Boolean
                initilised to be True, normalises the data
                
                
    
            Returns
            -------
            the fitted peaks, peak energy and FWHM as a function of temperature
    """
    peaksplot = []
    FWHMvals = []
    tempsplot = []
    covar_mats = []
    correls = []
    
    N = len(trial_temperature)
    cols = int(np.sqrt(N))
    rows = int(math.ceil(N / cols))
    if rows*cols < N:
        rows = rows + 1
    fig, axes = plt.subplots(rows, cols, sharex=True, sharey=True)
    
    
    for i in trial_temperature:
        test = interval_minimisation(data,temperature_range,interval,i,tolerance,scan_step, fix_amplitude, normalise)
        

        if test != 'Fit Failed':
            tempplot = plt.subplot(rows,cols,trial_temperature.index(i)+1)
            tempplot.set_title(str(round(i,1)) + ' K ') #+ str(round(test[4][0],2))+' - '+ str(round(test[4][1],2))+' fit')
            tempplot.plot(test[0],test[1],'b.')
            tempplot.plot(test[0],Gauss(test[0],test[2][0],test[2][1],test[2][2]),'r')
            tempplot.axvline(test[4][0], linestyle='--')
            tempplot.axvline(test[4][1], linestyle = '--')
            tempplot.legend()
            
            
            peaksplot.append(test[2][1])
            tempsplot.append(i)
            FWHMvals.append(FWHM(test[2][2]))
            
            print str(((trial_temperature.index(i)+1)/float(len(trial_temperature)))*100) +'% Complete'
            covar_mats.append(test[6])
            correls.append(test[7])
            
        else:
            tempplot = plt.subplot(rows,cols,trial_temperature.index(i)+1)
            tempplot.annotate(str(i) + ' K N/A',xy=(0, 0.5))
            print "The fit for " +str(i)+"K failed and was skipped."
    
    fig.tight_layout()
    print '100% Complete'

    upperlim = max(peaksplot) *2
    plt.figure()
    plt.title('Peak Energies')
    plt.xlabel('Temperature / K')
    plt.ylabel('Energy/ eV')
    plt.plot(tempsplot, peaksplot, 'o')
    plt.ylim([0,upperlim])
        
    
    plt.figure()
    plt.title('Peak widths')
    plt.xlabel('Temperature / K')
    plt.ylabel('FWHM/ eV')
    upperlim = max(FWHMvals) *2
    plt.plot(tempsplot, FWHMvals, 'o')
    plt.ylim([0,upperlim])
    plt.show()
    

    return tempsplot, FWHMvals, peaksplot,covar_mats,correls
    
def errsFromCov(cov_mats):
    height_errs = []
    mean_errs = []
    sigma_errs = []
    
    for i in range(len(cov_mats)):
        height_errs.append(np.sqrt(cov_mats[i][0][0]))
        mean_errs.append(np.sqrt(cov_mats[i][1][1]))
        sigma_errs.append(np.sqrt(cov_mats[i][2][2]))
        
    return height_errs, mean_errs, sigma_errs
    
def FWHMErr(cov_mats):
    sigma_errors = errsFromCov(cov_mats)[2]
    
    FWHM_error = [ FWHM(i) for i in sigma_errors]
    return np.array(FWHM_error)
    
def peakEnergyError(cov_mats):
    mean_errors = errsFromCov(cov_mats)[1]
    
    return np.array(mean_errors)
    

#-------------------------------------- Load Data -----------------------------------#
    
#Enter the name of the sample and the corresponding data for the temperature ramp
sample = 'MAPI'
tempdata = open("Temp_3.txt",'r')

#Open the data in the required format
files = openData(sample,tempdata)

#------------------------------------ Example Code ---------------------------------#
PL_data = {}
for i in range(len(files)):
    PL_data[ '%02d' % files[i][0]] = np.loadtxt("%02s"%files[i][1])
    
    
maxx = []
maxy = []
temps = [i for i,j in files]
colormap = plt.cm.gist_ncar
plt.gca().set_color_cycle([colormap(i) for i in np.linspace(0, 0.9, len(temps))])

keys = PL_data.keys()
keys = [int(i) for i in keys]

keys = sorted(keys)[0::]
print keys

for m in sorted(keys):
    m = str(m)
    
    x = [i for i,j,k in PL_data[m]]
    x = wavetoeV(x)
    y = [j for i,j,k in PL_data[m]]
    #y = np.array(y)/max(y)
    
    maxy.append(max(y))
    maxx.append(x[list(y).index(max(y))])
    plt.plot(x,y,label = str(m), linewidth = 1.0)

plt.ylabel('PL', size= 12)
plt.xlabel('Energy / eV', size= 12
)
#plt.plot(maxx, maxy, 'x')

#plt.legend()
plt.show()

   
"""
#Set the temperatures and wavelengths available
temps = [i for i,j in files]
interval = [1.4,1.75]
tem = temps[10:11]
tol = 50
step = 0.005
    
a = peaks_FWHM_plot(PL_data,temps,interval,temps[8:9],tol,step)
err1 = FWHMErr(a[3])
err2 = peakEnergyError(a[3])

plt.figure()
plt.title('MAPI Peak Width')
plt.errorbar(a[0],a[1],yerr = err1, fmt='o')
plt.xlabel('Temperature/ K')
plt.ylabel('FWHM/ eV')
plt.show()
"""




