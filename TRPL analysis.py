import numpy as np
import glob
import re
from testsdt import *
import matplotlib.pyplot as plt
import math
from scipy.optimize import curve_fit



def openFile():
    file_data = glob.glob('./*.sdt')
    bg_file = [i for i in file_data if 'bg' in i]
    file_data = [i for i in file_data if 'bg' not in i]
    file_data = [i for i in file_data if 'irf' not in i]
    
    temperatures = []
    wavelengths = []
    
    files = [re.split('-|_|',file_) for file_ in file_data]
    for labels in files:
        for label in labels:
            if 'k' in label:
                if '.sdt' not in label:
                    try:
                        temperatures.append(int(label.replace("k", "")))

                    except ValueError:
                        pass
                else:
                    temperatures.append(int(label.replace("k.sdt", "")))
                    
            if 'nm.sdt' in label:
                wavelengths.append(int(label.replace("nm.sdt", "")))

    if len(wavelengths) != 0:

        new_labels = zip(wavelengths,temperatures,file_data)
    
    
    
        d = {}

    
        for i in new_labels:
            if i[0] not in d.keys():
                d[i[0]] = [(i[1],i[2])]
            else:
                d[i[0]].append((i[1],i[2]))
                
    else:
        
        d = {'mapi' : [(temperatures[0], file_data[0])] }
        
        for i in np.arange(1,len(temperatures)):
            d['mapi'].append((temperatures[i],file_data[i]))
            
        
    return d, bg_file

    
def dataExtraction(wavelength):
    data,bg= openFile()
    keys = data.keys()

    
    new_data = {}
    new_bg = []
    
    if len(keys) != 1:
        for i in range(len(data[wavelength])):
            new_data[data[wavelength][i][0]] = SdtFile(data[wavelength][i][1]).data[0][0]
        
        for j in range(len(bg)):
            new_bg.append(SdtFile(bg[j]).data[0][0])
                   
    else:
        for i in range(len(data[keys[0]])):
            new_data[data[keys[0]][i][0]] = SdtFile(data[keys[0]][i][1]).data[0][0]
            
        for j in range(len(bg)):
            new_bg.append(SdtFile(bg[j]).data[0][0])
          
    return new_data, new_bg
    
    
def IRF():
    sdt = SdtFile("irf.sdt").data[0][0]
    return sdt
    
    
def removeBackground(TRPL_data):
    for key in TRPL_data.keys():
        background = np.mean(TRPL_data[key][110:170])
        TRPL_data[key] = [(i - background) for i in TRPL_data[key]]
    return TRPL_data
    
def decayIndex(TRPL_data):
    start_index = []

    keys = TRPL_data.keys()
    for i in range(len(keys)):
        start_index.append(list(TRPL_data[keys[i]]).index(max(TRPL_data[keys[i]])))
    
    return start_index
    
    
def irfPlacement(start_index,TRPL_data):
    irfxi = np.arange(0,500,500/1024.)
    irfyi = IRF()
    
    keys = TRPL_data.keys()

    irfx = []
    irfy = []
    irf_index = 0
    
    for i in range(len(keys)):
        ind = list(irfyi).index(max(irfyi))
        irf_index = ind
        irfx.append(irfxi)
        irfy.append(irfyi/float(max(irfyi))*max(TRPL_data[keys[i]]))
        
    return irfx, irfy, irf_index
    
    
    
def bi_exp(t,amp1, tau1,amp2,tau2):
    return  amp1*np.exp(-t/float(tau1))  + amp2*np.exp(-t/float(tau2))
      


    
def minimise4D(TRPLi_data,temp, x, variables, start, steps, irfyi, irf_index, TRPL_index):
    amp1 = np.arange(start[0],variables[0],steps[0]) #sort this later
    tau1 = np.arange(start[1],variables[1],steps[1])
    amp2 = np.arange(start[2],variables[2],steps[2])
    tau2 = np.arange(start[3],variables[3],steps[3])

    end = 500

    
    resid = []
    
    
    
    for i in range(amp1.size):
        print str(round(i*100/float(amp1.size),1)) + ' % Completed'
        for j in range(tau1.size):
            for k in range(amp2.size):
                for l in range(tau2.size):
                    expon = bi_exp(x,amp1[i],tau1[j],amp2[k],tau2[l])
                    convolution = np.convolve(irfyi,expon)
            
                    max_convolve = list(convolution).index(max(convolution))
            
                    TRPL = TRPLi_data[TRPL_index:TRPL_index+end]
                    convo = convolution[max_convolve:max_convolve+end]
            
                    residual = np.sum((convo-TRPL)*(convo-TRPL))
    
                    resid.append((residual,i,j,k,l))
            
    plt.figure()
    
    chis = [i for i,j,k,l,m in resid]
    least_index = chis.index(min(chis))
    final_residual = resid[least_index]
    fit_error = np.sqrt(min(chis)/(end-2))
    convolution = np.convolve(irfyi, bi_exp(x,amp1[final_residual[1]],tau1[final_residual[2]],amp2[final_residual[3]],tau2[final_residual[4]]))
    max_convolve = list(convolution).index(max(convolution))
    
    plt.title(str(temp)+"K")
    plt.plot(TRPLi_data[TRPL_index:TRPL_index+end], '.')
    plt.plot(convolution[max_convolve:max_convolve+end], label = "Amp1: " + str(amp1[final_residual[1]])+" \n Tau1: " + str(tau1[final_residual[2]])+" \n Amp2: " + str(amp2[final_residual[3]])+" \n Tau2: " + str(tau2[final_residual[4]]))
    
    plt.xlabel('Time/ ns')
    plt.ylabel('Counts')
    print "Amplitude1: " + str(amp1[final_residual[1]])+"\n Tau1: " + str(tau1[final_residual[2]])+"\n Amplitude2: " + str(amp2[final_residual[3]])+"\n Tau2: " + str(tau2[final_residual[4]])
    
    plt.legend()
    plt.show()
    
    return final_residual, fit_error
    
    
def mono_exp(t, amp, tau):
    return amp*np.exp(-t/float(tau))
    
def minimise2D(TRPLi_data,temp, x, variables, start, steps, irfyi, irf_index, TRPL_index):
    amp = np.arange(start[0],variables[0],steps[0])
    tau = np.arange(start[1],variables[1],steps[1])
    
    step  = 500/1024.
    end = 700
    
    x = np.arange(0,step*end,step)
    
    resid = []
    
    for i in range(amp.size):
        print str(round(i*100/float(amp.size),1)) + ' % Completed'
        for j in range(tau.size):
            expon = mono_exp(x,amp[i],tau[j])
            convolution = np.convolve(irfyi,expon)
            
            max_convolve = list(convolution).index(max(convolution))
            
            TRPL = TRPLi_data[TRPL_index:TRPL_index+end]
            convo = convolution[max_convolve:max_convolve+end]
            
            
            
            residual = np.sum((convo-TRPL)*(convo-TRPL))
            resid.append((residual,i,j))
            
    
    plt.figure()
    
    
    chis = [i for i,j,k in resid]
    least_index = chis.index(min(chis))
    fit_error = np.sqrt(min(chis)/(end-2))
    final_residual = resid[least_index]
    convolution = np.convolve(irfyi, mono_exp(x,amp[final_residual[1]],tau[final_residual[2]]))
    max_convolve = list(convolution).index(max(convolution))
    
    plt.title(str(temp)+"K")
    plt.plot(TRPLi_data[TRPL_index:TRPL_index+end],'.')
    plt.plot(convolution[max_convolve:max_convolve+end], label = "Amp: " + str(amp[final_residual[1]])+" \n Tau: " + str(tau[final_residual[2]]))
     
            
    plt.xlabel('Time/ ns')
    plt.ylabel('Counts')
    print "Amplitude1: " + str(amp[final_residual[1]])+"\n Tau1: " + str(tau[final_residual[2]])
    
    plt.legend()
    plt.show()
           
    return final_residual,fit_error
    
#LIFETIMES ARE GIVEN IN NATURAL UNITS. CONVERT TO REAL UNITS BY MULTIPLYING BY STEP SIZE

dats,bg=  dataExtraction(620)
dats = removeBackground(dats)
step = 500/1024.
index = decayIndex(dats)
keys = dats.keys()

"""
#Monoexp fitting
x = np.arange(0,500,step)
irfx,irfy,irf_index = irfPlacement(index,dats)
mini = minimise2D(dats[70],keys[0],x, [0.1,300], [0.00000001,150], [0.001,0.1],irfy[0],irf_index,index[0])

#plt.plot(bg[0],'.')
plt.ylim(0)
plt.show()
"""


# Biexponential Fitting

x = np.arange(0,500,step)
keys = dats.keys()
irfx,irfy,irf_index = irfPlacement(index,dats)
    
mini = minimise4D(dats[150],keys[4],x, [0.1,1,1.0,1], [0.000001,0.000001,0.000001,0.000001], [0.01,0.1,0.1,0.1],irfy[0],irf_index,index[0])

# [1.0,100,1.0,100], [0.000001,0.000001,0.000001,0.000001], [0.1,10,0.1,10]





