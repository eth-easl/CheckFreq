
### For running the script:
### Download data from: https://github.com/kadupitiya/goog-preemption-data
### Place the script inside the data folder
### set mtype, zone accordingly

import json
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

def func(x, A, b, t1, t2):
    return A * (1 - np.exp(-x/t1) + np.exp((x-b)/t2))

with open('data.json') as json_file:
    all_gcp_data = json.load(json_file)


mtype = 'n1-highcpu-32'
zone = 'us-west1-a'

times = []
for x in all_gcp_data:
    d = all_gcp_data[x]['instance_data']
    if (d['ZONE'] == zone and d['MACHINE_TYPE']==mtype):
        times.append(all_gcp_data[x]['time_in_sec']/3600)

times.sort()
print(len(times), " data were found!")
if (len(times) != 0):

    # getting data of the histogram
    count, bins_count = np.histogram(times, bins=5000)
    
    # finding the PDF of the histogram using count values
    pdf = count / sum(count)
    # using numpy np.cumsum to calculate the CDF
    # We can also find using the PDF values by looping and adding
    cdf = np.cumsum(pdf)
    plt.plot(bins_count[1:], cdf, label="data")

    ## fitting
    popt, pcov = curve_fit(func, bins_count[1:], cdf, bounds=([0., 23., 0.5, 0.7], [1., 24., 1.5, 0.9]))
    print(popt,)

    plt.plot(bins_count[1:], func(bins_count[1:], *popt), label="fit")
    plt.legend()
    plt.xlabel('Time to preempt')
    plt.ylabel('CDF')
    plt.savefig('fig2.png')

