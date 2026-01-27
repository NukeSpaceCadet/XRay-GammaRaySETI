#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 25 13:00:59 2026

@author: enrico-fermi
"""
from OpenMC_detector_with_background import spectrum_maker
import numpy as np
import matplotlib.pyplot as plt

ones_spectrum, ones_errors, lower_bounds, ones_energies = spectrum_maker(100, 0.07, 2**12, "Jan25_2025",100000,"1's")
zeros_spectrum, zeros_errors, lower_bounds, zero_energies = spectrum_maker(100, 0.07, 2**12, "Jan25_2025",0,"1's")

# Plot both lists
plt.plot(ones_spectrum, label='100 keV peak (1s)', color='blue')
plt.plot(zeros_spectrum, label='Background (0s)', color='red')

# Add legend and labels
plt.xlabel('Detector bin energy (keV)')
plt.ylabel('Normalized counts')
# plt.title('Overlayed Lists')
plt.legend()

# Show plot
plt.show()