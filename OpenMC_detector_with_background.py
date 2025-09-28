# -*- coding: utf-8 -*-
"""
Created on Wed Mar 26 18:44:49 2025

@author: Lucas Beveridge

Gamma-ray SETI detector simulation with some materials and geometry from 
OpenMC pulse-height tally validation model written by 
Christopher Fichtlscherer (fichtlscherer@mailbox.org)

References:
"Modeling gamma detectors in OpenMC: Validation of a newly implemented
    pulse-height tally," Christopher Fichtlscherer, Milon Miah,
    Friederike Frieß, Malte Göttsche, Moritz Kütt, 
    Progress in Nuclear Energy 172 (2024) 105186

Background for a Gamma-Ray Satellite on a Low-Earth Orbit
    Summary: This study models various background components in LEO, including gamma-ray emissions resulting from interactions with cosmic rays and Earth's atmosphere.​
    Link: https://arxiv.org/abs/1902.06944

Instrumental Background in Gamma-Ray Spectrometers Flown in Low Earth Orbit
    Summary: This research presents techniques for calculating the instrumental continuum background in gamma-ray spectrometers operating in LEO, highlighting contributions from cosmic diffuse radiation and Earth's albedo glow.​
    Link: https://www.sciencedirect.com/science/article/pii/016890029290832O

Orbits and Background of Gamma-Ray Space Instruments
    Summary: This paper discusses the challenges faced by gamma-ray space missions in LEO, particularly the need to avoid the Van Allen radiation belts and manage background noise from charged particles and photons.​
    SpringerLink+1arXiv+1
    Link: https://arxiv.org/pdf/2209.07316

Gamma Rays - NASA Science
    Summary: This NASA resource provides an overview of gamma rays, their sources, and their detection, including information relevant to space-based observations.​
    Link: https://science.nasa.gov/ems/12_gammarays/

Particle Background for Equatorial Low Earth Orbit (ELEO)
    Summary: This document examines the particle background for unshielded detectors in LEO, focusing on cosmic diffuse radiation and Earth's albedo glow resulting from cosmic ray interactions with the atmosphere.​
    agn.caltech.edu
    Link: https://agn.caltech.edu/~srk/XC/Homeworks/Particle_background_for_Low_Earth_Orbit__LEO_.pdf

"""
# import openmc
# import numpy as np
# import matplotlib.pyplot as plt
# import os
# import shutil
import time

time1 = time.time()

def spectrum_maker(peak_energy, FWHM, bins, time_index, pulse_count, name):
    
    import openmc
    import numpy as np
    # import matplotlib.pyplot as plt
    import os
    import shutil
    import gc
    
    print("\n=== New spectrum_maker run ===")
    print("GC tracked objects:", len(gc.get_objects()))
    
    # bins = number of MCA channels
    batches = 120
    particles = 10000 # number of particles for run
    
    # Materials
    alu = openmc.Material()         
    alu.add_nuclide('Al27', 1)      
    alu.temperature = 293           
    alu.set_density('g/cm3', 2.7)   
    
    sodium_iodide = openmc.Material()
    sodium_iodide.add_element('Na', 1)
    sodium_iodide.add_element('I', 1)
    sodium_iodide.temperature = 293
    sodium_iodide.set_density('g/cm3', 3.67)
    
    oxide = openmc.Material()                                                                       
    oxide.add_nuclide('O16', 0.4)                       
    oxide.add_element('Al', 0.6)                                                                    
    oxide.temperature = 293                                                                         
    oxide.set_density('g/cm3', 3.97)                                                                  
    
    materials = openmc.Materials([alu, sodium_iodide, oxide])
    
    materials.export_to_xml()
    
    # Geometry
    c1 = openmc.ZCylinder(r=2.54)                                                                  
    c2 = openmc.ZCylinder(r=2.70)                                                                  
    c3 = openmc.ZCylinder(r=2.84)                                                                  
                                                                                                   
    z1 = openmc.ZPlane(z0=0.00)                                                                    
    z2 = openmc.ZPlane(z0=0.10)                                                                    
    z3 = openmc.ZPlane(z0=5.18)                                                                    
    z4 = openmc.ZPlane(z0=7.18)                                                                    
                                                                                                   
    s = openmc.Sphere(r=20, boundary_type='vacuum')                                                
                                                                                                   
    crystal = openmc.Cell(cell_id=1)                                                          
    crystal.region = -c1 & -z3 & +z2                                                               
    crystal.fill = sodium_iodide                                                                   
                                                                                                   
    oxide_layer = openmc.Cell(cell_id=2)                                                                    
    oxide_layer.region = +c1 & -c2 & -z3 & +z2                                                      
    oxide_layer.fill = oxide                                                                       
                                                                                                   
    alu_casing = openmc.Cell(cell_id=3)                                                                     
    alu_casing.region = +c2 & -c3 & -z4 & +z1                                                      
    alu_casing.fill = alu                                                                          
                                                                                                   
    alu_window = openmc.Cell(cell_id=4)                                                                     
    alu_window.region = -c2 & -z2 & +z1                                                            
    alu_window.fill = alu                                                                          
                                                                                                   
    alu_back = openmc.Cell(cell_id=5)                                                                       
    alu_back.region = -c2 & -z4 & +z3                                                              
    alu_back.fill = alu                                                                            
                                                                                                   
    sourrounding = openmc.Cell(cell_id=6)                                                                   
    sourrounding.region = -s & ~(-c3 & -z4 & +z1)                                                  
    sourrounding.fill = None                                                                        
                                                                                                   
    cell_list = [crystal, oxide_layer, alu_casing, alu_window, alu_back, sourrounding]             
                                                                                                   
    universe = openmc.Universe(cells=cell_list)                                                    
    geometry = openmc.Geometry(universe)                                                           
    geometry.export_to_xml()
    
    # Settings
    settings = openmc.Settings()
    settings.particles = particles
    settings.batches = batches
    settings.inactive = 20
    settings.photon_transport = True
    
    # Create a flat disc source beaming towards the detector to simulate a distant 
    # Define the point source position (example)
    position = openmc.stats.Point((0.0, 0.0, -10))
    
    # Direction (towards positive z-axis)
    angle = openmc.stats.Monodirectional((0.0, 0.0, 1.0))
    
    # Define primary gamma peak (from your source) and typical LEO background lines
    gamma_peaks = [
        (peak_energy*1e3, (pulse_count/particles)*particles),   # Primary gamma peak at specified energy in keV (normalized to 1.0)
        (1.0e6, 0.2*particles),    # Cosmic background peak around 1 MeV
        (1.46e5, 0.05*particles),  # K-40 peak at 146 keV (trace amounts)
        (3.00e5, 0.1*particles),   # Earth's albedo gamma peak at 300 keV
        (6.00e5, 0.1*particles),   # Earth's albedo gamma peak at 600 keV
    ]
    
    # Add a cosmic ray continuum (broad Gaussian distribution)
    cosmic_energy = np.linspace(1e5, 3e6, 50)  # 100 keV to 3 MeV
    cosmic_intensity = np.exp(-0.5 * ((cosmic_energy - 1e6) / 7e5) ** 2) * 0.02  # Gaussian shape
    
    # Add the cosmic continuum to the peak list
    for e, i in zip(cosmic_energy, cosmic_intensity):
        gamma_peaks.append((e, i))
    
    # Apply Gaussian broadening to each peak (simulate NaI resolution ~7% at 662 keV)
    energy_values = []
    probability_values = []
    
    for energy, intensity in gamma_peaks:
        resolution = FWHM * energy  # FWHM = 7% of peak energy for NaI
        broadened_energies = np.random.normal(energy, resolution, 100)
        energy_values.extend(broadened_energies)
        probability_values.extend([intensity / len(broadened_energies)] * len(broadened_energies))
    
    # Normalize the probabilities
    total_prob = sum(probability_values)
    probability_values = [p / total_prob for p in probability_values]
    
    # Create the energy distribution
    energy_dist = openmc.stats.Discrete(energy_values, probability_values)
    
    # Create the source object
    source = openmc.Source(space=position, angle=angle, energy=energy_dist, particle='photon')
    
    settings.run_mode = 'fixed source'
    settings.source = source
    settings.export_to_xml()
    
    # Pulse height tally for simulating NaI detector
    tallies = openmc.Tallies()
    energy_bins = np.linspace(0,5e6, num=bins)
    energy_filter = openmc.EnergyFilter(energy_bins)
    cell_filter = openmc.CellFilter(crystal)
    
    tally = openmc.Tally(name='pulse-height tally')
    tally.filters = [cell_filter, energy_filter]
    tally.scores = ['pulse-height']
    tallies.append(tally)
    tallies.export_to_xml()
    
    openmc.run()
    
    # noise = [0.25e-7+1e-6*int.from_bytes(os.urandom(8), byteorder="big") / ((1 << 64) - 1) for i in range(bins-1)] # generate background noise to better approximate detector signal
    
    # Load the statepoint file
    sp = openmc.StatePoint("statepoint."+str(batches)+".h5")  # Adjust to your file name
    
    # Get the tally by ID or name
    tally = sp.get_tally()
    
    # Get energy bins (edges) from the tally filter
    energy_filter = tally.find_filter(openmc.EnergyFilter)
    edges = energy_filter.bins  # Shape (49, 2)
    
    # Extract the lower bounds from the first column
    lower_bounds = edges[:, 0]  # Shape (49,)
    
    # Access the tally results (mean values and standard deviation)
    spectrum = tally.mean.flatten()  # Shape (49,), multiply by number of photons in main signal peak during one interval (pileup)
    errors = tally.std_dev.flatten()  # Shape (49,)
    
    # Move Statepoint and Summary files to new directory to archive them
    dir_name = "spectrum_"+str(peak_energy)+"keV_index:_"+str(time_index)+"_"+name
    
    # Create the new directory in the current working directory
    os.makedirs(dir_name, exist_ok=True)
    
    # Specify the file names you want to move
    file1 = "statepoint."+str(batches)+".h5"
    file2 = "summary.h5"
    
    # Move the files into the new directory
    shutil.move(file1, os.path.join(dir_name, file1))
    shutil.move(file2, os.path.join(dir_name, file2))
    
    openmc.reset_auto_ids() # reset all OpenMC ID's
    del tallies
    del settings
    del geometry
    del materials
    
    print(f"Moved {file1} and {file2} to directory {dir_name}.")
    
    gc.collect()
    print("After GC:", len(gc.get_objects()))
    
    return spectrum, errors, lower_bounds, edges

# spectrum, errors, lower_bounds, energy_bin_edges = spectrum_maker(414e3, 0.07, 2**12, 2, 100, "test")
# time2 = time.time()
# print("total run time (hours): ", (time2-time1)/3600)
# # Plot the spectrum using the lower bounds
# plt.figure(figsize=(8, 5))
# plt.step(lower_bounds, spectrum, where='mid', label='Photon Spectrum', color='blue')
# plt.fill_between(lower_bounds, spectrum - errors, spectrum + errors, 
#                  step='mid', color='lightblue', alpha=0.6, label='Uncertainty')

# plt.xlabel('Energy (eV)')
# plt.ylabel('Flux (per source particle)')
# plt.title('Gamma-ray Spectrum')
# # plt.xscale('log')  # Log scale for energy
# plt.yscale('log')  # Log scale for the flux
# plt.legend()
# plt.grid(True, which='both', ls='--', alpha=0.5)
# plt.show()