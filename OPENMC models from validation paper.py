"""
Mär 12, 2022
Christopher Fichtlscherer (fichtlscherer@mailbox.org)
GNU General Public License

A structural search for the differences in the pht values
between OpenMC and MCNP.
"""

import numpy as np
import openmc
import datetime as dt
import matplotlib.pyplot as plt


def run_openmc(parameter_dic):

    alu = openmc.Material()         
    alu.add_nuclide('Al27', 1)      
    alu.temperature = 293           
    alu.set_density('g/cm3', 2.7)   

    sodium_iodide = openmc.Material()
    sodium_iodide.add_element('Na', 1)
    sodium_iodide.add_element('I', 1)
    sodium_iodide.temperature = 293
    sodium_iodide.set_density('g/cm3', parameter_dic["density"])
    
    oxide = openmc.Material()                                                                       
    oxide.add_nuclide('O16', 0.4) # same as MCNP atom or weight fraction                            
    oxide.add_element('Al', 0.6)                                                                    
    oxide.temperature = 293                                                                         
    oxide.set_density('g/cm3', 3.97)                                                                

    air = openmc.Material()                               
    air.add_nuclide('N14', 4)                               
    air.add_nuclide('O16', 1)                             
    air.temperature = 293                                
    air.set_density('g/cm3', 0.001225)    

    materials = openmc.Materials([alu, sodium_iodide, oxide, air])
    materials.cross_sections = parameter_dic["cross_sections_openmc"]

    materials.export_to_xml()
    
    c1 = openmc.ZCylinder(r=2.54)                                                                  
    c2 = openmc.ZCylinder(r=2.70)                                                                  
    c3 = openmc.ZCylinder(r=2.84)                                                                  
                                                                                                   
    z1 = openmc.ZPlane(z0=0.00)                                                                    
    z2 = openmc.ZPlane(z0=0.10)                                                                    
    z3 = openmc.ZPlane(z0=5.18)                                                                    
    z4 = openmc.ZPlane(z0=7.18)                                                                    
                                                                                                   
    s = openmc.Sphere(r=10, boundary_type='vacuum')                                                
                                                                                                   
    crystal = openmc.Cell(cell_id=1)                                                          
    crystal.region = -c1 & -z3 & +z2                                                               
    crystal.fill = sodium_iodide                                                                   
                                                                                                   
    oxide_layer = openmc.Cell(cell_id=2)                                                                    
    oxide_layer.region = +c1 & -c2 &-z3 & +z2                                                      
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
    sourrounding.fill = air                                                                        
                                                                                                   
    cell_list = [crystal, oxide_layer, alu_casing, alu_window, alu_back, sourrounding]             
                                                                                                   
    universe = openmc.Universe(cells=cell_list)                                                    
    geometry = openmc.Geometry(universe)                                                           
    geometry.export_to_xml()                                                                       

    settings = openmc.Settings()
    settings.particles = int(parameter_dic["number_particles"])
    settings.batches = 1
    settings.photon_transport = True

    source = openmc.Source()
    source.library = 'custom_source/build/libsource.so' 
    source.parameters = "radius=" + str(2.84) + ',energy=' + str(parameter_dic["energy"]) 
    settings.source = source

    settings.run_mode = 'fixed source'
    settings.export_to_xml()

    tallies = openmc.Tallies()
    cell_filter = openmc.CellFilter(crystal)

    # OpenMC - MCNP -> different handling of bin boarders -> shift openmc bins by epsilon
    energy_bins = parameter_dic["bins"][:2] + [i + 1e-6 for i in parameter_dic["bins"][2:]]
    energy_filter = openmc.EnergyFilter(energy_bins)

    tally = openmc.Tally(name='pht tally')
    tally.filters = [cell_filter, energy_filter]
    tally.scores = ['pulse-height']
    tallies.append(tally)
    tallies.export_to_xml()
    
    openmc.run(openmc_exec='/home/cpf/Desktop/openmc/build/bin/openmc', threads=1)
