import numpy as np 
import Exeter_CFD_Problems as TestSuite
import os
import shutil
import subprocess


settings = {}
settings['source_case'] = 'Exeter_CFD_Problems/data/PitzDaily/case_fine/' # source directory
settings['case_path'] = 'Exeter_CFD_Problems/data/PitzDaily/case_single/' # case path where CFD simulations will run
settings['boundary_files'] = ['Exeter_CFD_Problems/data/PitzDaily/boundary.csv']
settings['fixed_points_files'] = ['Exeter_CFD_Problems/data/PitzDaily/fixed.csv']
settings['n_control'] = [1] # you should be able to use any positive integer.

prob = TestSuite.PitzDaily(settings)
lb, ub = prob.get_decision_boundary() # lb = lower bounds, ub = upper bounds

boundaries_vels_dict = {}
total_sims = 30
sim_number = 0
features_array = np.empty((total_sims, 2))
targets_array = np.empty((total_sims))
# print(features_array)
while sim_number < total_sims:

    is_valid = False

    while is_valid == False:
        x = np.random.random((1, lb.shape[0])) * (ub-lb) + lb
        is_valid = prob.constraint(x)
    print(x)
        
    try:
        res = prob.evaluate(x, verbose=True)
    except TypeError:
        pass
    # move the sampleDict file to the case_single folder
    os.system("cp sampleDict Exeter_CFD_Problems/data/PitzDaily/case_single/system/")
    os.system("(cd Exeter_CFD_Problems/data/PitzDaily/case_single/; postProcess -func sampleDict)")

    # os.system("cp sampleDict Exeter_CFD_Problems/data/PitzDaily/case_single/system/")
    # os.system("cd Exeter_CFD_Problems/data/PitzDaily/case_single/; )")

    outlet_pressure = np.loadtxt('Exeter_CFD_Problems/data/PitzDaily/case_single/postProcessing/sampleDict/0/outlet_p.xy', dtype=float,usecols = 1)
    inlet_pressure = np.loadtxt('Exeter_CFD_Problems/data/PitzDaily/case_single/postProcessing/sampleDict/0/inlet_p.xy', dtype=float,usecols = 1)
    try:
        outlet_pressure_avg = np.average(outlet_pressure) 
        inlet_pressure_avg = np.average(inlet_pressure)
        delta_p = abs(outlet_pressure_avg - inlet_pressure_avg)
        print('outlet_p =', outlet_pressure_avg, 'inlet_p =', inlet_pressure_avg, 'delta_p =', delta_p)
        features_array[sim_number] = x
        targets_array[sim_number] = delta_p
        sim_number += 1
        print('simulation number =', sim_number)
    except ValueError:
        print('value error')
        pass


print(features_array)
print(targets_array)

np.savetxt('random_geometries.txt', features_array)
np.savetxt('random_pressures.txt', targets_array)

