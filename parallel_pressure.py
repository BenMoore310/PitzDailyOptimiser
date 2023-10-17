import multiprocessing
from multiprocessing import Pool
import subprocess
from typing import Dict
import Exeter_CFD_Problems as TestSuite
import shutil
import tempfile
import os
import numpy as np

n_iters = 100

# n_procs = multiprocessing.cpu_count()
n_procs = 2

features_array = np.empty((n_iters, 2))
targets_array = np.empty((n_iters))

def postprocess_result(settings):
    os.mkdir(os.path.join(settings["case_dir"], '/postProcessing/sampleDict/0'))
    os.mkdir(os.path.join(settings["case_dir"], '/system'))


    shutil.copy('/data/run/week_1_testing/PitzDailyOptimiser/sampleDict', settings["case_dir"] + '/system/sampleDict')


    subprocess.run(["postProcess -func sampleDict"], shell=True, cwd=settings["case_dir"])
    

    # os.system("cp sampleDict Exeter_CFD_Problems/data/PitzDaily/case_single/system/")
    # os.system("cd Exeter_CFD_Problems/data/PitzDaily/case_single/; )")

    outlet_pressure = np.loadtxt(os.path.join(settings["case_dir"], 'postProcessing/sampleDict/0/outlet_p.xy'), dtype=float,usecols = 1)
    inlet_pressure = np.loadtxt(os.path.join(settings["case_dir"], 'postProcessing/sampleDict/0/inlet_p.xy'), dtype=float,usecols = 1)

    try:
        outlet_pressure_avg = np.average(outlet_pressure) 
        inlet_pressure_avg = np.average(inlet_pressure)
        delta_p = abs(outlet_pressure_avg - inlet_pressure_avg)
        print('outlet_p =', outlet_pressure_avg, 'inlet_p =', inlet_pressure_avg, 'delta_p =', delta_p)

    except ValueError:
        print('value error')
        delta_p = np.nan

    return delta_p

def initialise_settings_and_run_dir():
     # run the openfoam calculation on the new samples
    # define the base settings
    settings = {}
    settings['source_case'] = 'Exeter_CFD_Problems/data/PitzDaily/case_fine/' # source directory
    settings['case_path'] = 'Exeter_CFD_Problems/data/PitzDaily/case_single/' # case path where CFD simulations will run
    settings['boundary_files'] = ['Exeter_CFD_Problems/data/PitzDaily/boundary.csv']
    settings['fixed_points_files'] = ['Exeter_CFD_Problems/data/PitzDaily/fixed.csv']
    settings['n_control'] = [1] # you should be able to use any 
    
    # tmpdir = tempfile.mkdtemp()

    # if not os.path.exists(tmpdir):
    #     os.makedirs(tmpdir)
    tmpdir = tempfile.mkdtemp()
    # create a tmpdir to run the job in for each sample
    # with tempfile.mkdtemp() as tmpdir:

        # if os.path.exists(tmpdir):
        #     shutil.rmtree(tmpdir)
        # copy the contents of case_path to the tmpdir
    shutil.copytree(settings['case_path'], tmpdir)
    # update the case_path to the tmpdir
    settings['case_path'] = tmpdir

    return settings

def generate_random_sample(args):

    settings = args

    prob = TestSuite.PitzDaily(settings)
    lb, ub = prob.get_decision_boundary() # lb = lower bounds, ub = upper bounds
    
    is_valid = False

    while is_valid == False:
        generated_vector = np.random.random((1, lb.shape[0])) * (ub-lb) + lb
        is_valid = prob.constraint(decision_vector)
    print(decision_vector)

    return generated_vector

def run_openfoam_calculation(args) -> float:

    # unpack the arguments
    settings = args
    prob = TestSuite.PitzDaily(settings)
    # run the openfoam calculation
    # return the output from the black box function
    # prob = TestSuite.PitzDaily(settings)
    # lb, ub = prob.get_decision_boundary() # lb = lower bounds, ub = upper bounds
    
    # is_valid = False

    # while is_valid == False:
    #     decision_vector = np.random.random((1, lb.shape[0])) * (ub-lb) + lb
    #     is_valid = prob.constraint(decision_vector)
    # print(decision_vector)
        
    try:
        res = prob.evaluate(decision_vector, verbose=True)
    except TypeError:
        pass
    # result = prob.evaluate(decision_vector)

    pressure = postprocess_result(settings)

    return pressure, decision_vector 

for _ in range(n_iters):

    # fit initial model
    # model = GaussianProcessRegressor()
    # model.fit(features_array, targets_array)

    # query the model with the acquisition strategy
    # new_samples = acquisition(model, decision_vectors, 10, mode='ei')


    settings = [initialise_settings_and_run_dir() for _ in range(n_procs)]

    generator_args = [settings[i] for i in range(n_procs)] #used when generating new samples each time

    with Pool(n_procs) as p:
        new_samples = generate_random_sample(generator_args)




    simulation_args = [(settings[i], new_samples[i]) for i in range(n_procs)] #also passes new samples



    with Pool(n_procs) as p:
        pressure, decision_vector = p.map(run_openfoam_calculation, simulation_args)
    
    print(pressure, decision_vector)
    

    # add the new samples to the training data
    # features_array = np.vstack((features_array, new_samples)) #will this add the right thing? ie the 10 parameters, not just an index?
    # targets_array = np.vstack((targets_array, results))