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

n_control = 1

settings = {}
settings['source_case'] = '/data/run/week_1_testing/PitzDailyOptimiser/Exeter_CFD_Problems/data/PitzDaily/case_fine/' # source directory
settings['case_path'] = '/data/run/week_1_testing/PitzDailyOptimiser/Exeter_CFD_Problems/data/PitzDaily/case_single/' # case path where CFD simulations will run
settings['boundary_files'] = ['/data/run/week_1_testing/PitzDailyOptimiser/Exeter_CFD_Problems/data/PitzDaily/boundary.csv']
settings['fixed_points_files'] = ['/data/run/week_1_testing/PitzDailyOptimiser/Exeter_CFD_Problems/data/PitzDaily/fixed.csv']
settings['n_control'] = [n_control] # you should be able to use any 

# /Users/benmoore/openfoam-data/run/week_1_testing/PitzDailyOptimiser/Exeter_CFD_Problems

# 0       PyFoamHistory                    PyFoamRunner.blockMesh.logfile   PyFoamRunner.checkMesh.logfile  PyFoamState.LastOutputSeen  PyFoamState.StartedAt  case_single.foam  pitzDaily_backup
# 0_orig  PyFoamRunner.blockMesh.analyzed  PyFoamRunner.checkMesh.analyzed  PyFoamState.CurrentTime         PyFoamState.LogDir          PyFoamState.TheState   constant          system



# 0				PyFoamRunner.blockMesh.logfile	PyFoamState.LastOutputSeen	pitzDaily_backup
# 0_orig				PyFoamRunner.checkMesh.analyzed	PyFoamState.LogDir		postProcessing
# Pressure.analyzed		PyFoamRunner.checkMesh.logfile	PyFoamState.StartedAt		system
# Pressure.logfile		PyFoamSolve.analyzed		PyFoamState.TheState
# PyFoamHistory			PyFoamSolve.logfile		case_single.foam
# PyFoamRunner.blockMesh.analyzed	PyFoamState.CurrentTime		constant


def get_samples(n_samples, n_control):
    print(settings)
    prob = TestSuite.PitzDaily(settings)
    lb, ub = prob.get_decision_boundary()

    random_samples = np.empty((n_samples, n_control*2))
    for i in range(0, n_samples):
        is_valid = False

        while is_valid == False:
            x = np.random.random((1, lb.shape[0])) * (ub-lb) + lb
            is_valid = prob.constraint(x)
        random_samples[i] = x
    return(random_samples)   


def simulate(decision_vector):
    
    tmpdir = tempfile.mkdtemp(dir='/data/run/week_1_testing/PitzDailyOptimiser/')

    # if os.path.exists(tmpdir):
    #     shutil.rmtree(tmpdir)
    
    shutil.copytree(settings['case_path'], tmpdir, dirs_exist_ok=True)
    settings['case_path'] = tmpdir

    print(settings)

    try:
        prob = TestSuite.PitzDaily(settings)
        res = prob.evaluate(decision_vector, verbose=True)
    except TypeError as e:
        # print("Error in evaluate function", e)
        print("Simulation finished with output", e)

    pressure = postprocess_result(settings)
    return pressure



def postprocess_result(settings):
    print(settings)
    # os.mkdir(os.path.join(settings["case_path"], '/postProcessing/sampleDict/0'))
    # os.mkdir(os.path.join(settings["case_path"], '/system'))

    shutil.copy('/data/run/week_1_testing/PitzDailyOptimiser/sampleDict', settings["case_path"] + '/system/')
    print("Current Working Directory:", os.getcwd())

    subprocess.run(["postProcess -func sampleDict"], shell=True, cwd=settings["case_path"], check=True, capture_output=True)
    # print(os.listdir(os.path.join(settings["case_path"], 'postProcessing/sampleDict/0')))

    outlet_pressure = np.loadtxt(os.path.join(settings["case_path"], 'postProcessing/sampleDict/0/outlet_p.xy'), dtype=float,usecols = 1)
    inlet_pressure = np.loadtxt(os.path.join(settings["case_path"], 'postProcessing/sampleDict/0/inlet_p.xy'), dtype=float,usecols = 1)

    try:
        outlet_pressure_avg = np.average(outlet_pressure) 
        inlet_pressure_avg = np.average(inlet_pressure)
        delta_p = abs(outlet_pressure_avg - inlet_pressure_avg)
        print('outlet_p =', outlet_pressure_avg, 'inlet_p =', inlet_pressure_avg, 'delta_p =', delta_p)

    except ValueError:
        print('value error')
        delta_p = np.nan

    print(delta_p)
    return delta_p



# prob = TestSuite.PitzDaily(settings)
# lb, ub = prob.get_decision_boundary() # lb = lower bounds, ub = upper bounds

random_samples = get_samples(n_procs, n_control)

simulation_vector = [(random_samples[i]) for i in range(n_procs)]

# with Pool(n_procs) as p: 
#     # pressures = p.map(simulate(settings, simulation_vector))
#     pressures = p.map(simulate, simulation_vector)
    #collects results in a list
pressures = simulate(simulation_vector[0])

print(pressures)
#set up temporary directories, run openfoam, postprocess


