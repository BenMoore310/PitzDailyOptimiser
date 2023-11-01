import multiprocessing
from multiprocessing import Pool
import subprocess
from typing import Dict
import Exeter_CFD_Problems as TestSuite
import shutil
import tempfile
import os
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel
from sklearn.gaussian_process.kernels import RBF
from scipy.stats import norm
from itertools import product


def get_samples(n_samples, n_control):
    """Finds and returns a specified number of valid random decision vectors to be explicitely simulated

    Args:
        n_samples (int): desired number of random samples
        n_control (int): number of control points to define case geometry
    """

    #loads original globally defined settings including directory structure
    prob = TestSuite.PitzDaily(settings)
    lb, ub = prob.get_decision_boundary()

    #finds, checks validity of, and returns specified number of random decision vectors
    random_samples = np.empty((n_samples, n_control*2))
    for i in range(0, n_samples):
        is_valid = False

        while is_valid == False:
            x = np.random.random((1, lb.shape[0])) * (ub-lb) + lb
            is_valid = prob.constraint(x)
        random_samples[i] = x
    return(random_samples)  


def simulate(decision_vector):
    """Runs simulation for assigned case geometry, post-processes and returns inlet-outlet pressure difference

    Args:
        decision_vector (array): contains coordinates for control points defining case geometry, length defined by n_control

    Returns:
        pressure (int): inlet-outlet pressure difference
    """
    
    #generates temporary directory for simulation to run in
    tmpdir = tempfile.mkdtemp(dir='/home/links/bm424/PitzDailyOptimiser/')

    #copies relevant files to temporary directory
    shutil.copytree('Exeter_CFD_Problems', tmpdir, dirs_exist_ok=True) 


    #re-defines settings dict and updates with paths to temporary directory
    settings = {}

    settings['source_case'] = tmpdir + '/data/PitzDaily/case_fine/' # source directory
    settings['case_path'] = tmpdir + '/data/PitzDaily/case_single/' # case path where CFD simulations will run
    settings['boundary_files'] =  [tmpdir + '/data/PitzDaily/boundary.csv']
    settings['fixed_points_files'] = [tmpdir + '/data/PitzDaily/fixed.csv']
    settings['n_control'] = [n_control] # you should be able to use any 

    prob = TestSuite.PitzDaily(settings)

    #run simulation for case geometry, ignore error thrown when simulation converges
    try:
        res = prob.evaluate(decision_vector, verbose=True)
    except TypeError as e:
        print("Simulation finished with output", e)

    pressure = postprocess_result(settings, tmpdir)
    return pressure



def postprocess_result(settings, tmpdir):
    """Runs postprocessing on passed simulation to find and return inlet-outlet pressure difference

    Args:
        settings (dict): contains paths to relevant directories and files, specific to each temporary directory
        tmpdir (string): path to temporary directory

    Returns:
        delta_p (int): calculated pressure difference
    """

    #copies sampleDict postprocessing file to temporary directory then runs script
    shutil.copy('/home/links/bm424/PitzDailyOptimiser/sampleDict', settings["case_path"] + '/system/')

    subprocess.run(["postProcess -func sampleDict"], shell=True, cwd=settings["case_path"], check=True, capture_output=True)

    #reads pressure files from postprocessing
    outlet_pressure = np.loadtxt(os.path.join(settings["case_path"], 'postProcessing/sampleDict/0/outlet_p.xy'), dtype=float,usecols = 1)
    inlet_pressure = np.loadtxt(os.path.join(settings["case_path"], 'postProcessing/sampleDict/0/inlet_p.xy'), dtype=float,usecols = 1)

    #find inlet/outlet average pressures, and pressure difference magnitude
    try:
        outlet_pressure_avg = np.average(outlet_pressure) 
        inlet_pressure_avg = np.average(inlet_pressure)
        delta_p = abs(outlet_pressure_avg - inlet_pressure_avg)
        print('outlet_p =', outlet_pressure_avg, 'inlet_p =', inlet_pressure_avg, 'delta_p =', delta_p)

    except ValueError:
        print('value error')
        delta_p = np.nan

    #remove temporary directory
    shutil.rmtree(tmpdir) 
    return delta_p 

def expected_improvement(x, gp_model, best_y, epsilon):
    """ use gaussian regression model to predict pressure differences for list of valid geometries, 
        use equations from De Ath (2020) to find expected improvement for each point

    Args:
        x (array): decision vector
        gp_model : gaussian process regressor with specified kernel
        best_y (int): current lowest pressure difference
        epsilon (_type_): exploration parameter, balances exploring uncertain areas vs exploiting known regions

    Returns:
        ei, y_pred, y_std (list): lists of expected improvement, predicted pressure difference, and uncertainty (standard deviation) for all predicted points
    """
    y_pred, y_std = gp_model.predict(x, return_std=True)
    # print(y_std)
    z = (best_y - y_pred - epsilon)/y_std
    ei = ((best_y - y_pred - epsilon) * norm.cdf(z)) + y_std*norm.pdf(z)
    print(ei[np.argmax(ei)])
    return ei, y_pred, y_std


n_iters = 5

n_procs = 15

n_control = 5


#Globally defined settings dictionary using home directory structure.
settings = {}
settings['source_case'] = '/home/links/bm424/PitzDailyOptimiser/Exeter_CFD_Problems/data/PitzDaily/case_fine/' # source directory
settings['case_path'] = '/home/links/bm424/PitzDailyOptimiser/Exeter_CFD_Problems/data/PitzDaily/case_single/' # case path where CFD simulations will run
settings['boundary_files'] = ['/home/links/bm424/PitzDailyOptimiser/Exeter_CFD_Problems/data/PitzDaily/boundary.csv']
settings['fixed_points_files'] = ['/home/links/bm424/PitzDailyOptimiser/Exeter_CFD_Problems/data/PitzDaily/fixed.csv']
settings['n_control'] = [n_control] # you should be able to use any 

#existing lists/arrays of vectors, pressures, and valid possible vectors are loaded in
known_vectors = np.loadtxt('original_vectors.txt')
known_pressures = np.loadtxt('original_pressures.txt')
valid_shapes = np.loadtxt('valid_shapes.txt')
simulation_vectors = np.empty((n_procs,(n_control*2)))



prob = TestSuite.PitzDaily(settings)
lb, ub = prob.get_decision_boundary() # lb = lower bounds, ub = upper bounds

#kernel = RBF(length_scale=0.01, length_scale_bounds=(1e-9, 1e2))

# kernel = ConstantKernel(0.1, constant_value_bounds="fixed") * RBF(0.015, length_scale_bounds="fixed")

# gp_model = GaussianProcessRegressor(kernel=kernel, normalize_y=True)

gp_model = GaussianProcessRegressor(normalize_y=True)

highest_ei = 1
iter=0

#run is assumed to have converged when highest expected improvement drops below 1e-7.
while highest_ei > 1e-7:

    #gaussian regression re-fitted each iteration to include newest sampled vectors
    new_model = gp_model.fit(known_vectors, known_pressures)
    
    #calculates best current result to pass to expected improvement function
    best_idx = np.argmin(known_pressures)
    best_y = known_pressures[best_idx]

    ei, y_pred, y_std = expected_improvement(valid_shapes, new_model, best_y, 0.01)

    #finds 10 index in ei array with the largest values
    top_n_ei = np.argsort(ei)[-n_procs:]

    #generates list of decision vectors to simulate
    for i in range(n_procs):
        simulation_vectors[i] = valid_shapes[top_n_ei[i]]

    print(simulation_vectors)

    #initialises multiprocess pool, simulates runs for all specified simulation vectors at once on separate cores
    with Pool(n_procs) as p: 

        simulation_pressures = p.map(simulate, simulation_vectors)
    
    #simulation parameters and results appended to vector and pressure arrays
    known_vectors = np.vstack((known_vectors, simulation_vectors))
    known_pressures = np.append(known_pressures, simulation_pressures)

    np.savetxt('vectors_good.txt', known_vectors)
    np.savetxt('pressures_good.txt', known_pressures)

    iter += 1
    highest_ei = np.max(ei)
    print('largest expected improvement = ', highest_ei)


    #search for simulation results with nan values, remove relevent indices from pressure and vector arrays to prevent script stopping 
    to_delete = []
    for i in range(0, len(known_pressures)):
        if np.isnan(known_pressures[i]) == True:
            to_delete.append(i)


    if len(to_delete) > 0:
        print('invalid values in simulation :', to_delete)
        known_pressures = np.delete(known_pressures, to_delete)
        known_vectors = np.delete(known_vectors, to_delete, axis=0)

print('Converged in ', iter, ' iterations!')
np.savetxt('vectors_ei_5d.txt', known_vectors)
np.savetxt('pressures_ei_5d.txt', known_pressures)


