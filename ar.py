import os
from glob import glob
from subprocess import run
from multiprocessing import Pool
from shutil import copytree, copy
from time import perf_counter

import numpy as np
from ase.io import read

from utils import xe_kr_input, find_minimum_image


CUTOFF = 16.0


def write_sim_files(cif_name, na, nb, nc):
    # writes simulation.input along with all other files needed for the raspa sim to a single dir
    # returns the name of the directoy containing all the files + cif file
    sim_details = xe_kr_input(cif_name.replace('.cif', ''), na, nb, nc)
    dir_name = os.path.join('working_dir', cif_name.replace('.cif', ''))
    copytree('simulation_template', dir_name)
    copy(os.path.join('cifs', cif_name), os.path.join(dir_name, cif_name))
    with open(os.path.join(dir_name, 'simulation.input'), 'w') as f:
        f.writelines(sim_details)
    return dir_name


def parse_output(cif_dir):
    path = os.path.join(cif_dir, 'Output', 'System_0')
    base_path = glob(F"{path}/*.data")[0]

    components = {}
    with open(base_path, 'r') as fd:
        for line in fd:
            if "Number of molecules:" in line:
                break
        for line in fd:
            if line.startswith("Component"):
                name = line.split()[-1][1:-1]
            if "Average loading absolute   " in line:
                res = float(line.split(" +/-")[0].split()[-1])
                components[name] = res
    return components


def run_simulation(cif_name):
    atoms = read(os.path.join('cifs', cif_name), format="cif")
    cell = np.array(atoms.cell)
    na, nb, nc = find_minimum_image(cell, CUTOFF)   # 1. get number of unit cells to use
    cif_dir = write_sim_files(cif_name, na, nb, nc)  # 2. write files to location
    run(["simulate", "simulation.input"], cwd=cif_dir)  # 3. run simulation
    components = parse_output(cif_dir)  # 4. extract results
    selectivity = np.log(1 + (4 * components['xenon'])) - np.log(1 + components['krypton']) # 5. calc selectivity
    return cif_name, selectivity  # 6. return selectivity along with cif name
      
    
if __name__ == '__main__':
    a = perf_counter()
    
    with open('cif_list.txt', 'r') as f:
        names = [i.strip() for i in f.readlines()]
        
    pool = Pool(processes=4)
    output = []
    for n in names:
        result = pool.apply_async(run_simulation, (n,))
        output.append(result)
        
    pool.close()
    pool.join()
    
    output = [a.get() for a in output]
        
    print(output)
    
    b = perf_counter()
    print(b-a)
    