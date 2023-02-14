import os
from glob import glob
from subprocess import run
from concurrent.futures import ProcessPoolExecutor
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


class CifRegistry:
    
    def __init__(self, cif_list_path):        
        with open(str(cif_list_path), 'r') as f:
            self.cifs = [i.strip() for i in f.readlines()][:1]
            
    def run_simulation(self, idx):
        cif_name = self.cifs[idx]
        atoms = read(os.path.join('cifs', cif_name), format="cif")
        cell = np.array(atoms.cell)
        na, nb, nc = find_minimum_image(cell, CUTOFF)   # 1. get number of unit cells to use
        cif_dir = write_sim_files(cif_name, na, nb, nc)  # 2. write files to location
        run(["simulate", "simulation.input"], cwd=cif_dir)  # 3. run simulation
        components = parse_output(cif_dir)  # 4. extract results
        selectivity = np.log(1 + (4 * components['xenon'])) - np.log(1 + components['krypton']) # 5. calc selectivity
        results = {'index': idx, 'name': cif_name, 'selectivity': selectivity, **components}
        return results  # 6. return selectivity along with cif name
    
    def __len__(self):
        return len(self.cifs)
      
    
if __name__ == '__main__':
    a = perf_counter()
    
    cifs = CifRegistry('cif_list.txt')
    
    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(cifs.run_simulation, idx) for idx in range(len(cifs))]
        
    output = [a.result() for a in futures]
        
    print(output)
    
    b = perf_counter()
    print(b-a)
    