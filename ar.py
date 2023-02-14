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


def write_sim_files(sim_dir, cif_name, na, nb, nc):
    # writes simulation.input along with all other files needed for the raspa sim to a single dir
    # returns the name of the directoy containing all the files + cif file
    sim_details = xe_kr_input(cif_name.replace('.cif', ''), na, nb, nc)
    copy(os.path.join('cifs', cif_name), os.path.join(sim_dir, cif_name))
    sim_file_name = F"simulation_{cif_name.replace('.cif', '')}.input"
    with open(os.path.join(sim_dir, sim_file_name), 'w') as f:
        f.writelines(sim_details)
    return sim_file_name


def parse_output(results_dir, cif_name_clean):
    path = os.path.join(results_dir, 'Output', 'System_0')
    base_path = glob(F"{path}/output_{cif_name_clean}_*.data")[0]

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


class RaspaRegistry:
    
    def __init__(self, cif_list_path, simulation_dir='raspa_dir'):
        self.simulation_dir = str(simulation_dir)
                
        with open(str(cif_list_path), 'r') as f:
            self.cifs = [i.strip() for i in f.readlines()]
            
        if not os.path.exists(self.simulation_dir):
            copytree('simulation_template', self.simulation_dir)
            
    def run_simulation(self, idx):
        cif_name = self.cifs[idx]
        cif_name_clean = cif_name.replace('.cif', '')
        
        atoms = read(os.path.join('cifs', cif_name), format="cif")
        cell = np.array(atoms.cell)
        na, nb, nc = find_minimum_image(cell, CUTOFF)   # 1. get number of unit cells to use
        sim_file_name = write_sim_files(self.simulation_dir, cif_name, na, nb, nc)  # 2. write files to location
        run(["simulate", "-i", sim_file_name], cwd=self.simulation_dir)  # 3. run simulation
        components = parse_output(self.simulation_dir, cif_name_clean)  # 4. extract results
        selectivity = np.log(1 + (4 * components['xenon'])) - np.log(1 + components['krypton']) # 5. calc selectivity
        results = {'index': idx, 'name': cif_name, 'selectivity': selectivity, **components}
        return results  # 6. return selectivity along with cif name
    
    def __len__(self):
        return len(self.cifs)
      
    
if __name__ == '__main__':
    a = perf_counter()
    
    raspa = RaspaRegistry('cif_list.txt', simulation_dir='raspa_dir')
    
    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(raspa.run_simulation, idx) for idx in range(len(raspa))]
        
    output = [a.result() for a in futures]
        
    for o in output:
        print(o)
    
    b = perf_counter()
    print(b-a)
    