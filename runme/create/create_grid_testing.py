# Grids for the indicated PLZ are generated
# can also be used for PLZ areas that are not part of the official municipal register and have been created by the user

import time

from src.grid_generator import GridGenerator

# enter a plz to generate grid for:
plz = "10000"  # test region created by user

# timing of the script
start_time = time.time()

# generate grid
gg = GridGenerator(plz=plz)
gg.generate_grid()
gg.calc_parameters_per_plz()

# end timing
print("--- %s seconds ---" % (time.time() - start_time))
