# Grids for the indicated regional_identifier are generated
# can also be used for regional_identifier areas that are not part of the official
# municipal register and have been created by the user

import time

from src.grid_generator import GridGenerator

# enter a regional_identifier to generate grid for:
regional_identifier = "10000"  # test region created by user

# timing of the script
start_time = time.time()

# generate grid
gg = GridGenerator(regional_identifier=regional_identifier)
gg.generate_grid()
gg.calc_parameters_per_regional_identifier()

# end timing
print("--- %s seconds ---" % (time.time() - start_time))
