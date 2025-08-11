# Must be run after get_parameters_per_regional_identifier.py
from src.parameter_calculator import ParameterCalculator

regional_identifier = 1
pc = ParameterCalculator()
pc.calc_parameters_per_grid(regional_identifier)
