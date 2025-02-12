# General functions to be used while developing a KMC simulation
import numpy as np
from ase import Atoms
from ase.visualize import view


boltz = 8.617333262e-5  # eV

def initialize_lattice(lattice_size):
    """
    Initializes a cubic lattice of size x size x size dimensions
    with 0s as every element

    Parameters:
        size: int
    Returns:
        numpy.ndarray
            Cubic array
    """

    return np.zeros((lattice_size, lattice_size, lattice_size), dtype=int)

def set_vacancy_center(lattice, lattice_size):
    """
    Initializes a vacancy in the center of the lattice
    """

    vacancy_position = [lattice_size // 2] * 3
    lattice[tuple(vacancy_position)] = 1

    return lattice, vacancy_position

def calc_rate_constant(attempt_frequency, activation_energy, temperature):
    """
    Calculates the rate constant of a transition
    """

    return attempt_frequency * np.exp(-activation_energy / (boltz * temperature))\

def calc_neighbor_offsets(lattice_type):
    """
    Should be able to generate list of offsets for a given lattice type
    """

def calc_diffusion_constant(final_vacancy_position, initial_vacancy_position, total_time):
    """
    Calculates the diffusion constant
    """

    coefficient = 1 / (6 * total_time)
    inside_arg = np.mean((final_vacancy_position - initial_vacancy_position) ** 2)

    return coefficient * inside_arg

def run_simulation(num_steps, lattice_size, temp, activation_energy, attempt_frequency, neighbor_offsets, rate_constant, trajectory_file_name, create_trajectory=False):
    """
    Use kinetic monte carlo to simulate and calculate the diffusion constant for vacancy diffusion.

    Parameters:
        num_steps: int
            Number of simulation steps
        lattice_size: int
            Size of the lattice
        temp: float or int
            Temperature of the simulation
        activation_energy: float or int
            Activation energy of the diffusion of a vacancy to another positon
        attempt_freqency: float or int
            Frequency of attempts to overcome the diffusion barrier
        neighbor_offsets: list[tuple(int, int, int)]
            Contains indices that represent the possible direction of vacancy diffusion

    Returns:
        numpy.ndarray
            Cubic array
    """
    time_per_num_steps = []
    diffusion_constants = []

    rate_constant = calc_rate_constant(attempt_frequency, activation_energy, temp)

    lattice, vacancy_position = set_vacancy_center(initialize_lattice(lattice_size), lattice_size)
    
    # Keep track of vacancy position and time elapsed
    vacancy_positions = [tuple(vacancy_position)]
    time = 0
    time_steps = [time]

    for step in range(num_steps):
        print(f"num_steps: {num_steps:,} - step: {step:,}")
        while True:
            possible_moves = []
            for offset in neighbor_offsets: # Calculate coordinates for neighboring cation sites
                neighbor = [
                    (vacancy_position[i] + offset[i]) % lattice_size for i in range(3)
                ]
                possible_moves.append(neighbor)

            probability = [rate_constant for move in possible_moves]
            probability /= np.sum(probability) # Calculate the probability of jumping to each site (all jumps are equally probable in this scenario)

            chosen_move_idx = np.random.choice(len(possible_moves))
            possible_position = possible_moves[chosen_move_idx]

            xi = np.random.random()

            if xi < probability[chosen_move_idx]:
                break
        
        lattice[tuple(vacancy_position)] = 0
        lattice[tuple(possible_position)] = 1
        vacancy_position = possible_position
        vacancy_positions.append(tuple(vacancy_position))

        total_rate = np.sum([rate_constant for _ in possible_moves])
        time += -np.log(np.random.random()) / total_rate # Why not time += 1 / total_rate?
        time_steps.append(time)
        time_per_num_steps.append(time_steps)

    coeff = 1 / (6 * time)
    inside = np.mean((np.array(vacancy_positions[-1]) - np.array(vacancy_positions[0])) ** 2)
    diffusion_constant = coeff * inside
    diffusion_constants.append(diffusion_constant)

    if not create_trajectory:
        return time_per_num_steps, diffusion_constants
    else:
        list_atoms = []
        for position in vacancy_positions:
            # Create a list of chemical symbols for the lattice sites

            chemical_symbols = []
            for i in range(lattice_size):
                for j in range(lattice_size):
                    for k in range(lattice_size):
                        if (i, j, k) == position:
                            chemical_symbols.append('F') # Vacancy
                        else:
                            chemical_symbols.append('O')

            atoms = Atoms(chemical_symbols, positions=[(i, j, k) for i in range(lattice_size) for j in range(lattice_size) for k in range(lattice_size)], pbc=True, cell=[lattice_size, lattice_size, lattice_size])
            list_atoms.append(atoms)


        from ase.io import write
        write(f'{trajectory_file_name}.extxyz', list_atoms)

    return time_per_num_steps, diffusion_constants