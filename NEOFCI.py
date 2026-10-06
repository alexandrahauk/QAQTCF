basis_sets_dir = "./basis-sets"

particle_properties_file = "particle-properties.json"

truncate_e = 0

mol_xyz = 'mols/H22.xyz'
spin_file = 'H2.spin'

num_eigvals = 20

import numpy as np
import scipy as sp
import json
import itertools
import argparse
import time
import math

from scipy.sparse.linalg import LinearOperator
from scipy.linalg import block_diag

from pyscf import gto, scf
from pyscf.lo import orth

from gbasis.wrappers import from_pyscf
from gbasis.parsers import parse_nwchem
from gbasis.parsers import make_contractions

from gbasis.integrals.overlap import overlap_integral
from gbasis.integrals.kinetic_energy import kinetic_energy_integral
from gbasis.integrals.electron_repulsion import electron_repulsion_integral

parser = argparse.ArgumentParser(
               prog='NEOFCI',
               description='Nuclear electronic orbitals full configuration interaction calculation',
               epilog='Written by Allie')
parser.add_argument('mol_xyz')
parser.add_argument('spin_file')
parser.add_argument('e_basis_set')
parser.add_argument('n_basis_set')

parser.add_argument('-t', '--truncate', help='Number of electronic SCF orbitals to truncate', type=int) # truncate
parser.add_argument('-g', '--gpu', help='Use GPU acceleration with cupy', action='store_true') # truncate

args = parser.parse_args()

if args.truncate is None:
    args.truncate = 0

if args.gpu:
    import cupy as cp
else:
    cp = np

# Load properties of all possible particles (spin, fermion/boson, mass, charge, etc)
with open(particle_properties_file, "r") as file:
    particle_properties = json.load(file)

# Build molecule for PySCF
mol = gto.Mole()
mol.atom = args.mol_xyz
mol.basis = args.e_basis_set
mol.build()

mol_zs = mol.atom_charges()
mol_symbs = [mol.atom_symbol(i) for i in range(mol.natm)] # Atomic symbols
mol_coords = mol.atom_coords()

# Run restricted Hartree Fock to get better orbitals (I might truncate the highest energy ones)
hf = scf.RHF(mol).run() # TODO: apparently there might be better choices of orbital to allow for truncations (FNO). look into?

# Load basis dictionary (atomic orbitals) for nuclear orbitals
n_basis_dict = parse_nwchem(basis_sets_dir + '/nuclear/' + args.n_basis_set + '.nw')

# Construct a dictionary of all the particle types that will be in our calculation, along with their orbitals and info like spin.
# Note: we will use the order of the dictionary. Python 3.7+ guarantees when we iterate, the dictionary will be ordered according to when the elements were added.
particles = {}

for i in range(mol.natm):
    symb = mol.atom_symbol(i)
    # If this is a particle type (nucleus) we haven't seen before
    if symb not in particles:
        # Add it to particle list
        particles[symb] = {}
        particles[symb]['coords'] = []
        particles[symb]['count'] = 0

    # Add its coordinates to the list
    particles[symb]['coords'].append(mol_coords[i])

    particles[symb]['count'] += 1

for symb in particles:
    # GBasis wants coords as numpy array
    particles[symb]['coords'] = np.array(particles[symb]['coords'])

    # Construct a basis w/ GBasis for each of the nuclear particles
    particles[symb]['basis'] = make_contractions(n_basis_dict, # Basis of (nuclear) AOs to use
                                                 [symb] * particles[symb]['count'], # Types of atoms (all the same)
                                                 particles[symb]['coords'], # Coordinates
                                                 coord_types='cartesian')

    # Transform to orthonormal orbitals
    overlap = overlap_integral(particles[symb]['basis'])
    particles[symb]['transform'] = orth.lowdin(overlap) # Symmetric orthonormalization of AOs

    # Number of spatial orbitals
    particles[symb]['no_spatial_orbitals'] = overlap.shape[0]

# Retrieve some important properties on each particle (spin, fermion/boson, mass, charge)
for symb in particles:
    particles[symb]['properties'] = particle_properties[symb]

# Add electrons to our particle list
particles['e'] = {}
particles['e']['basis'] = from_pyscf(mol) # Gbasis set of GTOs (gaussian type orbitals) for electronic particles
particles['e']['transform'] = hf.mo_coeff.T # transform to MOs that will be used for calculation
particles['e']['count'] = mol.nelectron  # Get number of electrons from PySCF, truncate off 10
particles['e']['no_spatial_orbitals'] = hf.mo_coeff.shape[0]

idx = 0

# Retrieve some important properties on each particle (spin, fermion/boson, mass, charge)
for symb in particles:
    particles[symb]['idx'] = idx
    idx += 1

    particles[symb]['properties'] = particle_properties[symb]

    # Number of spin orbitals, since we now have the particle's spin
    particles[symb]['no_spin_orbitals'] = particles[symb]['no_spatial_orbitals'] * particles[symb]['properties']['spin']

# Read in file containing the count of particles in each spin state (alpha, beta, etc) for each particle
with open(args.spin_file, 'r', encoding='utf-8') as file:
    while True:
        line = file.readline()
        if not line:
            break  # End of file reached

        split = line.strip().split()
        particles[split[0]]['occs'] = [int(s) for s in split[1:]]

# Amount of particles we have
particle_types = len(particles)

# List of particle names for easy indexing
particle_names = [symb for symb in particles]

# TODO: I should probably just do all the integrals at once by combining the bases of all particles.

# For FCI, we will construct the Hamiltonian in the subspace of all states with only the correct particle numbers
# A basis for this space is constructed from N-particle determinants/permanents of the one-particle basis states for correct N

# Total number of states in this space
total_states = 1

# Get states with non-zero coupling constant gamma^IJ_ij = <I|gamma_ij|J> = <I|a^\dag_i a_j|J>
# gamma^IJ_ij = parity
# i = occupy, j = deocccupy
# J = input state, I = output state
def get_diff_states(occupied, addr_array):
    no_orbitals = addr_array.shape[1]

    # Indices of the occupied & unoccupied one-particle states for this N-particle state
    unoccupied = [i for i in range(no_orbitals) if i not in occupied]

    new_states = []

    state_idx = get_addr(occupied, addr_array)

    for deoccupy_idx, deoccupy in enumerate(occupied):
        for occupy in unoccupied:
            # Construct new state
            new_occupied = occupied.copy()

            new_occupied[deoccupy_idx] = occupy
            new_occupied.sort()


            # States in common
            common = [i for i in occupied if i != deoccupy]

            # Get index of this new state
            new_state_idx = get_addr(new_occupied, addr_array)

            # How many permutations to align the old state and new state's determinants?
            perms = sum([min(deoccupy, occupy) < c < max(deoccupy, occupy) for c in common])

            # Fermion exchange parity from aligning the determinants
            parity = 1 if perms % 2 == 0 else -1

            # Package all this info together
            new_states.append((new_state_idx, occupy, deoccupy, parity))

        # Now for replacements that give us the same state, E_ii:
        new_states.append((state_idx, deoccupy, deoccupy, 1))

    return new_states

def create_addr_array(M, N):
    if N == 0:
        return np.array([[]])

    addr_array = np.zeros((N, M), dtype=int)

    for k in range(N-1):
        for l in range(k, M-N+k+1):
            addr_array[k][l] = sum([
                math.comb(m, N-k-1) - math.comb(m-1, N-k-2) for m in range(M-l, M-k)
            ])

    # second case (k = N-1, or N in the original paper, non-zero indexed)
    for l in range(N-1, M):
        addr_array[N-1][l] = l+1-N

    return addr_array

def get_addr(orbitals, addr_array):
    return sum([addr_array[i,j] for i,j in enumerate(orbitals)])

string_mtx_shape = []

tensor_idx = 0
full_basis_idx = 0

# Combine bases of all the particles (the overlaps between different particles will be nonsensical)
full_basis = tuple(itertools.chain.from_iterable([particles[symb]['basis'] for symb in particles]))

# Full orthonormalization transform for this basis
full_transform = block_diag(*tuple([particles[symb]['transform'] for symb in particles]))

# All coulomb integrals contained in this
full_cmb_int = electron_repulsion_integral(full_basis, notation='chemist', transform=full_transform)

print("2e ints formed")

particles['e']['no_spatial_orbitals'] -= args.truncate

# Construct N-particle states
for symb in particles:
    particle = particles[symb]
    particle['states'] = []
    particle['addr_arrays'] = []
    particle['counts'] = []

    no_states = 1

    # Spatial orbital:  11223344
    for spin in range(particle['properties']['spin']):
        states = [] # States for this particular particle ("strings" in Handy-Knowles paper)

        occ = particle['occs'][spin] # Number of this particle with this spin
        for indices in itertools.combinations(range(particle['no_spatial_orbitals']), occ):
            states.append(list(indices))

        particle['states'].append(states)
        particle['counts'].append(len(states))

        particle['addr_arrays'].append(create_addr_array(particle['no_spatial_orbitals'], occ))

        string_mtx_shape.append(len(states))

        no_states *= len(states)
        total_states *= len(states)

    particle['tensor_idx'] = tensor_idx # index in the tensor of strings (think handy knowles alpha beta string)
    particle['full_basis_idx'] = full_basis_idx

    tensor_idx += particle['properties']['spin']
    full_basis_idx += particle['no_spatial_orbitals']

# Now we will multiple these two-electron integrals by the factors coming from the charge and particle statistics
for symb1 in particles:
    particle1 = particles[symb1]
    charge1 = particles[symb1]['properties']['charge']
    fbi1 = particle1['full_basis_idx']
    obtl1 = particle1['no_spatial_orbitals']

    for symb2 in particles:
        particle2 = particles[symb2]
        charge2 = particles[symb2]['properties']['charge']
        fbi2 = particle2['full_basis_idx']
        obtl2 = particle2['no_spatial_orbitals']

        half = 0.5 if symb1 == symb2 else 1

        full_cmb_int[fbi1:fbi1+obtl1, fbi1:fbi1+obtl1, fbi2:fbi2+obtl2, fbi2:fbi2+obtl2] *= charge1 * charge2 * half

# Construct single replacements
for symb in particles:
    particle = particles[symb]

    particle['states_r'] = []

    for spin in range(particle['properties']['spin']):
        states_r = [get_diff_states(state, particle['addr_arrays'][spin]) for state in particle['states'][spin]]
        particle['states_r'].append(states_r)

int_time = time.perf_counter()

def matvec(v):
    C = v.reshape(string_mtx_shape, order='F')
    if args.gpu: # gpu acceleration
        C = cp.asarray(C)

    sigma = cp.zeros(string_mtx_shape)

    for symb in particles:
        particle = particles[symb]
        tensor_idx = particle['tensor_idx']
        fbi = particle['full_basis_idx']

        mass = particle['properties']['mass']
        spin = particle['properties']['spin']

        fbi = particle['full_basis_idx']
        obtl = particle['no_spatial_orbitals']

        ke_int = kinetic_energy_integral(particle['basis'], transform=particle['transform']) / mass
        cmb_int = full_cmb_int[fbi:fbi+obtl, fbi:fbi+obtl, fbi:fbi+obtl, fbi:fbi+obtl]

        if args.gpu:
            ke_int = cp.asarray(ke_int)
            cmb_int = cp.asarray(cmb_int)

        D = cp.zeros(tuple(string_mtx_shape) + (particle['no_spatial_orbitals'], particle['no_spatial_orbitals']))

        # Coulomb interaction between like particles only
        for s in range(spin):
            idx_lhs = [slice(None)] * D.ndim
            idx_rhs = [slice(None)] * C.ndim

            for state_no, (state, states_r) in enumerate(zip(particle['states'][s], particle['states_r'][s])):
                idx_lhs[tensor_idx + s] = state_no
                for state_r_no, i, j, parity in states_r:

                    idx_rhs[tensor_idx + s] = state_r_no

                    idx_lhs[-2], idx_lhs[-1] = i, j

                    idxt_lhs, idxt_rhs = tuple(idx_lhs), tuple(idx_rhs)

                    D[idxt_lhs] += C[idxt_rhs] * parity


        E = cp.tensordot(D, cmb_int, axes=([-2, -1], [-2, -1]))

        for s in range(spin):
            idx_lhs = [slice(None)] * sigma.ndim
            idx_rhs = [slice(None)] * E.ndim

            for state_no, (state, states_r) in enumerate(zip(particle['states'][s], particle['states_r'][s])):
                idx_rhs[tensor_idx + s] = state_no
                for state_r_no, i, j, parity in states_r:

                    idx_lhs[tensor_idx + s] = state_r_no

                    idx_rhs[-2], idx_rhs[-1] = i, j

                    idxt_lhs, idxt_rhs = tuple(idx_lhs), tuple(idx_rhs)

                    sigma[idxt_lhs] += E[idxt_rhs] * parity

        for s in range(spin):
            idx_lhs = [slice(None)] * sigma.ndim
            idx_rhs = [slice(None)] * C.ndim

            for state_no, (state, states_r) in enumerate(zip(particle['states'][s], particle['states_r'][s])):
                idx_rhs[tensor_idx + s] = state_no  # state_no is J
                for state_r_no, i, l, parity in states_r:
                    idx_lhs[tensor_idx + s] = state_r_no # state_r_no is I

                    elmt = sum(full_cmb_int[i+fbi, j+fbi, j+fbi, l+fbi] for j in range(obtl))

                    idxt_lhs, idxt_rhs = tuple(idx_lhs), tuple(idx_rhs)

                    sigma[idxt_lhs] -= elmt * parity * C[idxt_rhs]

        # Kinetic energy
        for s in range(spin):
            idx_lhs = [slice(None)] * sigma.ndim
            idx_rhs = [slice(None)] * C.ndim

            for state_no, (state, states_r) in enumerate(zip(particle['states'][s], particle['states_r'][s])):
                idx_rhs[tensor_idx + s] = state_no  # J
                for state_r_no, i, j, parity in states_r:
                    idx_lhs[tensor_idx + s] = state_r_no # I

                    idxt_lhs, idxt_rhs = tuple(idx_lhs), tuple(idx_rhs)

                    sigma[idxt_lhs] += parity * C[idxt_rhs] * ke_int[i, j]

    # Coulomb interactions between
    for symb1, symb2 in itertools.combinations(particles, 2):
        particle1 = particles[symb1]
        particle2 = particles[symb2]

        fbi1 = particle1['full_basis_idx']
        obtl1 = particle1['no_spatial_orbitals']

        fbi2 = particle2['full_basis_idx']
        obtl2 = particle2['no_spatial_orbitals']

        tensor_idx1 = particle1['tensor_idx']
        tensor_idx2 = particle2['tensor_idx']

        spin1 = particle1['properties']['spin']
        spin2 = particle2['properties']['spin']

        cmb_int = full_cmb_int[fbi1:fbi1+obtl1, fbi1:fbi1+obtl1, fbi2:fbi2+obtl2, fbi2:fbi2+obtl2]

        if args.gpu:
            cmb_int = cp.asarray(cmb_int)

        D = cp.zeros(tuple(string_mtx_shape) + (obtl2, obtl2))
        E = cp.zeros(tuple(string_mtx_shape) + (obtl1, obtl1))

        for s2 in range(spin2):
            idx_lhs = [slice(None)] * D.ndim
            idx_rhs = [slice(None)] * C.ndim

            for state_no, (state, states_r) in enumerate(zip(particle2['states'][s2], particle2['states_r'][s2])):
                idx_lhs[tensor_idx2 + s2] = state_no
                for state_r_no, a, b, parity in states_r:
                    idx_rhs[tensor_idx2 + s2] = state_r_no

                    idx_lhs[-2], idx_lhs[-1] = a, b

                    idxt_lhs, idxt_rhs = tuple(idx_lhs), tuple(idx_rhs)

                    D[idxt_lhs] += C[idxt_rhs] * parity

        E = cp.tensordot(D, cmb_int, axes=([-2, -1], [-2, -1]))

        for s1 in range(spin1):
            idx_lhs = [slice(None)] * sigma.ndim
            idx_rhs = [slice(None)] * E.ndim

            for state_no, (state, states_r) in enumerate(zip(particle1['states'][s1], particle1['states_r'][s1])):
                idx_rhs[tensor_idx1 + s1] = state_no
                for state_r_no, i, j, parity in states_r:
                    idx_lhs[tensor_idx1 + s1] = state_r_no

                    idx_rhs[-2], idx_rhs[-1] = i, j

                    idxt_lhs, idxt_rhs = tuple(idx_lhs), tuple(idx_rhs)

                    sigma[idxt_lhs] += E[idxt_rhs] * parity

    if args.gpu:
        sigma = cp.asnumpy(sigma)

    sigma = sigma.reshape(total_states, order='F')

    return sigma

H = LinearOperator(shape=(total_states, total_states), matvec=matvec, dtype=float)

h_eigvals, h_eigvecs = sp.sparse.linalg.eigsh(H, k=num_eigvals, which='SA', tol=1e-10, maxiter=250)

diag_time = time.perf_counter()

print(f"Iterative diagonalization completed in {diag_time-int_time:.1f}s")

print(h_eigvals)

np.savetxt('eigvals.dat', h_eigvals, delimiter=',', fmt='%f')
np.savetxt('eigvecs.dat', h_eigvecs, delimiter=',', fmt='%f')