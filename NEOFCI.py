from scipy.constants import electron_mass

mol_dir = "./mols"
basis_sets_dir = "./basis-sets"

particle_properties_file = "particle-properties.json"

mol_name = "H2"

truncate_e = 0

mtx_elmt_threshold = 1e-7

import numpy as np
import scipy as sp
import torch
import json
import itertools
import argparse
import time

from scipy.sparse import coo_matrix, csr_matrix
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
parser.add_argument('e_basis_set')
parser.add_argument('n_basis_set')

args = parser.parse_args()

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using GPU")
elif torch.backends.mps.is_available():
    device = torch.device("mps")  # For Apple Silicon
else:
    device = torch.device("cpu")
    print("Using CPU")


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
particles['e']['no_spatial_orbitals'] = hf.mo_coeff.shape[0] - truncate_e

idx = 0

# Retrieve some important properties on each particle (spin, fermion/boson, mass, charge)
for symb in particles:
    particles[symb]['idx'] = idx
    idx += 1

    particles[symb]['properties'] = particle_properties[symb]

    # Number of spin orbitals, since we now have the particle's spin
    particles[symb]['no_spin_orbitals'] = particles[symb]['no_spatial_orbitals'] * particles[symb]['properties']['spin']

# Amount of particles we have
particle_types = len(particles)

# List of particle names for easy indexing
particle_names = [symb for symb in particles]

# TODO: I should probably just do all the integrals at once by combining the bases of all particles.

# Our basis in particles[symb]['basis'] is a basis of spatial orbitals, so the integrals will be between spatial oritals.
# This class will take a matrix (or 4D array for 2-body interactions) of these spatial integrals and give us a matrix that indexes spin orbitals
# We use the convention defined below for indexing spin orbitals
#
# This class ASSUMES 2-body integrals are in CHEMIST'S NOTATION:
# data[i,j,k,l] = \int dr_1 dr_2 \chi_i^*(r_1) \chi_j(r_1) \hat{O}_2 \chi_k^*(r_2) \chi_l(r_2)
class IntegralSpinWrapper:
    # data: table of integrals (2D array for 1-body, 4D array for 2-body). Should be a numpy array, which it will be for integrals from GBasis
    # spin: number of spin states if 1-body, or 2-tuple of the number of spin states for each particle if 2-body
    def __init__(self, data, spin):
        self.data = data
        self.spin = spin

        self.is_two_body = len(data.shape) == 4

    def __getitem__(self, index):
        # If one-body interaction
        if not self.is_two_body:
            # Spin states must be equal.
            if index[0] % self.spin == index[1] % self.spin:
                return self.data[int(index[0] / self.spin), int(index[1] / self.spin)]

            # Otherwise by orthonormality the integral is 0
            else:
                return 0

        # If two-body interaction
        else:
            # Spin states must be equal for both coordinates:
            if (index[0] % self.spin[0] == index[1] % self.spin[0]) and (index[2] % self.spin[1] == index[3] % self.spin[1]):
                return self.data[int(index[0]/self.spin[0]), int(index[1]/self.spin[0]), int(index[2]/self.spin[1]), int(index[3]/self.spin[1])]

            # Otherwise by orthonormality the integral is 0
            else:
                return 0

# For full FCI, we will construct the Hamiltonian in the subspace of all states with only the correct particle numbers
# A basis for this space is constructed from N-particle determinants/permanents of the one-particle basis states for correct N

# Total number of states in this space
total_states = 1

# Number of permanents/determinants for each particle.
no_states = []

# For the Hamiltonian matrix, we will index the states as follows:
# Let N_i be the number of states for the i-th particle
# The state formed from the c_0 - th particle 1 permanent/determinant, c_1 - th particle 2 perminant/determinant, etc (c_i's zero indexed)
# Will get the index (c_0) + (c_1 * N_0) + (c_2 * N_1 * N_0) + (c_3 * N_2 * N_1 * N_0) + ...

# This works basically like a numeral base system where each digit has a different base (each digit is a particle).
# This array will contain the bases for each particle: [1, N_0, N_1 * N_0, ...]
bases = []

# Construct N-particle states
for symb in particles:
    particles[symb]['states'] = []

    # Construct all N-particle states for the correct N, in the forms of arrays of 1s and 0s. They will be indexed in the order we get them from itertools
    # States with the same spatial wave function will be grouped together. For example, for 4 spatial orbitals with A and B spin states, the significance of the bits will be
    # Bit number:       12345678
    # Spin:             ABABABAB
    # Spatial orbital:  11223344
    for indices in itertools.combinations(range(particles[symb]['no_spin_orbitals']), particles[symb]['count']):
        array = [0] * particles[symb]['no_spin_orbitals']
        for index in indices:
            array[index] = 1

        particles[symb]['states'].append(array)

    particles[symb]['no_states'] = len(particles[symb]['states'])
    particles[symb]['base'] = total_states

    no_states.append(particles[symb]['no_states'])
    bases.append(particles[symb]['base'])
    total_states *= particles[symb]['no_states']

bases.append(total_states) # Having this extra element will be helpful in construct_1_particle_interaction

# Given a certain N-particle state index, construct an array of the indices of all states differing by d one-particle states, combined with info on which states are different and what the parity factor is after aligning up all the common states
def get_diff_states(particle, state_idx, d):
    state = particle['states'][state_idx]

    # Indices of the occupied & unoccupied one-particle states for this N-particle state
    occupied = [i for i in range(len(state)) if state[i] == 1]
    unoccupied = [i for i in range(len(state)) if state[i] == 0]

    new_states = []

    deoccupy_comb = itertools.combinations(occupied, d)
    occupy_comb = itertools.combinations(unoccupied, d)

    comb = itertools.product(itertools.combinations(occupied, d), # Which d one-particle states to de-occupy in the new state
                             itertools.combinations(unoccupied, d)) # Which d one-particle states to occupy in the new state

    for deoccupy, occupy in comb:
        # Construct new state
        new_state = state.copy()
        for d in deoccupy:
            new_state[d] = 0 # Deoccupy these states

        for o in occupy:
            new_state[o] = 1 # Occupy these states

        # States in common
        common = [i for i in occupied if i not in deoccupy]

        # Get index of this new state
        new_state_idx = particle['states'].index(new_state)

        # How many permutations to align the old state and new state's determinants?
        perms = sum([sum(min(d, o) < c < max(d, o) for c in common) for d, o in zip(deoccupy, occupy)])

        # Fermion exchange parity from aligning the determinants
        parity = 1 if perms % 2 == 0 else -1

        # Package all this info together
        new_states.append((new_state_idx, common, deoccupy, occupy, parity))

    return new_states

print('[ ] Constructing kinetic energy mtx elements...', end='\r')
start_ke = time.perf_counter()

t1s = []

# One-body "interactions" (kinetic energy)
for symb in particles:
    particle = particles[symb]

    mass = particle['properties']['mass'] # Mass of the particle
    spin = particle['properties']['spin'] # Spin of the particle

    no_states = particle['no_states'] # Number of N-particle states

    # Get kinetic energy integrals
    ke_int = kinetic_energy_integral(particle['basis'], particle['transform']) / mass
    particle['ke_int'] = ke_int # Store them

    # Get the spin orbital integrals
    ke_int_spin = IntegralSpinWrapper(ke_int, spin)

    t1_rows = []
    t1_cols = []
    t1_values = []

    # For all N-particle bras, use Slater-Condon rules to calculate matrix elements
    for bra_idx in range(no_states):
        # Matrix elements for states that differ by 0 one-particle states
        for ket_idx, common, deoccupy, occupy, parity in get_diff_states(particle, bra_idx, 0):
            elmt = sum([ke_int_spin[c, c] for c in common])
            elmt *= parity

            #t_mtx += construct_1_particle_interaction(particle, bra_idx, ket_idx, elmt)
            if abs(elmt) > mtx_elmt_threshold:
                t1_rows.append(bra_idx)
                t1_cols.append(ket_idx)
                t1_values.append(elmt)

        # Matrix elements for states that differ by 1 one-particle state
        for ket_idx, common, deoccupy, occupy, parity in get_diff_states(particle, bra_idx, 1):
            elmt = ke_int_spin[deoccupy[0], occupy[0]]
            elmt *= parity

            #t_mtx += construct_1_particle_interaction(particle, bra_idx, ket_idx, elmt)
            if abs(elmt) > mtx_elmt_threshold:
                t1_rows.append(bra_idx)
                t1_cols.append(ket_idx)
                t1_values.append(elmt)

    t1s.append((particle, torch.sparse_coo_tensor([t1_rows, t1_cols], t1_values, size=(no_states, no_states), dtype=torch.double)))


start_pe_like = time.perf_counter()
time_ke = start_pe_like - start_ke

print(f"[X] Kinetic energy mtx elements constructed in {time_ke:.1f}s")
print('[ ] Constructing potential energy mtx elements between like particles...', end='\r')

v1s = []

# Two-body interactions between particles of the SAME type. TODO: implement boson interactions with their different exchange behavior
for symb in particles:
    particle = particles[symb]

    spin = particle['properties']['spin'] # Spin of the particle
    no_states = particle['no_states'] # Number of N-particle states

    # Get two body integrals (coulomb force)
    cmb_int = electron_repulsion_integral(particle['basis'], particle['transform'], notation='chemist') * (particle['properties']['charge'] ** 2)
    particle['cmb_int'] = cmb_int # Store them

    # Get the spin orbital integrals
    cmb_int_spin = IntegralSpinWrapper(cmb_int, (spin, spin))

    v1_rows = []
    v1_cols = []
    v1_values = []

    # For all N-particle bras, use Slater-Condon rules to calculate matrix elements
    for bra_idx in range(no_states):
        # Matrix elements for states that differ by 0 one-particle states
        for ket_idx, common, deoccupy, occupy, parity in get_diff_states(particle, bra_idx, 0):
            elmt = sum([cmb_int_spin[i,i,j,j]-cmb_int_spin[i,j,j,i] for i, j in itertools.combinations(common, 2)])
            elmt *= parity

            #v_mtx += construct_1_particle_interaction(particle, bra_idx, ket_idx, elmt)
            if abs(elmt) > mtx_elmt_threshold:
                v1_rows.append(bra_idx)
                v1_cols.append(ket_idx)
                v1_values.append(elmt)

        # Matrix elements for states that differ by 1 one-particle state
        for ket_idx, common, deoccupy, occupy, parity in get_diff_states(particle, bra_idx, 1):
            elmt = sum([cmb_int_spin[deoccupy[0], occupy[0], i, i] - cmb_int_spin[deoccupy[0], i, i, occupy[0]] for i in common])
            elmt *= parity

            #v_mtx += construct_1_particle_interaction(particle, bra_idx, ket_idx, elmt)
            if abs(elmt) > mtx_elmt_threshold:
                v1_rows.append(bra_idx)
                v1_cols.append(ket_idx)
                v1_values.append(elmt)

        # Matrix elements for states that differ by 2 one-particle state
        for ket_idx, common, deoccupy, occupy, parity in get_diff_states(particle, bra_idx, 2):
            elmt = cmb_int_spin[deoccupy[0], occupy[0], deoccupy[1], occupy[1]] - cmb_int_spin[deoccupy[0], occupy[1], deoccupy[1], occupy[0]]
            elmt *= parity

            #v_mtx += construct_1_particle_interaction(particle, bra_idx, ket_idx, elmt)
            if abs(elmt) > mtx_elmt_threshold:
                v1_rows.append(bra_idx)
                v1_cols.append(ket_idx)
                v1_values.append(elmt)


    v1s.append((particle, torch.sparse_coo_tensor([v1_rows, v1_cols], v1_values, size=(no_states, no_states), dtype=torch.double)))

start_pe_diff = time.perf_counter()
time_pe_like = start_pe_diff - start_pe_like

print(f"[X] Potential energy mtx elements for like particles constructed in {time_pe_like:.1f}s")
print('[ ] Constructing potential energy mtx elements between different particles...', end='\r')

v2s = []

# Two-body interactions between particles of the SAME type. TODO: implement boson interactions with their different exchange behavior
for symb1, symb2 in itertools.combinations(particles, 2):
    particle1 = particles[symb1]
    particle2 = particles[symb2]

    no_states1 = particle1['no_states']
    no_states2 = particle2['no_states']

    b1 = particle1['no_spatial_orbitals'] # number of spatial basis functions for each particle
    b2 = particle2['no_spatial_orbitals']

    # Get two body integrals (coulomb force). Have to combine the bases and then select only the integrals between the two particles
    cmb_int = electron_repulsion_integral(particle1['basis'] + particle2['basis'], transform=block_diag(particle1['transform'], particle2['transform']), notation='chemist')[0:b1, 0:b1, b1:b1+b2, b1:b1+b2]
    cmb_int *= particle1['properties']['charge'] * particle2['properties']['charge']
        #particle['cmb_int'] = cmb_int # Store them

    # Get the spin orbital integrals
    cmb_int_spin = IntegralSpinWrapper(cmb_int, (particle1['properties']['spin'], particle2['properties']['spin']))

    v2 = ((particle1, particle2), [])

    v2_bra1s = []
    v2_bra2s = []
    v2_ket1s = []
    v2_ket2s = []
    v2_values = []

    # For all N-particle bras, use Slater-Condon rules to calculate matrix elements
    for bra1_idx in range(no_states1):
        for bra2_idx in range(no_states2):
            # Matrix elements for states that differ by 0 one-particle states for both bras
            for ket1_idx, common1, deoccupy1, occupy1, parity1 in get_diff_states(particle1, bra1_idx, 0):
                for ket2_idx, common2, deoccupy2, occupy2, parity2 in get_diff_states(particle2, bra2_idx, 0):
                    elmt = sum([cmb_int_spin[i,i,j,j] for i, j in itertools.product(common1, common2)])
                    elmt *= parity1 * parity2

                    #v_mtx += construct_2_particle_interaction(particle1, particle2, bra1_idx, ket1_idx, bra2_idx, ket2_idx, elmt)
                    if abs(elmt) > mtx_elmt_threshold:
                        v2_bra1s.append(bra1_idx)
                        v2_bra2s.append(bra2_idx)
                        v2_ket1s.append(ket1_idx)
                        v2_ket2s.append(ket2_idx)
                        v2_values.append(elmt)

            # States that differ by 0 for bra1, 1 for bra2
            for ket1_idx, common1, deoccupy1, occupy1, parity1 in get_diff_states(particle1, bra1_idx, 0):
                for ket2_idx, common2, deoccupy2, occupy2, parity2 in get_diff_states(particle2, bra2_idx, 1):
                    elmt = sum([cmb_int_spin[i,i,deoccupy2[0],occupy2[0]] for i in common1])
                    elmt *= parity1 * parity2

                    #v_mtx += construct_2_particle_interaction(particle1, particle2, bra1_idx, ket1_idx, bra2_idx, ket2_idx, elmt)
                    if abs(elmt) > mtx_elmt_threshold:
                        v2_bra1s.append(bra1_idx)
                        v2_bra2s.append(bra2_idx)
                        v2_ket1s.append(ket1_idx)
                        v2_ket2s.append(ket2_idx)
                        v2_values.append(elmt)

            # States that differ by 1 for bra1, 0 for bra2
            for ket1_idx, common1, deoccupy1, occupy1, parity1 in get_diff_states(particle1, bra1_idx, 1):
                for ket2_idx, common2, deoccupy2, occupy2, parity2 in get_diff_states(particle2, bra2_idx, 0):
                    elmt = sum([cmb_int_spin[deoccupy1[0],occupy1[0],j,j] for j in common2])
                    elmt *= parity1 * parity2

                    #v_mtx += construct_2_particle_interaction(particle1, particle2, bra1_idx, ket1_idx, bra2_idx, ket2_idx, elmt)
                    if abs(elmt) > mtx_elmt_threshold:
                        v2_bra1s.append(bra1_idx)
                        v2_bra2s.append(bra2_idx)
                        v2_ket1s.append(ket1_idx)
                        v2_ket2s.append(ket2_idx)
                        v2_values.append(elmt)

            # States that differ by 1 for bra1 and bra2
            for ket1_idx, common1, deoccupy1, occupy1, parity1 in get_diff_states(particle1, bra1_idx, 1):
                for ket2_idx, common2, deoccupy2, occupy2, parity2 in get_diff_states(particle2, bra2_idx, 1):
                    elmt = cmb_int_spin[deoccupy1[0],occupy1[0],deoccupy2[0],occupy2[0]]
                    elmt *= parity1 * parity2

                    #v_mtx += construct_2_particle_interaction(particle1, particle2, bra1_idx, ket1_idx, bra2_idx, ket2_idx, elmt)
                    if abs(elmt) > mtx_elmt_threshold:
                        v2_bra1s.append(bra1_idx)
                        v2_bra2s.append(bra2_idx)
                        v2_ket1s.append(ket1_idx)
                        v2_ket2s.append(ket2_idx)
                        v2_values.append(elmt)



    v2s.append(((particle1, particle2), torch.sparse_coo_tensor([v2_bra1s, v2_ket1s, v2_bra2s, v2_ket2s], v2_values, size=(no_states1, no_states1, no_states2, no_states2), dtype=torch.double)))

start_diag = time.perf_counter()
time_pe_diff = start_diag - start_pe_diff

print(f"[X] Potential energy mtx elements for different particles constructed in {time_pe_diff:.1f}s")
print('[ ] Iterative diagonalization...', end='\r')

no_statess = tuple([particles[symb]['no_states'] for symb in particles])


def matvec(v):
    vn = torch.tensor(v.reshape(no_statess, order='F'), dtype=torch.double).to(device)
    vr = torch.zeros(no_statess).to(device)

    for particle, mtx in t1s:
        idx = particle['idx']

        a = torch.tensordot(vn, mtx.to_dense(), dims=([idx], [1]))
        permute = np.concatenate((np.arange(idx), [particle_types - 1], np.arange(idx, particle_types - 1)))

        a = np.transpose(a, tuple(permute))

        vr += a

    for particle, mtx in v1s:
        idx = particle['idx']

        a = torch.tensordot(vn, mtx.to_dense(), dims=([idx], [1]))
        permute = np.concatenate((np.arange(idx), [particle_types - 1], np.arange(idx, particle_types - 1)))

        a = np.transpose(a, tuple(permute))

        vr += a

    for particles, tensor in v2s:
        idx1 = particles[0]['idx']
        idx2 = particles[1]['idx']

        a = torch.tensordot(vn, tensor.to_dense(), dims=([idx1, idx2], [1, 3]))
        permute = np.arange(particle_types - 2)
        permute = np.insert(permute, idx1, particle_types - 2)
        permute = np.insert(permute, idx2, particle_types - 1)

        a = torch.permute(a, tuple(permute))

        vr += a

    return vr.numpy().reshape(total_states, order='F')

import scipy as sp
#h_mtx = t_mtx + v_mtx
#h_eigvals, h_eigvecs = sp.sparse.linalg.eigsh(h_mtx, k=20, which='SA')

from scipy.sparse.linalg import LinearOperator
A = LinearOperator(shape=(total_states, total_states), matvec=matvec, dtype=float)

a_eigvals, a_eigvecs = sp.sparse.linalg.eigsh(A, k=25, which='SA', tol=1e-6, maxiter=250)

end_diag = time.perf_counter()
time_diag = end_diag - start_diag

print(f"[X] Iterative diagonalization completed in {time_diag:.1f}s")

print(a_eigvals)