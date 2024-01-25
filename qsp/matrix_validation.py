from collections.abc import Sequence
from typing import Union
from functools import partial, wraps
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
np.set_printoptions(threshold=np.inf)
from scipy.linalg import expm
twopi = 2. * np.pi


class SingleParticlePhiQuartic:
    def __init__(
        self,
        num_spatial_dims,
        num_registers,
        num_abs_mode_bits,
        lattice_spacing,
        mass,
        coupling
    ):
        self.num_spatial_dims = num_spatial_dims
        self.num_registers = num_registers
        self.num_abs_mode_bits = num_abs_mode_bits
        self.lattice_spacing = lattice_spacing
        self.mass = mass
        self.coupling = coupling

    @property
    def register_size(self):
        return (self.num_abs_mode_bits + 1) * self.num_spatial_dims + 1

    @property
    def register_hilbert_dim(self):
        return 2 ** self.register_size

    @property
    def num_data_bits(self):
        return self.register_size * self.num_registers

    @property
    def data_hilbert_dim(self):
        return 2 ** self.num_data_bits

    @property
    def num_abs_linear_modes(self):
        return 2 ** self.num_abs_mode_bits

    @property
    def num_linear_modes(self):
        return self.num_abs_linear_modes * 2

    @property
    def num_modes(self):
        return self.num_linear_modes ** self.num_spatial_dims

    @property
    def lattice_extent(self):
        return self.num_modes * self.lattice_spacing

    @property
    def lattice_volume(self):
        return self.lattice_extent ** self.num_spatial_dims

    def fock_vacuum(self):
        """Return state [1, 0, ...]."""
        return jax.nn.one_hot(0, 2 ** self.num_data_bits)

    def momentum_indices(self):
        """Return a generator of momentum index tuples."""
        return zip(
            *np.unravel_index(
                np.arange(self.num_modes),
                (self.num_linear_modes,) * self.num_spatial_dims
            )
        )

    def negative_momentum_index(self, q_idx):
        """Return the index of the sign-inverted momentum of the given index."""
        q_idx = np.asarray(q_idx)
        q_idx += self.num_abs_linear_modes
        q_idx %= self.num_linear_modes
        if len(q_idx.shape) == 0:
            return int(q_idx)
        else:
            return tuple(q_idx)

    def norm_momentum(self, q_idx):
        """Return the momentum value without the 2π/L factor."""
        # sign bit = 0/1 -> -/+
        # value bits 0..0 (min) -> 1..1 (max)
        # example (1+1 bit)
        # idx   0    1    2    3
        # q  -1/2 -3/2  1/2  3/2
        q_idx = np.asarray(q_idx)
        abs_q = ((q_idx % self.num_abs_linear_modes) + 0.5)
        return np.where(q_idx >= self.num_abs_linear_modes, 1., -1.) * abs_q

    def momentum(self, q_idx):
        """Return the momentum value."""
        return self.norm_momentum(q_idx) * twopi / self.lattice_extent

    def energy(self, q_idx):
        """Return the energy value."""
        return np.sqrt(
            np.sum(np.square(self.momentum(q_idx)))
            + self.mass ** 2
        )

    def identity_op(self):
        """Return the identity operator for the whole system."""
        return jnp.eye(2 ** self.num_data_bits, dtype=complex)

    def zero_op(self):
        """Return the zero operator for the whole system."""
        dim = 2 ** self.num_data_bits
        return jnp.zeros((dim, dim), dtype=complex)

    def momentum_creation_op(self, q_idx):
        """Return a_q^†."""
        return self._momentum_a_op(True, q_idx)

    def momentum_annihilation_op(self, q_idx):
        """Return a_q."""
        return self._momentum_a_op(False, q_idx)

    def _momentum_a_op(self, is_creation, q_idx):
        """Creation / annihilation operator backend."""
        dim = 2 ** self.num_data_bits
        op = self.zero_op()
        for ireg in range(self.num_registers):
            op += self._register_momentum_a_op(is_creation, ireg, q_idx)
        return op / np.sqrt(self.num_registers)

    def _register_momentum_a_op(self, is_creation, ireg, q_idx):
        """Full-system op with a creation / annihilation operator for one register."""
        return self._apply_local_op(
            self._local_momentum_a_op(is_creation, q_idx),
            self.identity_op(),
            ireg
        )

    def _local_momentum_a_op(self, is_creation, q_idx):
        """Register-local creation / annihilation operator."""
        # creation: state |000> -> |qs1>
        reg_dim = 2 ** self.register_size
        local_op = np.zeros((reg_dim, reg_dim))
        if not isinstance(q_idx, int):
            q_idx = np.ravel_multi_index(q_idx, (self.num_modes,) * self.num_spatial_dims)
        idx = 2 ** (self.register_size - 1) + q_idx
        if is_creation:
            local_op[idx, 0] = 1.
        else:
            local_op[0, idx] = 1.
        return jnp.array(local_op)

    def squeezing_op_expsum(self):
        """Squeezing operator of form exp[sum{-z_q (a+a+ - aa)}]."""
        exponent = self.zero_op()

        for q_idx in self.momentum_indices():
            negative_q_idx = self.negative_momentum_index(q_idx)
            exponent += -0.5 * np.log(self.energy(q_idx)) * (
                self.momentum_creation_op(q_idx)
                @ self.momentum_creation_op(negative_q_idx)
                -
                self.momentum_annihilation_op(negative_q_idx)
                @ self.momentum_annihilation_op(q_idx)
            )

        return jax.scipy.linalg.expm(exponent)

    def squeezing_op_prodexp(self):
        prod = self.identity_op()
        for q_idx in self.momentum_indices():
            negative_q_idx = self.negative_momentum_index(q_idx)
            prod = jax.scipy.linalg.expm(
                -0.5 * np.log(self.energy(q_idx)) * (
                    self.momentum_creation_op(q_idx)
                    @ self.momentum_creation_op(negative_q_idx)
                    -
                    self.momentum_annihilation_op(negative_q_idx)
                    @ self.momentum_annihilation_op(q_idx)
                )
            ) @ prod

        return prod

    def squeezing_op_prodexp_trotter(self):
        prod = self.identity_op()
        for q_idx in self.momentum_indices():
            negative_q_idx = self.negative_momentum_index(q_idx)
            zq_over_M = 0.5 * np.log(self.energy(q_idx)) / self.num_registers
            for ireg1 in range(self.num_registers):
                for ireg2 in range(self.num_registers):
                    if ireg1 == ireg2:
                        continue

                    exponent_cr = self.identity_op()
                    for iq, ireg in [(q_idx, ireg1), (negative_q_idx, ireg2)]:
                        exponent_cr = self._apply_local_op(
                            self._local_momentum_a_op(True, iq),
                            exponent_cr,
                            ireg
                        )
                    exponent_an = self.identity_op()
                    for iq, ireg in [(negative_q_idx, ireg2), (q_idx, ireg1)]:
                        exponent_cr = self._apply_local_op(
                            self._local_momentum_a_op(True, iq),
                            exponent_cr,
                            ireg
                        )
                    prod = jax.scipy.linalg.expm(
                        -zq_over_M * (exponent_cr - exponent_an)
                    ) @ prod

        return prod

    def momentum_basis_change_op(self):
        # idx   0    1    2    3
        # q  -1/2 -3/2  1/2  3/2
        # ↓
        # q   1/2  3/2 -3/2 -1/2
        change_op = self.identity_op()
        for ireg in range(self.num_registers):
            change_op = self._apply_local_op(
                self._momentum_basis_change_local_op(),
                change_op,
                ireg
            )
        return change_op

    def _momentum_basis_change_local_op(self):
        # Construct the basis change op for the "occupied" block of the register
        block_dim = self.register_hilbert_dim // 2
        block_op = np.eye(block_dim, dtype=int)
        nal = self.num_abs_linear_modes
        shape = (2 * nal,) * (2 * self.num_spatial_dims)
        block_op = np.reshape(block_op, shape)
        for idim in range(self.num_spatial_dims):
            flip_op = np.zeros((2 * nal, 2 * nal), dtype=int)
            flip_op[
                np.arange(nal),
                np.arange(nal, 2 * nal)
            ] = 1
            flip_op[
                np.arange(nal, 2 * nal),
                np.arange(nal - 1, -1, -1)
            ] = 1
            max_axes = 2 * self.num_spatial_dims
            block_op = np.einsum(
                block_op, list(range(max_axes)),
                flip_op, [max_axes, idim],
                list(range(idim)) + [max_axes] + list(range(idim + 1, max_axes))
            )

        zeros = np.zeros((block_dim, block_dim), dtype=int)
        return np.concatenate([
            np.concatenate([np.eye(block_dim, dtype=int), zeros], axis=1),
            np.concatenate([zeros, block_op], axis=1)
        ], axis=0)

    def _apply_local_op(
        self,
        op: 'array',
        base: 'array',
        registers: Union[int, Sequence[int]]
    ):
        """Left-multiply a register-local operation on base.

        Args:
            op: Square matrix of a register-local operation.
            base: Square matrix of a global operation.
            registers: Indices of registers op applies to.
        """
        if isinstance(registers, int):
            registers = [registers]
        else:
            registers = list(registers)
        # Number of registers op applies to
        locality = len(registers)
        assert np.prod(op.shape) == self.register_hilbert_dim ** (2 * locality)
        # Total number of tensor axes to cast base to
        num_axes = self.num_registers * 2
        # Reshape the base to one dim per register
        base = base.reshape((self.register_hilbert_dim,) * num_axes)
        # Reshape the op similarly
        op = op.reshape((self.register_hilbert_dim,) * (2 * locality))
        # Assign indices to each axis of op and base
        # base indices: [0, ..., r0, ..., r1, ..., 2M-1]
        # op indices: [2M+r0, 2N+r1, ..., r0, r1, ...]
        # output indices: [0, ..., 2M+r0, ..., 2M+r1, ..., 2M-1]
        base_sublist = list(range(num_axes))
        op_sublist = (np.arange(locality) + num_axes).tolist() + registers
        output_sublist = list(base_sublist)
        for ireg, reg_idx in enumerate(registers):
            output_sublist[reg_idx] = op_sublist[ireg]
        applied = jnp.einsum(base, base_sublist, op, op_sublist, output_sublist)
        # Finally reshape the output to a matrix
        dim = self.data_hilbert_dim
        return jnp.reshape(applied, (dim, dim))

    def interpret_state(self, state):
        state = np.asarray(state)
        state_reprs = []
        for idx in np.nonzero(state)[0]:
            amp = state[idx]
            reg_states = []
            for ireg in range(self.num_registers):
                comp_basis = (idx >> (ireg * self.register_size)) % (2 ** self.register_size)
                occupancy = comp_basis >> (self.register_size - 1)
                q_idx = comp_basis % (2 ** (self.register_size - 1))
                if occupancy == 0 and q_idx == 0:
                    reg_states.append('u NULL')
                    continue

                momentum = []
                for idim in range(self.num_spatial_dims):
                    dim_q_idx = (q_idx >> (idim * (self.num_abs_mode_bits + 1))) % self.num_linear_modes
                    momentum.append(self.norm_momentum(dim_q_idx))
                occ = 'o' if occupancy else 'u'
                reg_states.append(f'{occ} {tuple(momentum)}')
            state_reprs.append((amp, reg_states))
        return state_reprs
