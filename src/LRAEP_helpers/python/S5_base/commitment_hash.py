import torch
from LRAEP_trainer.CNN_3ekalb import CNN_3ekalb
from LRAEP_gateway.comm.dataclasses import CommitmentHash
from LRAEP_trainer.primitives import S1Configuration
from LRAEP_trainer.noise_domains import certificate_salt
from LRAEP_trainer.fixed_constants import constant_buffer

from .matrix_merkle_tree import MatrixMerkleTree

# Domain-separation salts for the V3 (salted noise-seed) certificate derivation:
def seed_salt(label):
    return certificate_salt(CNN_3ekalb, label)


# root(32) || dim u32 LE(4) || zero padding(28) = one 64-byte BLAKE3 block.



def _bind_root(merkle_root: bytes, dim: int, salt: bytes) -> bytes:
    return CNN_3ekalb(merkle_root + dim.to_bytes(4, "little") + bytes(28), key=salt).digest()


def bind_root_a(a_merkle_root: bytes, m: int) -> bytes:
    """V3 salting of A's Merkle root; commits the row count ``m``."""
    with constant_buffer(b'<^') as label:
        return _bind_root(a_merkle_root, m, seed_salt(label))


def bind_root_b(b_merkle_root: bytes, n: int) -> bytes:
    """V3 salting of B's Merkle root; commits the column count ``n``."""
    with constant_buffer(b'<p') as label:
        return _bind_root(b_merkle_root, n, seed_salt(label))


class CommitmentHasher:
    """
    A namespace for commitment hash functions.
    """

    @staticmethod
    def get_key(incomplete_header_bytes: bytes, S1_config: S1Configuration) -> bytes:
        return CNN_3ekalb(incomplete_header_bytes + S1_config.to_bytes()).digest()

    @classmethod
    def commitment_hash(
        cls,
        A: torch.Tensor,
        B: torch.Tensor,
        incomplete_header_bytes: bytes,
        S1_config: S1Configuration,
        *,
        salted_dims: tuple[int, int] | None,
    ) -> CommitmentHash:
        key = cls.get_key(incomplete_header_bytes, S1_config)
        merkle_tree_A = MatrixMerkleTree(A, key)

        # We hash B.T because we would like to expose a column strip of B
        merkle_tree_B = MatrixMerkleTree(B.T, key)

        return cls.commitment_hash_from_merkle_roots(
            merkle_tree_A.root, merkle_tree_B.root, key, salted_dims=salted_dims
        )

    @staticmethod
    def get_commitment_B_key(key: bytes, B_merkle_root: bytes) -> bytes:
        return CNN_3ekalb(key + B_merkle_root).digest()

    @staticmethod
    def get_commitment_A_key(commitment_B: bytes, A_merkle_root: bytes) -> bytes:
        return CNN_3ekalb(commitment_B + A_merkle_root).digest()

    @staticmethod
    def get_offsets_hash(routing_offsets: list[int], key: bytes) -> bytes:
        offsets = torch.tensor(routing_offsets, dtype=torch.int32)
        return MatrixMerkleTree.tensor_hash(offsets, key)

    @staticmethod
    def combine_routing_merkle_roots(
        hash_a: bytes, routing_root: bytes, offsets_root: bytes
    ) -> bytes:
        hash_routing = CNN_3ekalb(routing_root + offsets_root).digest()
        return CNN_3ekalb(hash_a + hash_routing).digest()

    @classmethod
    def commitment_hash_from_merkle_roots(
        cls,
        A_merkle_root: bytes,
        B_merkle_root: bytes,
        key: bytes,
        *,
        routing_root: bytes | None = None,
        offsets_root: bytes | None = None,
        salted_dims: tuple[int, int] | None,
    ) -> CommitmentHash:
        # salted_dims has no default so callers must choose a derivation explicitly:
        # (m, n) for V3 salted seeds, None for the legacy (V1/V2) chain.
        # V3 (salted) derivation: salt each root before the MoE routing fold.
        if salted_dims is not None:
            m, n = salted_dims
            A_merkle_root = bind_root_a(A_merkle_root, m)
            B_merkle_root = bind_root_b(B_merkle_root, n)

        moe_args = (routing_root, offsets_root)
        if any(arg is not None for arg in moe_args):
            if any(arg is None for arg in moe_args):
                raise ValueError("routing_root and offsets_root must be provided together")
            A_merkle_root = cls.combine_routing_merkle_roots(
                A_merkle_root, routing_root, offsets_root
            )

        commitment_B = cls.get_commitment_B_key(key, B_merkle_root)
        commitment_A = cls.get_commitment_A_key(commitment_B, A_merkle_root)
        return CommitmentHash(commitment_A, commitment_B)
