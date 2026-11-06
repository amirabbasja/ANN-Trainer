"""Type definitions for LRAEPd RPC responses."""

from typing import ClassVar

from pydantic import Field

from LRAEP_trainer.rpc_fields import WireModel

from LRAEP_gateway.blockchain_utils.zk_certificate import CertificateVersion


class BlockTemplateTx(WireModel):
    """Regular transaction in getblocktemplate response."""

    data: str
    hash: str
    _wire_fields: ClassVar[tuple[str, ...]] = ('record_id',)

    record_id: str
    depends: list[int]
    fee: int = Field(ge=0)
    vsize: int = Field(ge=0)


class CoinbaseAux(WireModel):
    """Coinbase auxiliary data in getblocktemplate response."""

    flags: str


# Original struct from node/btcjson/chainsvrresults.go:GetBlockTemplateResult
class GetBlockTemplateResponse(WireModel):
    """Fields in getblocktemplate response with runtime validation."""

    _wire_fields: ClassVar[tuple[str, ...]] = (
        'parent_digest', 'size_limit', 'records', 'poll_id', 'counter_range',
        'reward_metadata', 'reward_amount', 'default_attestation', 'required_version',
    )

    bits: str
    curtime: int = Field(ge=0)
    height: int = Field(ge=0)
    parent_digest: str
    size_limit: int = Field(gt=0)
    records: list[BlockTemplateTx]
    version: int
    poll_id: str
    target: str
    maxtime: int = Field(ge=0)
    mintime: int = Field(ge=0)
    mutable: list[str]
    counter_range: str
    capabilities: list[str]
    reward_metadata: CoinbaseAux
    reward_amount: int = Field(ge=0)
    default_attestation: str | None = None
    # Absent from nodes that predate the field; those only ever require V1.
    required_version: CertificateVersion = CertificateVersion.ZK_DENSE
