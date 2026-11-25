"""Current paper names for the existing VistaMix implementation.

Legacy safemix identifiers and serialized schemas remain compatible.
"""
from .safemix import (
    SafeMixContract as VistaMixContract,
    combine_safemix_gradients as combine_vistamix_gradients,
    safemix_contract_errors as vistamix_contract_errors,
    validate_safemix_contract as validate_vistamix_contract,
)
FEDVISTA_TITLE = "FedVista: Robust Federated VLA Learning from Diverse Visual Worlds"
