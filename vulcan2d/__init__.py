"""VULCAN-2D research package with an audited quasi-static handoff domain."""
from .model import Params, simulate_cycles, divider, i_hbn, i_tr
from .features import extract_all, summary
from .model_v7 import run_identified_reset
from .model_v8 import (Params as ResetParams, estimate_initial_lrs,
                       run_read_initialized_reset)
from .cycle_link import (ResetEndpointAttractorLaw, SetResetTransferLaw,
                         fit_reset_endpoint_attractor,
                         fit_set_reset_transfer, run_set_conditioned_cycle,
                         set_forward_exposure)
from .hrs_transport import (EffectivePooleFrenkelLaw,
                            effective_pf_basis,
                            fit_effective_pf_law)
from .model_v11 import (Params as RefinedResetParams,
                        run_effective_pf_reset,
                        run_refined_set_conditioned_cycle)
from .set_identification import (SetTransitionLaw, SetTransportLaw,
                                 fit_set_transition_law,
                                 fit_set_transport_law,
                                 reconstruct_set)
from .reset_identification import (PFResetTransitionLaw,
                                   fit_pf_reset_transition_law,
                                   identify_pf_reset_state)
from .model_v12 import (AutonomousCalibration, SetResetLinkLaw,
                        default_calibration, fit_set_reset_link_law,
                        run_autonomous_set)
from .model_v13 import (HandoffDomainError, run_autonomous_full_cycle,
                        simulate_calibrated_cycle)
from .transistor import (TransistorLookup, TransistorMapping,
                         load_default_lookup, load_reverse_lookup)

__version__ = "0.13.0"

__all__ = ["Params", "simulate_cycles", "divider", "i_hbn", "i_tr",
           "TransistorLookup", "TransistorMapping", "load_default_lookup",
           "load_reverse_lookup",
           "ResetParams", "run_identified_reset", "estimate_initial_lrs",
           "run_read_initialized_reset",
           "SetResetTransferLaw", "set_forward_exposure",
           "fit_set_reset_transfer", "run_set_conditioned_cycle",
           "ResetEndpointAttractorLaw", "fit_reset_endpoint_attractor",
           "EffectivePooleFrenkelLaw", "effective_pf_basis",
           "fit_effective_pf_law", "RefinedResetParams",
           "run_effective_pf_reset", "run_refined_set_conditioned_cycle",
           "SetTransportLaw", "SetTransitionLaw", "reconstruct_set",
           "fit_set_transport_law", "fit_set_transition_law",
           "SetResetLinkLaw", "fit_set_reset_link_law",
           "run_autonomous_set", "run_autonomous_full_cycle",
           "AutonomousCalibration", "default_calibration",
           "simulate_calibrated_cycle",
           "HandoffDomainError",
           "PFResetTransitionLaw", "identify_pf_reset_state",
           "fit_pf_reset_transition_law",
           "extract_all", "summary", "__version__"]
