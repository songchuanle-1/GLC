"""Full model targets using a separately obtained and locally integrated MVGGT."""

from mvggt.models.variants.m93_rpm.model import MVGGT_M93_RPM

from .fusion import install_plain_fusion

from .interventions import (
    install_no_composed_write,
    install_no_token_write,
    install_no_visual_read,
    install_self_only_composition,
    install_shared_visual_read,
)


class MVGGT_M93_SharedVisualRead(MVGGT_M93_RPM):
    """Average visual evidence only; preserve each token's text and state."""

    innovation_control = 'shared_visual_read'
    innovation_condition = innovation_control

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        install_shared_visual_read(self)


class MVGGT_M93_SelfOnlyComposition(MVGGT_M93_RPM):
    """Keep tokenwise pair transforms and both writes without j-dependent reads."""

    innovation_control = 'self_only_composition'
    innovation_condition = innovation_control

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        install_self_only_composition(self)


class MVGGT_M93_NoVisualRead(MVGGT_M93_RPM):
    """Remove only the added visual evidence, retaining its executed branch."""

    innovation_control = 'no_visual_read'
    innovation_condition = innovation_control

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        install_no_visual_read(self)


class MVGGT_M93_NoTokenWrite(MVGGT_M93_RPM):
    """Keep memory computation; suppress only the first visual residual."""

    innovation_control = 'no_token_write'
    innovation_condition = innovation_control

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        install_no_token_write(self)


class MVGGT_M93_NoComposedWrite(MVGGT_M93_RPM):
    """Keep composed carry; suppress only the second visual residual."""

    innovation_control = 'no_composed_write'
    innovation_condition = innovation_control

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        install_no_composed_write(self)


class MVGGT_M93_PlainFusion(MVGGT_M93_RPM):
    """Full ordinary fusion control; no recurrent state or explicit pair MLP."""

    innovation_control = 'plain_fusion'
    innovation_condition = innovation_control

    def __init__(self, *args, plain_fusion_seed=1111, **kwargs):
        super().__init__(*args, **kwargs)
        install_plain_fusion(self, initialization_seed=plain_fusion_seed)

