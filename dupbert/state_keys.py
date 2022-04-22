from dataclasses import dataclass


@dataclass()
class StateKeys:
    input_tuple: str = 'input_tuple'
    left_sample: str = 'left_sample'
    right_sample: str = 'right_sample'
    targets: str = 'targets'
    model_output: str = 'logits'
