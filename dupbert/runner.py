from catalyst.dl import SupervisedRunner

from .state_keys import StateKeys


class TripletRunner(SupervisedRunner, StateKeys):

    def __init__(self, **kwargs):
        super().__init__(
            input_key=self.input_tuple,
            target_key=self.targets,
            output_key=self.model_output,
            **kwargs
        )

    def forward(self,  batch):
        output = self.model(batch[self.input_tuple])
        output = self._process_output(output)
        return output
