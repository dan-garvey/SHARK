import requests
import torch
from torchvision import transforms
import sys
from shark.shark_inference import SharkInference
from shark.collectives import nccl_utils


class LinearModule(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.train(False)

    def forward(self, x):
        return torch.mul(x,x)


##############################################################################

input = torch.randn(512)

## The img is passed to determine the input shape.
shark_module = SharkInference(LinearModule(), (input,))
shark_module.compile()

## Can pass any img or input to the forward module.
results = shark_module.forward((input,))

tmlir = torch.from_numpy(results)
golden = LinearModule()(input)
print(torch.allclose(tmlir,golden))
