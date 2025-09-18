import torch
from model import GPT  
import os

modelname="gpt2"

def create_folder_if_not_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder '{folder_path}' created.")
    else:
        print(f"Folder '{folder_path}' already exists.")


folder_path = 'src/utils/model/params_output'
create_folder_if_not_exists(folder_path)


class wrapper(torch.nn.Module):
    def __init__(self, model):
        super(wrapper, self).__init__()
        self.model = model
        self.layer_num = self.model.config.n_layer  
        self.head_num = self.model.config.n_head   

    def forward(self, input):
        outputs = self.model(input)
        
        extracted_outputs = []
        for i in range(self.layer_num):
            for j in range(self.head_num):
                extracted_outputs.extend([
                    outputs["block"][f"block_{i}"]["attn"][f"head_{j}"]["attn"],
                    outputs["block"][f"block_{i}"]["attn"][f"head_{j}"]["attn_scaled"],
                    outputs["block"][f"block_{i}"]["attn"][f"head_{j}"]["attn_masked"],
                    outputs["block"][f"block_{i}"]["attn"][f"head_{j}"]["attn_softmax"],
                    outputs["block"][f"block_{i}"]["attn"][f"head_{j}"]["attn_dropout"]
                ])
        extracted_outputs.append(outputs["linear"]["output"])
        return tuple(extracted_outputs)


model = GPT.from_pretrained(modelname)
model.eval()
wrapped_model = wrapper(model)

dummy_input = torch.tensor([[6601, 32704, 795, 30132, 2985, 284]])


onnx_model_path = "src/utils/model/params_output/"+ modelname +".onnx"


output_names = [
    f"block_{i}_attn_head_{j}_{suffix}"
    for i in range(model.config.n_layer) 
    for j in range(model.config.n_head) 
    for suffix in ["attn", "attn_scaled", "attn_masked", "attn_softmax", "attn_dropout"]
]
output_names.append("linear_output")

torch.onnx.export(
    wrapped_model,
    dummy_input,
    onnx_model_path,
    export_params=True,
    opset_version=11,
    do_constant_folding=True,
    input_names=["input"],
    output_names=output_names,
    dynamic_axes={
        'input': {0: '0', 1: '1'},
        **{
            name: {0: '0', 1: '1', 2: '2'}
            for name in output_names if name != "linear_output"
        },
        'linear_output': {0: '0', 1: '1', 2: '2'}
    }
)

print("Model has been successfully exported to ONNX format.")

