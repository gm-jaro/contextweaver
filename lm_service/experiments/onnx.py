import os
import onnx


mobile_model_path = os.path.expanduser("~/.cache/modelscope/hub/models/RapidAI/RapidOCR/onnx/PP-OCRv5/det/ch_PP-OCRv5_server_det.onnx")
server_model_path = "models/Paddle/PP-OCRv5_server_det_infer.onnx"

mobile_model = onnx.load(mobile_model_path)
server_model = onnx.load(server_model_path)

onnx.checker.check_model(mobile_model)
onnx.checker.check_model(server_model)


def print_model_info(model, name):
    print(f"=== {name} ===")
    print(f"Inputs: {[ (i.name, i.type if i.type else None) for i in model.graph.input ]}")
    print(f"Outputs: {[ (o.name, o.type if o.type else None) for o in model.graph.output ]}")
    print(f"Initializers: {[ init.name for init in model.graph.initializer ]}")
    print(f"Nodes: {[ n.op_type for n in model.graph.node[:10] ]} ... total {len(model.graph.node)} nodes\n")


print_model_info(mobile_model, "Their Model")
print_model_info(server_model, "My Model")
