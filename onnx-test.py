
import sys

from classify import load_model, IMAGE_SIZE, classes
from src.config import USE_ALT_CLASSES

suffix = ""
if USE_ALT_CLASSES:
    suffix = "-" + USE_ALT_CLASSES
MODEL_FILE = f"trained_models/whole_model_quickdraw{suffix}.onnx"
CLASSES_FILE = f"classes{suffix}.json"


def convert_model():
    import torch

    print(f"Exporting model")

    print(f"Loading existing model...")
    model = load_model()
    example_input = torch.randn(1, 1, IMAGE_SIZE, IMAGE_SIZE)

    #MODEL_FILE = "trained_models/whole_model_quickdraw.pt"
    #traced_script_module = torch.jit.trace(model, example_input)
    #traced_script_module.save(MODEL_FILE)

    #pip install onnxscript
    onnx_program = torch.onnx.export(model, example_input, dynamo=True)
    onnx_program.save(MODEL_FILE)

    print(f"Saved model to: {MODEL_FILE}")
    # Can visualize the model at: https://netron.app/

    # Save classes
    list_classes = classes()
    import json
    import os
    existing_data = {}
    if os.path.exists(CLASSES_FILE):
        print(f"Updating existing classes file: {CLASSES_FILE}")
        try:
            with open(CLASSES_FILE, 'r') as f:
                existing_data = json.load(f)
        except FileNotFoundError:
            pass
    
    # Update classes
    existing_data['classes'] = list_classes

    # Save file
    with open(CLASSES_FILE, 'w') as f:
        json.dump(existing_data, f, indent=4)
        print(f"Saving: {CLASSES_FILE}")

    print(f"Loading and checking model: {MODEL_FILE}")
    import onnx
    onnx_model = onnx.load(MODEL_FILE)
    onnx.checker.check_model(onnx_model)
    print("Done.")

    # List inputs and outputs
    for idx, input_tensor in enumerate(onnx_model.graph.input):
        print(f"Input {idx}: {input_tensor.name}, shape: {[dim.dim_value for dim in input_tensor.type.tensor_type.shape.dim]}")
    for idx, output_tensor in enumerate(onnx_model.graph.output):
        print(f"Output {idx}: {output_tensor.name}, shape: {[dim.dim_value for dim in output_tensor.type.tensor_type.shape.dim]}")


def test():
    #pip install onnxruntime

    import torch
    example_inputs = [torch.randn(1, 1, IMAGE_SIZE, IMAGE_SIZE)]
    onnx_inputs = [tensor.numpy(force=True) for tensor in example_inputs]
    print(f"Input length: {len(onnx_inputs)}")
    #print(f"Sample input: {onnx_inputs}")

    import onnxruntime
    ort_session = onnxruntime.InferenceSession(MODEL_FILE, providers=["CPUExecutionProvider"])
    onnxruntime_input = {input_arg.name: input_value for input_arg, input_value in zip(ort_session.get_inputs(), onnx_inputs)}

    # ONNX Runtime returns a list of outputs
    print(f"Running ONNX Runtime inference...")
    onnxruntime_outputs = ort_session.run(None, onnxruntime_input)[0]
    print(f"ONNX Runtime output: {onnxruntime_outputs}")


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "export":
        convert_model()
    elif len(sys.argv) == 2 and sys.argv[1] == "test":
        test()
    else:
        print("Usage: python onnx-test.py <export|test>")


