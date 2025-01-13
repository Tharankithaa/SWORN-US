from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import os

# Define the local directory to save the model and processor
LOCAL_MODEL_DIR = "./trocr_base_handwritten"

# Check if the directory exists; if not, download the model and processor
if not os.path.exists(LOCAL_MODEL_DIR):
    os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)
    print("Downloading the model and processor...")

    # Download the processor
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
    processor.save_pretrained(LOCAL_MODEL_DIR)

    # Download the model
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
    model.save_pretrained(LOCAL_MODEL_DIR)

    print(f"Model and processor downloaded and saved to {LOCAL_MODEL_DIR}")
else:
    print(f"Model and processor already exist in {LOCAL_MODEL_DIR}")
