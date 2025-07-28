import torch
from torchvision import transforms
from PIL import Image
import os
from model import CNN_ViT_UnderwaterRestorer

def load_model(model_path, weights_only=False):
    model = CNN_ViT_UnderwaterRestorer()
    if weights_only:
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=True)
        model.load_state_dict(checkpoint)
    else:
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        model = checkpoint  # already contains architecture
    model.eval()
    return model

def preprocess_image(image_path):
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),  # [0, 1]
        transforms.Normalize([0.5]*3, [0.5]*3)  # → [-1, 1]
    ])
    return transform(image).unsqueeze(0)  # [1, 3, 224, 224]

def postprocess_tensor(tensor):
    tensor = tensor.squeeze(0).detach().cpu()  # [3, 224, 224]
    tensor = tensor.clamp(0, 1)  # Ensure range
    return transforms.ToPILImage()(tensor)

def restore_image(image_path, model_path, output_path, weights_only=False):
    model = load_model(model_path, weights_only=weights_only)
    input_tensor = preprocess_image(image_path)
    with torch.no_grad():
        output_tensor = model(input_tensor)
    output_image = postprocess_tensor(output_tensor)
    output_image.save(output_path)
    print(f"✅ Saved restored image at: {output_path}")

if __name__ == "__main__":
    input_image_path = "Blurry-and-Bluish-Effects-from-Haze-and-Color-cast-in-Underwater-Images.png"  # Input
    model_file = r"underwater_model_full.pth"  # 🔁 or 'underwater_model_weights.pth'
    output_image_path = "restored_output.jpg"

    # Choose True if using state_dict file
    restore_image(input_image_path, model_file, output_image_path, weights_only=False)
