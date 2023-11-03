import tkinter as tk
from tkinter import filedialog, messagebox
import os
from PIL import Image
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch.nn as nn
from torchvision.models import efficientnet_v2_l
import itertools

# Define custom dataset class for loading images
class CustomDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_paths = [os.path.join(image_dir, file) for file in os.listdir(image_dir) if file.endswith(tuple('.png .jpg .jpeg .bmp .tif .tiff'.split()))]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image

# Image transformation
test_transforms = transforms.Compose([
    transforms.Resize((480, 480)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

# Function to perform prediction
def genetic_risk_predict(image_dir, model_path, output_folder):
    dataset = CustomDataset(image_dir=image_dir, transform=test_transforms)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = efficientnet_v2_l(pretrained=False)
    model.classifier = nn.Sequential(nn.Dropout(p=0.5), nn.Linear(1280, 1))

    model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
    model.to(device)
    model.eval()

    predictions, probabilities = [], []
    for inputs in dataloader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        output_prob = torch.sigmoid(outputs)

        # Apply threshold to determine predictions
        threshold = 0.45
        predicted_labels = (output_prob > threshold).float().cpu().detach().numpy()
        predictions.extend(predicted_labels)
        probabilities.extend(output_prob.cpu().detach().numpy())

    image_names = [os.path.basename(path) for path in dataset.image_paths]
    df_output = pd.DataFrame({
        'img_name': image_names,
        'Predict_Prob': list(itertools.chain(*probabilities)),
        'Pred_label': list(itertools.chain(*predictions))
    })
    output_csv = os.path.join(output_folder, 'genetic_risk_output.csv')
    df_output.to_csv(output_csv, index=False)
    return output_csv

# Global variable to store the model path
MODEL_PATH = None

# Function to select the model file
def select_model_file():
    global MODEL_PATH
    model_file_path = filedialog.askopenfilename(
        title="Select Model File",
        filetypes=[("PyTorch Model", "*.pth")],
        initialdir=os.path.expanduser("~")  # Optionally start at user's home directory
    )
    if model_file_path:
        MODEL_PATH = model_file_path
        model_path_label.config(text=f"Model selected: {os.path.basename(MODEL_PATH)}")
    else:
        model_path_label.config(text="No model selected. Please select a model.")
        MODEL_PATH = None  # Reset to None if user cancels file dialog

# GUI part
def select_folder_and_predict():
    directory = filedialog.askdirectory(title="Select the Image Folder")
    output_directory = filedialog.askdirectory(title="Select the Output Folder for CSV")
    if directory and output_directory:
        try:
            csv_file = genetic_risk_predict(directory, MODEL_PATH, output_directory)
            messagebox.showinfo("Success", f"CSV file has been generated: {csv_file}")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {e}")

# Model path
#MODEL_PATH = 'C:\\Users\\poolo\\CEI_projects\\DNA_image_IV\\genetic_risk_1102_AUC85.pth'

# Main GUI window setup
root = tk.Tk()
root.title("Genetic Risk Prediction Dashboard")

# Set window size (width x height)
root.geometry("600x400")

# Button for selecting model file
button_model = tk.Button(root, text="Select Model File", command=select_model_file)
button_model.pack(padx=20, pady=10)

# Label to show selected model path
model_path_label = tk.Label(root, text="No model selected. Please select a model.")
model_path_label.pack(padx=20, pady=10)

# Button for selecting folder and starting prediction
button_predict = tk.Button(root, text="Select Image Folder and Predict", command=select_folder_and_predict)
button_predict.pack(padx=20, pady=20)

# Run the main loop
root.mainloop()