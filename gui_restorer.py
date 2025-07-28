import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from PIL import Image, ImageTk
import numpy as np
import cv2
import os
from threading import Thread
import time

class CNN_ViT_UnderwaterRestorer(nn.Module):
    def __init__(self):
        super(CNN_ViT_UnderwaterRestorer, self).__init__()
        
        # CNN for local structure
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 3, kernel_size=3, padding=1)
        )
        
        # ViT for global context (with error handling)
        try:
            self.vit = timm.create_model('vit_base_patch16_224.augreg2_in21k_ft_in1k', pretrained=True)
            self.vit.head = nn.Identity()
            print("✅ ViT model loaded successfully")
        except Exception as e:
            print(f"⚠️  Warning: Could not load pretrained ViT ({e}). Using simple CNN instead.")
            # Fallback to simple CNN if ViT fails
            self.vit = nn.Sequential(
                nn.AdaptiveAvgPool2d((7, 7)),
                nn.Flatten(),
                nn.Linear(3 * 7 * 7, 768)
            )
        
        self.vit_decoder = nn.Sequential(
            nn.Linear(768, 1024),
            nn.ReLU(),
            nn.Linear(1024, 224 * 224 * 3),
        )
        
        # Fusion of CNN + ViT
        self.fusion = nn.Sequential(
            nn.Conv2d(6, 3, kernel_size=3, padding=1),
        )
        
        # Color Correction Module (CCM)
        self.color_correction = nn.Sequential(
            nn.Conv2d(3, 3, kernel_size=1),  # Learnable channel mixing
            nn.Tanh()  # Allow negative shifts in color space
        )
    
    def forward(self, x):
        cnn_feat = self.cnn(x)  # [B, 3, 224, 224]
        
        vit_feat = self.vit(x)  # [B, 768]
        vit_out = self.vit_decoder(vit_feat).view(-1, 3, 224, 224)
        
        fused = torch.cat([cnn_feat, vit_out], dim=1)
        out = self.fusion(fused)  # [B, 3, 224, 224]
        
        out = self.color_correction(out)  # Apply CCM
        
        out = torch.sigmoid(out)  # [0,1] range for image output
        
        return out

class UnderwaterRestorationGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("🌊 Underwater Image Restorer")
        self.root.geometry("1200x800")
        self.root.configure(bg='#001122')
        
        # Model variables
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.original_image = None
        self.restored_image = None
        self.processing = False
        self.model_paths = []  # Will store possible model paths
        
        # Find and load model
        self.find_model_files()
        self.load_model()
        
        # Setup UI
        self.setup_styles()
        self.create_widgets()
        
    def find_model_files(self):
        """Find potential model files in various locations"""
        possible_paths = [
            # Current directory
            'underwater_model_full.pth',
            'underwater_model_weights.pth',
            'model.pth',
            'best_model.pth',
            
            # Common subdirectories
            'models/underwater_model_full.pth',
            'models/underwater_model_weights.pth',
            'models/model.pth',
            'weights/underwater_model_full.pth',
            'weights/underwater_model_weights.pth',
            'checkpoints/underwater_model_full.pth',
            'checkpoints/underwater_model_weights.pth',
            
            # User's Documents (original path)
            r'C:\Users\ajite\Documents\marine\cnn+vit\underwater_restoration\underwater_model_full.pth',
            
            # Desktop (common location)
            os.path.expanduser('~/Desktop/underwater_model_full.pth'),
            os.path.expanduser('~/Desktop/underwater_model_weights.pth'),
            
            # Downloads folder
            os.path.expanduser('~/Downloads/underwater_model_full.pth'),
            os.path.expanduser('~/Downloads/underwater_model_weights.pth'),
        ]
        
        self.model_paths = [path for path in possible_paths if os.path.exists(path)]
        print(f"Found {len(self.model_paths)} potential model files:")
        for path in self.model_paths:
            print(f"  📁 {path}")
        
    def load_model(self):
        """Load the underwater restoration model with robust error handling"""
        try:
            # Initialize model architecture
            print("🔧 Initializing model architecture...")
            self.model = CNN_ViT_UnderwaterRestorer()
            model_loaded = False
            
            # Try to load from found model files
            for model_path in self.model_paths:
                try:
                    print(f"🔄 Trying to load: {model_path}")
                    
                    # First, let's inspect the file to understand its structure
                    try:
                        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
                        print(f"📊 File type: {type(checkpoint)}")
                        
                        if isinstance(checkpoint, dict):
                            print(f"📋 Dict keys: {list(checkpoint.keys())}")
                            
                    except Exception as inspect_error:
                        print(f"📊 File inspection failed: {inspect_error}")
                        # Try with weights_only=True as fallback
                        try:
                            checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
                            print(f"📊 Loaded with weights_only=True, type: {type(checkpoint)}")
                        except Exception as weights_only_error:
                            print(f"❌ Both loading methods failed: {weights_only_error}")
                            continue
                    
                    # Handle different save formats
                    if isinstance(checkpoint, CNN_ViT_UnderwaterRestorer):
                        # Full model saved
                        self.model = checkpoint
                        print(f"✅ Loaded full model from {model_path}")
                        model_loaded = True
                        break
                        
                    elif isinstance(checkpoint, dict):
                        # Find the state dict in various possible keys
                        state_dict = None
                        possible_keys = ['model_state_dict', 'state_dict', 'model', 'net']
                        
                        for key in possible_keys:
                            if key in checkpoint:
                                state_dict = checkpoint[key]
                                print(f"📦 Found state dict in key: '{key}'")
                                break
                        
                        if state_dict is None:
                            # If no known keys, assume the whole dict is the state dict
                            state_dict = checkpoint
                            print(f"📦 Using entire dict as state dict")
                        
                        # Try to load the state dict
                        try:
                            # First try strict loading
                            self.model.load_state_dict(state_dict, strict=True)
                            print(f"✅ Loaded state dict (strict) from {model_path}")
                            model_loaded = True
                            break
                        except Exception as strict_error:
                            print(f"⚠️  Strict loading failed: {strict_error}")
                            try:
                                # Try non-strict loading
                                missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=False)
                                print(f"📝 Missing keys: {missing_keys}")
                                print(f"📝 Unexpected keys: {unexpected_keys}")
                                print(f"✅ Loaded state dict (non-strict) from {model_path}")
                                model_loaded = True
                                break
                            except Exception as non_strict_error:
                                print(f"⚠️  Non-strict loading failed: {non_strict_error}")
                                # Try partial loading
                                try:
                                    self.partial_load_state_dict(state_dict)
                                    print(f"✅ Partial load successful from {model_path}")
                                    model_loaded = True
                                    break
                                except Exception as partial_error:
                                    print(f"❌ Partial loading failed: {partial_error}")
                    
                    else:
                        # Try to treat as state dict directly
                        try:
                            self.model.load_state_dict(checkpoint, strict=False)
                            print(f"✅ Loaded as direct state dict from {model_path}")
                            model_loaded = True
                            break
                        except Exception as direct_error:
                            print(f"❌ Direct state dict loading failed: {direct_error}")
                    
                except Exception as e:
                    print(f"❌ Failed to load {model_path}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
            
            if not model_loaded:
                print("⚠️  No compatible model file found.")
                print("   Creating untrained model - results will not be optimal.")
                print("   Please provide a trained model file to get better results.")
                
                # Initialize with random weights (for demonstration)
                self.initialize_random_weights()
                
            self.model.to(self.device)
            self.model.eval()
            print(f"🎯 Model moved to device: {self.device}")
            
        except Exception as e:
            error_msg = f"Failed to initialize model: {str(e)}"
            print(f"❌ {error_msg}")
            messagebox.showerror("Model Error", 
                f"{error_msg}\n\nThe application will continue with basic image processing.")
            self.model = None
    
    def partial_load_state_dict(self, state_dict):
        """Load compatible layers from state dict"""
        model_dict = self.model.state_dict()
        compatible_dict = {}
        
        for k, v in state_dict.items():
            if k in model_dict:
                if model_dict[k].shape == v.shape:
                    compatible_dict[k] = v
                else:
                    print(f"⚠️  Shape mismatch for {k}: model {model_dict[k].shape} vs checkpoint {v.shape}")
            else:
                print(f"⚠️  Key not found in model: {k}")
        
        model_dict.update(compatible_dict)
        self.model.load_state_dict(model_dict)
        print(f"✅ Loaded {len(compatible_dict)}/{len(state_dict)} compatible parameters")
    
    def initialize_random_weights(self):
        """Initialize model with reasonable random weights"""
        for module in self.model.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        
    def setup_styles(self):
        """Setup custom styles for the water theme"""
        self.style = ttk.Style()
        
        # Configure custom styles
        self.style.configure('Ocean.TFrame', background='#001122')
        self.style.configure('Wave.TButton', 
                           background='#0066CC',
                           foreground='white',
                           font=('Arial', 10, 'bold'))
        self.style.configure('Deep.TLabel',
                           background='#001122',
                           foreground='#66CCFF',
                           font=('Arial', 12))
        self.style.configure('Title.TLabel',
                           background='#001122',
                           foreground='#00FFFF',
                           font=('Arial', 18, 'bold'))
        
    def create_widgets(self):
        """Create and arrange GUI widgets"""
        # Main container
        main_frame = ttk.Frame(self.root, style='Ocean.TFrame')
        main_frame.pack(fill='both', expand=True, padx=20, pady=20)
        
        # Title
        title_label = ttk.Label(main_frame, 
                               text="🌊 Underwater Image Restoration 🐠",
                               style='Title.TLabel')
        title_label.pack(pady=10)
        
        # Control panel
        control_frame = ttk.Frame(main_frame, style='Ocean.TFrame')
        control_frame.pack(fill='x', pady=10)
        
        # Buttons
        btn_frame = ttk.Frame(control_frame, style='Ocean.TFrame')
        btn_frame.pack(side='left')
        
        self.select_btn = tk.Button(btn_frame, 
                                   text="📁 Select Image",
                                   command=self.select_image,
                                   bg='#0066CC', fg='white',
                                   font=('Arial', 10, 'bold'),
                                   padx=20, pady=10,
                                   relief='raised',
                                   cursor='hand2')
        self.select_btn.pack(side='left', padx=5)
        
        self.restore_btn = tk.Button(btn_frame,
                                    text="🔧 Restore Image",
                                    command=self.restore_image,
                                    bg='#0088AA', fg='white',
                                    font=('Arial', 10, 'bold'),
                                    padx=20, pady=10,
                                    relief='raised',
                                    cursor='hand2',
                                    state='disabled')
        self.restore_btn.pack(side='left', padx=5)
        
        self.save_btn = tk.Button(btn_frame,
                                 text="💾 Save Result",
                                 command=self.save_image,
                                 bg='#00AA88', fg='white',
                                 font=('Arial', 10, 'bold'),
                                 padx=20, pady=10,
                                 relief='raised',
                                 cursor='hand2',
                                 state='disabled')
        self.save_btn.pack(side='left', padx=5)
        
        # Model reload button
        self.reload_btn = tk.Button(btn_frame,
                                   text="🔄 Reload Model",
                                   command=self.reload_model,
                                   bg='#AA6600', fg='white',
                                   font=('Arial', 10, 'bold'),
                                   padx=20, pady=10,
                                   relief='raised',
                                   cursor='hand2')
        self.reload_btn.pack(side='left', padx=5)
        
        # Diagnostic button
        self.diagnose_btn = tk.Button(btn_frame,
                                     text="🔍 Diagnose Model",
                                     command=self.diagnose_model,
                                     bg='#AA0066', fg='white',
                                     font=('Arial', 10, 'bold'),
                                     padx=20, pady=10,
                                     relief='raised',
                                     cursor='hand2')
        self.diagnose_btn.pack(side='left', padx=5)
        
        # Progress bar
        self.progress_frame = ttk.Frame(control_frame, style='Ocean.TFrame')
        self.progress_frame.pack(side='right')
        
        self.progress_label = ttk.Label(self.progress_frame,
                                       text="Ready",
                                       style='Deep.TLabel')
        self.progress_label.pack()
        
        self.progress_bar = ttk.Progressbar(self.progress_frame,
                                           mode='indeterminate',
                                           style='TProgressbar')
        
        # Image display area
        image_frame = ttk.Frame(main_frame, style='Ocean.TFrame')
        image_frame.pack(fill='both', expand=True, pady=20)
        
        # Original image panel
        original_panel = ttk.Frame(image_frame, style='Ocean.TFrame')
        original_panel.pack(side='left', fill='both', expand=True, padx=10)
        
        orig_label = ttk.Label(original_panel,
                              text="📷 Original Image",
                              style='Deep.TLabel')
        orig_label.pack(pady=5)
        
        self.original_canvas = tk.Canvas(original_panel,
                                        bg='#002244',
                                        highlightthickness=2,
                                        highlightbackground='#0066CC')
        self.original_canvas.pack(fill='both', expand=True)
        
        # Restored image panel
        restored_panel = ttk.Frame(image_frame, style='Ocean.TFrame')
        restored_panel.pack(side='right', fill='both', expand=True, padx=10)
        
        rest_label = ttk.Label(restored_panel,
                              text="✨ Restored Image",
                              style='Deep.TLabel')
        rest_label.pack(pady=5)
        
        self.restored_canvas = tk.Canvas(restored_panel,
                                        bg='#002244',
                                        highlightthickness=2,
                                        highlightbackground='#00FFFF')
        self.restored_canvas.pack(fill='both', expand=True)
        
        # Status bar
        status_frame = ttk.Frame(main_frame, style='Ocean.TFrame')
        status_frame.pack(fill='x', pady=5)
        
        model_status = '✅ Loaded' if self.model else '❌ Not Loaded'
        found_models = f"Found {len(self.model_paths)} model files" if self.model_paths else "No model files found"
        
        self.status_label = ttk.Label(status_frame,
                                     text=f"Device: {self.device} | Model: {model_status} | {found_models} | Status: Ready",
                                     style='Deep.TLabel')
        self.status_label.pack(side='left')
    
    def diagnose_model(self):
        """Detailed diagnosis of model files"""
        if not self.model_paths:
            messagebox.showinfo("No Model Files", "No model files found to diagnose.")
            return
        
        diagnosis_text = "🔍 MODEL FILE DIAGNOSIS\n" + "="*50 + "\n\n"
        
        for i, model_path in enumerate(self.model_paths, 1):
            diagnosis_text += f"📁 File {i}: {os.path.basename(model_path)}\n"
            diagnosis_text += f"   Path: {model_path}\n"
            
            try:
                # Get file size
                file_size = os.path.getsize(model_path) / (1024*1024)  # MB
                diagnosis_text += f"   Size: {file_size:.1f} MB\n"
                
                # Try to load and inspect
                try:
                    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
                    diagnosis_text += f"   Type: {type(checkpoint).__name__}\n"
                    
                    if isinstance(checkpoint, dict):
                        diagnosis_text += f"   Keys: {list(checkpoint.keys())}\n"
                        
                        # Check for common state dict patterns
                        for key in checkpoint.keys():
                            if 'state_dict' in key.lower() or key == 'model':
                                try:
                                    state_dict = checkpoint[key]
                                    if isinstance(state_dict, dict):
                                        diagnosis_text += f"   State dict in '{key}': {len(state_dict)} parameters\n"
                                        # Show first few parameter names
                                        param_names = list(state_dict.keys())[:5]
                                        diagnosis_text += f"   First params: {param_names}\n"
                                except:
                                    pass
                        
                        # If direct dict, check if it looks like a state dict
                        if all(isinstance(k, str) and '.' in k for k in list(checkpoint.keys())[:10]):
                            diagnosis_text += f"   Appears to be direct state dict: {len(checkpoint)} parameters\n"
                            param_names = list(checkpoint.keys())[:5]
                            diagnosis_text += f"   First params: {param_names}\n"
                    
                    elif hasattr(checkpoint, 'state_dict'):
                        diagnosis_text += f"   Model object with state_dict method\n"
                        try:
                            state_dict = checkpoint.state_dict()
                            diagnosis_text += f"   Parameters: {len(state_dict)}\n"
                        except:
                            pass
                    
                    diagnosis_text += f"   ✅ File readable\n"
                    
                except Exception as load_error:
                    diagnosis_text += f"   ❌ Load error: {str(load_error)}\n"
                    
                    # Try weights_only=True
                    try:
                        checkpoint = torch.load(model_path, map_location='cpu', weights_only=True)
                        diagnosis_text += f"   ✅ Readable with weights_only=True\n"
                        diagnosis_text += f"   Type: {type(checkpoint).__name__}\n"
                        if isinstance(checkpoint, dict):
                            diagnosis_text += f"   Keys: {list(checkpoint.keys())}\n"
                    except Exception as weights_error:
                        diagnosis_text += f"   ❌ weights_only error: {str(weights_error)}\n"
                
            except Exception as e:
                diagnosis_text += f"   ❌ Error: {str(e)}\n"
            
            diagnosis_text += "\n"
        
        # Show our model's expected structure
        diagnosis_text += "🏗️  EXPECTED MODEL STRUCTURE\n" + "="*30 + "\n"
        if self.model:
            model_params = list(self.model.state_dict().keys())
            diagnosis_text += f"Expected parameters: {len(model_params)}\n"
            diagnosis_text += f"First few: {model_params[:10]}\n"
            diagnosis_text += f"Last few: {model_params[-5:]}\n"
        
        # Show in a scrollable text window
        self.show_diagnosis_window(diagnosis_text)
    
    def show_diagnosis_window(self, text):
        """Show diagnosis in a new window"""
        diag_window = tk.Toplevel(self.root)
        diag_window.title("🔍 Model Diagnosis")
        diag_window.geometry("800x600")
        diag_window.configure(bg='#001122')
        
        # Create text widget with scrollbar
        frame = tk.Frame(diag_window, bg='#001122')
        frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        text_widget = tk.Text(frame, 
                             bg='#002244', 
                             fg='#66CCFF',
                             font=('Consolas', 10),
                             wrap=tk.WORD)
        
        scrollbar = tk.Scrollbar(frame, command=text_widget.yview)
        text_widget.config(yscrollcommand=scrollbar.set)
        
        scrollbar.pack(side='right', fill='y')
        text_widget.pack(side='left', fill='both', expand=True)
        
        text_widget.insert('1.0', text)
        text_widget.config(state='disabled')  # Make read-only
        
        # Add copy button
        copy_btn = tk.Button(diag_window,
                            text="📋 Copy to Clipboard",
                            command=lambda: self.copy_to_clipboard(text),
                            bg='#0066CC', fg='white',
                            font=('Arial', 10, 'bold'))
        copy_btn.pack(pady=5)
    
    def copy_to_clipboard(self, text):
        """Copy text to clipboard"""
        self.root.clipboard_clear()
        self.root.clipboard_append(text)
        messagebox.showinfo("Copied", "Diagnosis copied to clipboard!")
    
    def reload_model(self):
        """Reload the model (useful after adding new model files)"""
        self.progress_label.config(text="🔄 Reloading model...")
        self.find_model_files()
        self.load_model()
        
        model_status = '✅ Loaded' if self.model else '❌ Not Loaded'
        found_models = f"Found {len(self.model_paths)} model files" if self.model_paths else "No model files found"
        
        self.status_label.config(text=f"Device: {self.device} | Model: {model_status} | {found_models} | Status: Ready")
        self.progress_label.config(text="Model reload complete!")
        
        if not self.model_paths:
            messagebox.showinfo("Model Files", 
                "No model files found. Please place your model file in:\n"
                "• Current directory (underwater_model_full.pth)\n"
                "• models/ subdirectory\n"
                "• Desktop or Downloads folder\n\n"
                "Then click 'Reload Model' button.")
    
    def select_image(self):
        """Select an image file for restoration"""
        file_path = filedialog.askopenfilename(
            title="Select Underwater Image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            try:
                # Load and display original image
                self.original_image = Image.open(file_path)
                self.display_image(self.original_image, self.original_canvas)
                
                # Enable restore button
                self.restore_btn.config(state='normal')
                self.progress_label.config(text="Image loaded - Ready to restore")
                
                # Clear previous restoration
                self.restored_canvas.delete("all")
                self.restored_image = None
                self.save_btn.config(state='disabled')
                
            except Exception as e:
                messagebox.showerror("Image Error", f"Failed to load image: {str(e)}")
    
    def display_image(self, pil_image, canvas):
        """Display PIL image on canvas with proper scaling"""
        canvas.delete("all")
        
        # Get canvas dimensions
        canvas.update()
        canvas_width = canvas.winfo_width()
        canvas_height = canvas.winfo_height()
        
        if canvas_width <= 1 or canvas_height <= 1:
            # Canvas not ready, try again later
            self.root.after(100, lambda: self.display_image(pil_image, canvas))
            return
        
        # Calculate scaling to fit canvas
        img_width, img_height = pil_image.size
        scale_x = canvas_width / img_width
        scale_y = canvas_height / img_height
        scale = min(scale_x, scale_y, 1.0)  # Don't upscale
        
        new_width = int(img_width * scale)
        new_height = int(img_height * scale)
        
        # Resize image
        resized_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Convert to PhotoImage
        photo = ImageTk.PhotoImage(resized_image)
        
        # Center image on canvas
        x = (canvas_width - new_width) // 2
        y = (canvas_height - new_height) // 2
        
        canvas.create_image(x, y, anchor='nw', image=photo)
        canvas.image = photo  # Keep a reference
    
    def preprocess_image(self, pil_image):
        """Preprocess image for model input"""
        # Resize to model input size
        img = pil_image.resize((224, 224), Image.Resampling.LANCZOS)
        
        # Convert to tensor
        img_array = np.array(img).astype(np.float32) / 255.0
        
        # Handle grayscale images
        if len(img_array.shape) == 2:
            img_array = np.stack([img_array] * 3, axis=-1)
        elif img_array.shape[2] == 4:  # RGBA
            img_array = img_array[:, :, :3]  # Remove alpha channel
            
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)
        
        return img_tensor.to(self.device)
    
    def postprocess_output(self, tensor_output, target_size):
        """Convert model output back to PIL image"""
        # Convert to numpy
        output_np = tensor_output.squeeze().detach().cpu().numpy()
        output_np = np.transpose(output_np, (1, 2, 0))
        
        # Clip to valid range
        output_np = np.clip(output_np, 0, 1)
        
        # Convert to uint8
        output_np = (output_np * 255).astype(np.uint8)
        
        # Create PIL image and resize back to original size
        pil_image = Image.fromarray(output_np)
        pil_image = pil_image.resize(target_size, Image.Resampling.LANCZOS)
        
        return pil_image
    
    def restore_image_worker(self):
        """Worker function for image restoration"""
        try:
            # Update UI
            self.root.after(0, lambda: self.progress_label.config(text="🔧 Processing image..."))
            self.root.after(0, lambda: self.progress_bar.pack(pady=5))
            self.root.after(0, lambda: self.progress_bar.start())
            
            # Preprocess image
            original_size = self.original_image.size
            input_tensor = self.preprocess_image(self.original_image)
            
            self.root.after(0, lambda: self.progress_label.config(text="🧠 Running AI model..."))
            
            if self.model is not None:
                # Run inference with model
                with torch.no_grad():
                    output_tensor = self.model(input_tensor)
            else:
                # Fallback: simple image enhancement without model
                print("⚠️  Model not available, using basic image enhancement")
                self.root.after(0, lambda: self.progress_label.config(text="📸 Applying basic enhancement..."))
                output_tensor = self.basic_image_enhancement(input_tensor)
            
            self.root.after(0, lambda: self.progress_label.config(text="✨ Finalizing image..."))
            
            # Postprocess
            self.restored_image = self.postprocess_output(output_tensor, original_size)
            
            # Update UI on main thread
            self.root.after(0, self.restoration_complete)
            
        except Exception as e:
            error_msg = f"Failed to restore image: {str(e)}"
            print(f"❌ Restoration error: {error_msg}")
            self.root.after(0, lambda: messagebox.showerror("Restoration Error", error_msg))
            self.root.after(0, self.restoration_complete)
    
    def basic_image_enhancement(self, input_tensor):
        """Basic image enhancement when model is not available"""
        # Simple color correction and contrast enhancement
        img = input_tensor.squeeze().cpu().numpy()
        
        # Increase contrast and adjust gamma
        img = np.power(img, 0.8)  # Gamma correction
        img = np.clip(img * 1.2, 0, 1)  # Contrast boost
        
        # Convert back to tensor
        output_tensor = torch.from_numpy(img).unsqueeze(0).to(self.device)
        return output_tensor
    
    def restore_image(self):
        """Start image restoration process"""
        if self.original_image is None:
            messagebox.showwarning("No Image", "Please select an image first!")
            return
        
        if self.processing:
            return
        
        self.processing = True
        self.restore_btn.config(state='disabled')
        
        # Start restoration in separate thread
        thread = Thread(target=self.restore_image_worker)
        thread.daemon = True
        thread.start()
    
    def restoration_complete(self):
        """Called when restoration is complete"""
        self.processing = False
        self.progress_bar.stop()
        self.progress_bar.pack_forget()
        
        if self.restored_image:
            self.display_image(self.restored_image, self.restored_canvas)
            self.save_btn.config(state='normal')
            if self.model:
                self.progress_label.config(text="✅ Restoration complete!")
            else:
                self.progress_label.config(text="✅ Basic enhancement complete!")
        else:
            self.progress_label.config(text="❌ Restoration failed!")
        
        self.restore_btn.config(state='normal')
    
    def save_image(self):
        """Save the restored image"""
        if self.restored_image is None:
            messagebox.showwarning("No Result", "No restored image to save!")
            return
        
        file_path = filedialog.asksaveasfilename(
            title="Save Restored Image",
            defaultextension=".png",
            filetypes=[
                ("PNG files", "*.png"),
                ("JPEG files", "*.jpg"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            try:
                self.restored_image.save(file_path)
                messagebox.showinfo("Success", f"Image saved successfully to:\n{file_path}")
                self.progress_label.config(text="💾 Image saved successfully!")
            except Exception as e:
                messagebox.showerror("Save Error", f"Failed to save image: {str(e)}")

def main():
    root = tk.Tk()
    app = UnderwaterRestorationGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()