import gdown
import torch
from flask import Flask, request, send_file, jsonify
from PIL import Image, ImageEnhance
import numpy as np
from io import BytesIO

app = Flask(__name__)

# Define the Google Drive link for your model file
model_drive_id = '19wCVevCnjF2TNQomEnmP9ffgWGOkkapy'
model_drive_link = f'https://drive.google.com/uc?id={model_drive_id}'

# Download the model file when the app starts
gdown.download(model_drive_link, 'models/RealESRGAN_x4plus.pth', quiet=False)

# Load the RealESRGAN model and other necessary configurations here
model_path = 'models/RealESRGAN_x4plus.pth'  # Path to your downloaded RealESRGAN model

# Define the route for processing images
@app.route('/process_image', methods=['POST'])
def process_image():
    # Receive the uploaded image
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    try:
        # Load the image
        image = Image.open(file)
        
        # Resize the image to 256 pixels wide while maintaining the aspect ratio
        target_width = 256
        aspect_ratio = image.height / image.width
        target_height = int(target_width * aspect_ratio)
        image_resized = image.resize((target_width, target_height), Image.LANCZOS)
        
        # Define the RRDBNet architecture
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        
        # Initialize RealESRGANer
        upscaler = RealESRGANer(
            scale=2,
            model_path=model_path,
            model=model,
            tile=1024,
            tile_pad=5,
            pre_pad=0,
            half=False  # Disable FP16 since we're using CPU
        )
        
        # Load the model weights directly
        model_weights = torch.load(model_path, map_location='cpu')
        upscaler.model.load_state_dict(model_weights['params_ema'])
        
        # Ensure the model is loaded on the CPU
        device = torch.device('cpu')
        upscaler.device = device
        upscaler.model.to(device)
        
        # Convert the image to RGB format
        image_resized = image_resized.convert('RGB')
        
        # Ensure the NumPy array is writable before converting to tensor
        img_np = np.array(image_resized)
        if not img_np.flags.writeable:
            img_np = np.copy(img_np)
        
        # Normalize the image
        img_np = img_np / 255.0
        
        # Pre-process the image (to tensor)
        img_tensor = torch.from_numpy(np.transpose(img_np, (2, 0, 1))).float().unsqueeze(0)
        img_tensor = img_tensor.to(device)
        
        # Perform the upscaling
        with torch.no_grad():
            output_tensor = upscaler.model(img_tensor)
        
        # Denormalize the output image
        output_tensor = output_tensor.squeeze().cpu().numpy()
        output_tensor = np.transpose(output_tensor, (1, 2, 0))
        output_tensor = np.clip(output_tensor, 0, 1) * 255.0
        output_img = output_tensor.astype(np.uint8)
        
        # Convert the NumPy array back to a PIL Image
        upscaled_image = Image.fromarray(output_img)
        
        # Resize the upscaled image to 4096 pixels wide while maintaining the aspect ratio
        target_width = 4096
        aspect_ratio = upscaled_image.height / upscaled_image.width
        target_height = int(target_width * aspect_ratio)
        upscaled_image_resized = upscaled_image.resize((target_width, target_height), Image.LANCZOS)
        
        # Post-processing (Contrast Enhancement)
        enhancer = ImageEnhance.Contrast(upscaled_image_resized)
        upscaled_image_final = enhancer.enhance(1.2)
        
        # Save the processed image
        processed_img_io = BytesIO()
        upscaled_image_final.save(processed_img_io, format='JPEG')
        processed_img_io.seek(0)
        
        # Return the processed image
        return send_file(processed_img_io, mimetype='image/jpeg')
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
