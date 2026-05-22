import os
from PIL import Image
from tqdm import tqdm

def resize_png_images_safe(input_folder, target_size=(1280, 1024)):

    if not os.path.exists(input_folder):
        print(f"❌ Error: Input folder '{input_folder}' does not exist. Please check the path.")
        return

    output_folder = os.path.join(os.path.dirname(input_folder), f"{os.path.basename(input_folder)}_resized_1280x1024")
    os.makedirs(output_folder, exist_ok=True)

    png_files = [f for f in os.listdir(input_folder) if f.lower().endswith('.png')]
    
    if not png_files:
        print(f"ℹ️ No .png images found in '{input_folder}'.")
        return

    print(f"📁 Source folder: {input_folder}")
    print(f"📁 Saving resized images to: {output_folder}")
    print(f"🔍 Found {len(png_files)} PNG images. Starting process...\n")

    # 4. Batch Processing
    for filename in tqdm(png_files, desc="Processing progress"):
        src_path = os.path.join(input_folder, filename)
        dst_path = os.path.join(output_folder, filename)
        
        try:
            with Image.open(src_path) as img:
                orig_w, orig_h = img.size
                
                # Resize image (using high-quality LANCZOS filter)
                resized_img = img.resize(target_size, Image.Resampling.LANCZOS)
                
                # Save to the new target path without overwriting the original file
                resized_img.save(dst_path)
                
        except Exception as e:
            print(f"❌ Error processing image {filename}: {e}")

    print(f"\n✨ Processing complete! All new images are safely saved at: \n👉 {output_folder}")

if __name__ == "__main__":

    TARGET_FOLDER = "/media/mitiadmin/Micron_7450_1/tianming/dataset/CMC_grasp10_medical_format/JPEGImages/cmc_sequence" 
    
    resize_png_images_safe(TARGET_FOLDER)