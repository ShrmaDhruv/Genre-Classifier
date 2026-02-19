import shutil
import os

print("Copying data from Google Drive to local Colab storage...")
print("This is a ONE-TIME operation and will take ~2-3 minutes")
print("-" * 60)

source = "/content/drive/MyDrive/gtzan_spectrograms/output_img"
destination = "/content/gtzan_spectrograms"

if os.path.exists(destination):
    print(f"Data already exists at {destination}")
else:
    shutil.copytree(source, destination)
    print(f"Data copied successfully!")

total_files = sum([len(files) for r, d, files in os.walk(destination)])
print(f"Total files copied: {total_files}")
print("-" * 60)