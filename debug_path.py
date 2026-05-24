import os

# Check the first folder (e.g., Animal)
folder_path = os.path.join("data", "Animal")

print(f"🕵️‍♂️ Investigating: {folder_path}")

if not os.path.exists(folder_path):
    print("❌ Error: Python cannot find this folder at all!")
else:
    files = os.listdir(folder_path)
    print(f"📂 Found {len(files)} files/folders inside.")
    
    # Print the first 5 names to see what they look like
    print("First 5 items:", files[:5])