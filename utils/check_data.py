import os

# CONFIGURATION
DATA_PATH = "keypoints"  # Path to your data folder

if not os.path.exists(DATA_PATH):
    print(f"❌ Error: Folder '{DATA_PATH}' not found.")
    exit()

actions = os.listdir(DATA_PATH)
# Sort alphabetically to keep the list clean
actions.sort()

print(f"📂 Found {len(actions)} Action Classes.\n")
print(f"{'ACTION':<20} | {'SAMPLES':<10} | {'STATUS'}")
print("-" * 45)

low_data_count = 0
valid_classes = 0

for action in actions:
    action_path = os.path.join(DATA_PATH, action)
    if not os.path.isdir(action_path):
        continue
        
    # --- THE FIX: Count .npy FILES, not folders ---
    files = [f for f in os.listdir(action_path) if f.endswith('.npy')]
    count = len(files)
    
    # We want at least 40 samples (Original + Mirrored)
    status = "✅ OK" if count >= 40 else "⚠️ LOW"
    
    if count < 40: 
        low_data_count += 1
    else:
        valid_classes += 1
        
    print(f"{action:<20} | {count:<10} | {status}")

print("-" * 45)
print(f"📊 Summary: {valid_classes} Ready / {low_data_count} Low Data")

if low_data_count == 0:
    print("🎉 ALL SYSTEMS GO! Every word has enough data.")
else:
    print(f"⚠️ WARNING: {low_data_count} words have less than 40 videos.")