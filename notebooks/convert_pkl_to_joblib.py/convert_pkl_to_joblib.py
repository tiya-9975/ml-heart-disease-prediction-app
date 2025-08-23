import os
import glob
import joblib

# Paths
notebooks_dir = os.path.join(os.getcwd(), "notebooks")  # where your .pkl models are
models_dir = os.path.join(os.getcwd(), "models")        # where you want .joblib models

# Create models folder if it doesn't exist
os.makedirs(models_dir, exist_ok=True)

# Find all .pkl files in notebooks/
pkl_files = glob.glob(os.path.join(notebooks_dir, "*.pkl"))

if not pkl_files:
    print("No .pkl files found in notebooks/")
else:
    for pkl_path in pkl_files:
        model_name = os.path.splitext(os.path.basename(pkl_path))[0]
        try:
            # Load the old .pkl model
            model = joblib.load(pkl_path)
            # Save as .joblib in models folder
            joblib.dump(model, os.path.join(models_dir, f"{model_name}.joblib"))
            print(f"✅ Converted {model_name}.pkl → {model_name}.joblib")
        except Exception as e:
            print(f"❌ Failed to convert {model_name}.pkl: {e}")

print("All done! Check the models/ folder.")
