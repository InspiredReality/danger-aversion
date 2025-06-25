#!/usr/bin/env python3
"""
Model download script for Railway deployment
This runs during Docker build to pre-download the sentence transformer model
"""

import os
import sys

def main():
    try:
        print("=== Starting model download ===")
        
        # Create model directory
        model_dir = '/tmp/models'
        os.makedirs(model_dir, exist_ok=True)
        print(f"Created model directory: {model_dir}")
        
        # Import sentence transformers
        from sentence_transformers import SentenceTransformer
        print("✅ SentenceTransformer imported successfully")
        
        # Download the model
        print("📥 Downloading all-MiniLM-L6-v2 model...")
        model = SentenceTransformer('all-MiniLM-L6-v2', cache_folder=model_dir)
        print("✅ Model downloaded successfully!")
        
        # Test the model
        print("🧪 Testing model...")
        test_embedding = model.encode(['test sentence'])
        print(f"✅ Model test passed! Embedding shape: {test_embedding.shape}")
        
        # List downloaded files for verification
        print("📁 Downloaded model files:")
        file_count = 0
        for root, dirs, files in os.walk(model_dir):
            level = root.replace(model_dir, '').count(os.sep)
            indent = '  ' * level
            print(f'{indent}{os.path.basename(root)}/')
            subindent = '  ' * (level + 1)
            for file in files[:5]:  # Show first 5 files per directory
                file_path = os.path.join(root, file)
                try:
                    size = os.path.getsize(file_path)
                    print(f'{subindent}{file} ({size:,} bytes)')
                    file_count += 1
                except:
                    print(f'{subindent}{file}')
            if len(files) > 5:
                print(f'{subindent}... and {len(files)-5} more files')
        
        print(f"📊 Total files processed: {file_count}")
        print("=== Model download complete ===")
        
        return 0
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Required packages might not be installed")
        return 1
        
    except Exception as e:
        print(f"❌ ERROR: Model download failed: {e}")
        print(f"Error type: {type(e).__name__}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)