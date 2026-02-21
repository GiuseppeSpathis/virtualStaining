import os
import shutil
import random

def create_dataset_split(base_path, train_ratio=0.7):
    src_he = os.path.join(base_path, 'he')
    src_ihc = os.path.join(base_path, 'ihc')
    
    splits = ['training', 'testing']
    categories = ['he', 'ihc']
    
    for s in splits:
        for c in categories:
            target_path = os.path.join(base_path, s, c)
            os.makedirs(target_path, exist_ok=True)
            
    def process_split(source_dir, category_name):
        files = [f for f in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, f))]
        random.shuffle(files) 
        
        split_idx = int(len(files) * train_ratio)
        train_files = files[:split_idx]
        test_files = files[split_idx:]
        
        
        for f in train_files:
            shutil.copy2(os.path.join(source_dir, f), os.path.join(base_path, 'training', category_name, f))
            
        for f in test_files:
            shutil.copy2(os.path.join(source_dir, f), os.path.join(base_path, 'testing', category_name, f))
            
        print(f"  -> {len(train_files)} in training/{category_name}")
        print(f"  -> {len(test_files)} in testing/{category_name}")

    if os.path.exists(src_he):
        process_split(src_he, 'he')

    if os.path.exists(src_ihc):
        process_split(src_ihc, 'ihc')
    
if __name__ == "__main__":
    data_path = os.path.expanduser("~/giuSpathis/data/")
    create_dataset_split(data_path)
