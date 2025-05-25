import os
import pandas as pd

def cleancsv(csvpath, imagedir):
    df = pd.read_csv(csvpath)
    valid_rows = []
    
    for _, row in df.iterrows():
        img_path = os.path.join(imagedir, f"{row['imdbId']}.jpg")
        if os.path.exists(img_path):
            valid_rows.append(row)

    cleaned_df = pd.DataFrame(valid_rows)
    cleaned_path = f"cleaned{os.path.basename(csvpath)}"
    cleaned_df.to_csv(cleaned_path, index=False)
    print(f"Cleaned {len(valid_rows)}/{len(df)} entries -> {cleaned_path}")
    return cleaned_df

if __name__ == "__main__":
    IMAGE_DIR = r"C:\Users\Zain Ul Ibad\Desktop\aistuff\images"  
    CSV_PATHS = [                    #put files here jis ko u wanna clean
        'trainlabels.csv',     
        'validatelabels.csv',
        'testinglabels.csv'
    ]
    
    # ===== RUN CLEANING ===== #
    for path in CSV_PATHS:
        cleancsv(path, IMAGE_DIR)