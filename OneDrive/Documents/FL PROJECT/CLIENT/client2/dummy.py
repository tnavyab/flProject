import os
import pandas as pd

csv = r"C:\Users\tnavy\OneDrive\Documents\FL PROJECT\DATA\messidor_data.csv"
img_dir = r"C:\Users\tnavy\Documents\FL PROJECT\DATA\IMAGES"

df = pd.read_csv(csv)

print("CSV columns:", df.columns.tolist())
print("First image value:", df.iloc[0])

print("\nImages in folder (first 10):")
print(os.listdir(img_dir)[:10])
