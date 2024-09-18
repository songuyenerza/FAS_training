import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('/data/datasets/thainq/sonnt373/dev/FAS/data/Data_FAS_v0/labels.csv')

# Split train và val
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, shuffle=True)  # 80% train, 20% val

def save_to_txt(df, filename):
    with open(filename, 'w') as f:
        for index, row in df.iterrows():
            f.write(f"celeb_rose/{row['filename']} {row['label']}\n")

save_to_txt(train_df, '/data/datasets/thainq/sonnt373/dev/FAS/data/Data_FAS_v0/train_1.txt')
save_to_txt(val_df, '/data/datasets/thainq/sonnt373/dev/FAS/data/Data_FAS_v0/val_1.txt')


# import cv2
# import os
# root = "/data/datasets/thainq/sonnt373/dev/FAS/data/Data_FAS_v0"
# data_list = "/data/datasets/thainq/sonnt373/dev/FAS/data/Data_FAS_v0/val.txt"
# with open(data_list, 'r') as f:
#     lines = f.readlines()
#     for line in lines:

#         items = line.strip().split()
#         print(items)
#         img_path = items[0]
#         label = items[1]
#         img_path = os.path.join(root, f"celeb_rose/{img_path}")
#         img = cv2.imread(img_path)
#         print("img_path: ", img_path, img.shape)