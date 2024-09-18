#!/bin/bash
URL_PATH=$(python3.8 -c "from config import ApplicationConfig as cfg; print(cfg.url_data)")
echo "URL_PATH: $URL_PATH"
# mkdir DATASET 
DATASET_DIR="./DATASET"
if [ ! -d "$DATASET_DIR" ]; then
  echo "Creating directory: $DATASET_DIR"
  mkdir -p "$DATASET_DIR"
else
  echo "Directory $DATASET_DIR already exists."
fi

DESTINATION="$DATASET_DIR/dataset.zip"

echo "Downloading file ..."
wget --no-check-certificate "$URL_PATH" -O "$DESTINATION"

if [ -f "$DESTINATION" ]; then
  echo "File downloaded successfully: $DESTINATION"
else
  echo "Failed to download file."
  exit 1
fi

# unzip
echo "Extracting file..."
unzip "$DESTINATION" -d "$DATASET_DIR"

if [ $? -eq 0 ]; then
  echo "Extraction completed."
  rm "$DESTINATION"
  echo "Deleted the zip file: $DESTINATION"
else
  echo "Failed to extract the file."
  exit 1
fi

git config --global --add safe.directory /app-src/face-anti-spoofing-training
