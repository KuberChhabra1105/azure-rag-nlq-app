import os
from dotenv import load_dotenv
from azure.storage.blob import BlobServiceClient

load_dotenv()

connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
container_name = os.getenv("AZURE_STORAGE_CONTAINER")

blob_service = BlobServiceClient.from_connection_string(connection_string)
container = blob_service.get_container_client(container_name)

try:
    container.create_container()
except:
    pass

pdf_folder = "data/pdfs"

files = []

for root, dirs, names in os.walk(pdf_folder):
    for name in names:
        if name.endswith(".pdf"):
            files.append(os.path.join(root, name))

print(f"Found {len(files)} PDF files")

for file in files:

    blob_name = os.path.relpath(file, pdf_folder).replace("\\", "/")

    print("Uploading :", blob_name)

    blob = container.get_blob_client(blob_name)

    with open(file, "rb") as f:
        blob.upload_blob(f, overwrite=True)

print("\nUpload Completed")