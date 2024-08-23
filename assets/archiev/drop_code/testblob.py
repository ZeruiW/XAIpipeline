from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
import os
from dotenv import load_dotenv


# 加载 .env 文件中的环境变量
load_dotenv()

# 从环境变量中获取连接字符串
connect_str = os.getenv('AZURE_STORAGE_CONNECTION_STRING')

# 确保连接字符串被正确读取
if connect_str:
    print("连接字符串读取成功。")
else:
    print("未能读取连接字符串。请检查 .env 文件。")

# 创建 BlobServiceClient 对象
blob_service_client = BlobServiceClient.from_connection_string(connect_str)

# 选择或创建容器
container_name = 'xaidata12'
container_client = blob_service_client.get_container_client(container_name)
try:
    container_client.create_container()
except Exception as e:
    print(f"注意：{e}")

# 上传文件
def upload_file_to_blob(file_name, blob_name):
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)
    with open(file_name, "rb") as data:
        blob_client.upload_blob(data)
    print(f"'{file_name}' 上传至 Blob 存储中的 '{blob_name}' 完成。")

# 下载文件
def download_file_from_blob(blob_name, download_file_name):
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)
    with open(download_file_name, "wb") as download_file:
        download_file.write(blob_client.download_blob().readall())
    print(f"'{blob_name}' 从 Blob 存储下载至 '{download_file_name}' 完成。")

# 示例用法
upload_file_to_blob("/home/z/Music/devnew_xaiservice/XAIport/testblob.py", "0.JPEG")
# download_file_from_blob("0.JPEG", "2.JPEG")
