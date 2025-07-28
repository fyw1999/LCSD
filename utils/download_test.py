import os
import requests
from threading import Thread, Lock
from tqdm import tqdm

lock = Lock()  # 用于同步进度条更新

def download_chunk(url, start, end, file_name, chunk_number, pbar):
    headers = {'Range': f'bytes={start}-{end}'}
    response = requests.get(url, headers=headers, stream=True)
    chunk_file_name = f"{file_name}.part{chunk_number}"
    
    with open(chunk_file_name, 'wb') as file:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                file.write(chunk)
                with lock:
                    pbar.update(len(chunk))
    print(f"Chunk {chunk_number} downloaded.")

def combine_chunks(file_name, total_chunks):
    with open(file_name, 'wb') as output_file:
        for i in range(total_chunks):
            chunk_file_name = f"{file_name}.part{i}"
            with open(chunk_file_name, 'rb') as chunk_file:
                output_file.write(chunk_file.read())
            os.remove(chunk_file_name)
    print(f"{file_name} completed.")

def multi_threaded_download(url, file_name, num_threads=4):
    response = requests.head(url)
    file_size = int(response.headers['Content-Length'])
    chunk_size = file_size // num_threads

    pbar = tqdm(total=file_size, unit='B', unit_scale=True, desc=file_name)

    threads = []
    for i in range(num_threads):
        start = i * chunk_size
        end = start + chunk_size - 1 if i != num_threads - 1 else file_size - 1
        thread = Thread(target=download_chunk, args=(url, start, end, file_name, i, pbar))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    pbar.close()
    combine_chunks(file_name, num_threads)

url = "https://download01.fangcloud.com/download/cf77ea39048745bbb79137511668755c/782e8586534ae19c2307b4b25cf375028880ba0dc74fddc0471cbeedca80564b/GCC_part_2.tar.gz"
file_name = '/data/fyw/dataset/crowdcount/GCC/GCC_part_2.tar.gz'
multi_threaded_download(url, file_name, num_threads=4)
