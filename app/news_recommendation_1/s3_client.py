import os
import boto3
from tqdm import tqdm

from app.news_recommendation_1 import time_it


class S3Client:
    def __init__(self):
        self.s3_client = boto3.client('s3')

    @time_it
    def download_folder_from_s3(self, bucket_name, prefix, local_path):
        paginator = self.s3_client.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=bucket_name, Prefix=prefix)

        for page in pages:
            for obj in page.get('Contents', []):
                key = obj['Key']
                if key.endswith('/'):
                    continue
                self.__download_file(bucket_name, key, local_path)

    def __download_file(self, bucket_name, s3_key, local_path):
        local_file_path = os.path.join(local_path, s3_key.split('/')[-1])
        os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
        meta_data = self.s3_client.head_object(Bucket=bucket_name, Key=s3_key)
        total_length = int(meta_data.get('ContentLength', 0))

        with tqdm(total=total_length, desc=f'source: s3://{bucket_name}/{s3_key}', bar_format="{percentage:.1f}%|{bar:25} | {rate_fmt} | {desc}", unit='B', unit_scale=True, unit_divisor=1024) as pbar:
            with open(local_file_path, 'wb') as f:
                self.s3_client.download_fileobj(bucket_name, s3_key, f, Callback=pbar.update)
