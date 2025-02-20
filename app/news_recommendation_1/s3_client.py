import os
import boto3
from tqdm import tqdm

from app.news_recommendation_1 import time_it


class S3Client:
    """
    A class to handle interactions with Amazon S3.

    Attributes
    ----------
    s3_client : boto3.client
        A low-level client representing Amazon Simple Storage Service (S3).

    Methods
    -------
    __init__(self)
        Initializes the S3Client with a boto3 S3 client.
    download_folder_from_s3(self, bucket_name: str, prefix: str, local_path: str) -> None
        Downloads a folder from S3 to a local path.
    __download_file(self, bucket_name: str, s3_key: str, local_path: str) -> None
        Downloads a single file from S3 to a local path.

    Usage Examples
    --------------
    >>> s3_client = S3Client()
    >>> s3_client.download_folder_from_s3('my-bucket', 'my-prefix/', '/local/path')
    """
    def __init__(self):
        """
        Initializes the S3Client with a boto3 S3 client.
        """
        self.s3_client = boto3.client('s3')

    @time_it
    def download_folder_from_s3(self, bucket_name, prefix, local_path):
        """
        Downloads a folder from S3 to a local path.

        This method paginates through the objects in the specified S3 bucket and prefix, and downloads each file to the local path.

        Parameters
        ----------
        bucket_name : str
            The name of the S3 bucket.
        prefix : str
            The prefix (folder path) in the S3 bucket.
        local_path : str
            The local path where the files will be downloaded.

        Returns
        -------
        None

        Example
        -------
        >>> s3_client = S3Client()
        >>> s3_client.download_folder_from_s3('my-bucket', 'my-prefix/', '/local/path')
        """
        paginator = self.s3_client.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=bucket_name, Prefix=prefix)

        for page in pages:
            for obj in page.get('Contents', []):
                key = obj['Key']
                if key.endswith('/'):
                    continue
                self.__download_file(bucket_name, key, local_path)

    def __download_file(self, bucket_name, s3_key, local_path):
        """
        Downloads a single file from S3 to a local path.

        This method downloads a single file from the specified S3 bucket and key to the local path, displaying a progress bar during the download.

        Parameters
        ----------
        bucket_name : str
            The name of the S3 bucket.
        s3_key : str
            The key (file path) in the S3 bucket.
        local_path : str
            The local path where the file will be downloaded.

        Returns
        -------
        None

        Example
        -------
        >>> s3_client = S3Client()
        >>> s3_client._S3Client__download_file('my-bucket', 'my-prefix/my-file.txt', '/local/path')
        """
        local_file_path = os.path.join(local_path, s3_key.split('/')[-1])
        os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
        meta_data = self.s3_client.head_object(Bucket=bucket_name, Key=s3_key)
        total_length = int(meta_data.get('ContentLength', 0))

        with tqdm(total=total_length, desc=f'source: s3://{bucket_name}/{s3_key}', bar_format="{percentage:.1f}%|{bar:25} | {rate_fmt} | {desc}", unit='B', unit_scale=True, unit_divisor=1024) as pbar:
            with open(local_file_path, 'wb') as f:
                self.s3_client.download_fileobj(bucket_name, s3_key, f, Callback=pbar.update)
