import logging
import boto3
from botocore.exceptions import ClientError
import os
import argparse


def upload_file(s3_client,file_name, bucket, object_name=None):
    """Upload a file to an S3 bucket

    :param file_name: File to upload
    :param bucket: Bucket to upload to
    :param object_name: S3 object name. If not specified then file_name is used
    :return: True if file was uploaded, else False
    """

    # If S3 object_name was not specified, use file_name
    if object_name is None:
        object_name = os.path.basename(file_name)

    # Upload the file
    # s3_client = boto3.client('s3')
    try:
        response = s3_client.upload_file(file_name, bucket, object_name)
    except ClientError as e:
        logging.error(e)
        return False
    return True

import boto3

ACCESS_KEY=os.getenv('ACCESS_KEY')
SECRET_KEY=os.getenv('SECRET_KEY')

def generate_s3_client():

    client = boto3.client(
        's3',
        aws_access_key_id=ACCESS_KEY,
        aws_secret_access_key=SECRET_KEY,
    )
    return client

if __name__=="__main__":

 
    parser = argparse.ArgumentParser(description="",formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-up", "--upload_file", help="path to file to be uploaded")
    parser.add_argument("-dw", "--download_file",  help="file to be downloaded")
    parser.add_argument("-bk", "--bucket", help="aws bucket")
    args = parser.parse_args()
    config = vars(args)

    client=generate_s3_client()
    bucket=config['bucket']

    if config['download_file']:
        pass
    
    if config['upload_file']:
        file2upload=config['upload_file']
        upload_file(client,file2upload,bucket)
        

  
    print(config)