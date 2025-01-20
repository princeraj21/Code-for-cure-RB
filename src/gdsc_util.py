import pandas as pd
import boto3
import logging
import sys
import tarfile
import re
import os
from botocore.exceptions import ClientError
from tqdm import tqdm
from typing import Dict
from pathlib import Path
from config import DEFAULT_BUCKET, DEFAULT_REGION

PROJECT_DIR = Path(os.path.dirname((os.path.dirname(os.path.abspath(__file__)))))


def set_up_logging():
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[
            logging.FileHandler(str(PROJECT_DIR / "log.txt")),
            logging.StreamHandler(sys.stdout),
        ],
    )


def load_sections_df(file_path: str, sep: str = ";") -> pd.DataFrame:
    col_types = {
        "xmin": int,
        "xmax": int,
        "ymin": int,
        "ymax": int,
        "height": int,
        "width": int,
        "staining": "string",
        "study": "string",
    }
    df = pd.read_csv(file_path, sep=sep, dtype=col_types, index_col="section_id")
    return df


def download_file(
    s3_path: str, local_dir: str, bucket: str = DEFAULT_BUCKET, region: str = DEFAULT_REGION
):
    """
    Download a file to a local directory from s3 # TODO: Add example usage
    Args:
        s3_path (str): s3 path to file (relative to bucket name)
        local_dir (str): path of the local machine to save downloaded file # TODO: this is wrong
        bucket (str): bucket name #TODO: Explain / at the end
        region (str): Region in which the bucket is located
    Returns:
         returns file to given local path
    """
    logger = logging.getLogger(__name__)
    s3 = boto3.resource("s3", region_name=region)
    file_name = s3_path.split("/")[-1]
    f = f"{PROJECT_DIR}/{local_dir}/{file_name}"
    if os.path.isfile(f):
        logger.info(f"File {s3_path} already exists. Skipping download")
        return f
    s3.Bucket(bucket).download_file(s3_path, f)
    return f

def upload_file(local_file_path: str, s3_path: str = None, bucket: str = DEFAULT_BUCKET):
    """Upload a file to an S3 bucket

    :param local_file_path: File to upload
    :param s3_path: S3 object name. If not specified then file_name is used
    :param bucket: Bucket to upload to
    :return: True if file was uploaded, else False
    """
    logger = logging.getLogger(__name__)
    # If S3 object_name was not specified, use file_name
    if s3_path is None:
        s3_path = os.path.basename(local_file_path)

    # Upload the file
    s3_client = boto3.client('s3')
    try:
        response = s3_client.upload_file(local_file_path, bucket, s3_path)
    except ClientError as e:
        logger.error(e)
        return False
    return True


def download_directory(
    s3_path: str, local_dir: str, bucket: str = DEFAULT_BUCKET, region: str = DEFAULT_REGION
):
    """
    Download a folder to a local directory from s3 # TODO: Add example usage
    Args:
        s3_path (str): s3 path to folder (relative to bucket name)
        local_dir (str): path of the local machine to save downloaded file
        bucket (str): bucket name
        region (str): location
    """
    logger = logging.getLogger(__name__)
    s3_resource = boto3.resource("s3", region_name=region)
    bucket = s3_resource.Bucket(bucket)
    local_abs_path = PROJECT_DIR / local_dir

    logger.info(f"Downloading {s3_path} to {local_abs_path}")
    for obj in tqdm(bucket.objects.filter(Prefix=s3_path)):
        f = f"{local_abs_path}/{obj.key}"
        if not os.path.exists(os.path.dirname(f)):
            os.makedirs(os.path.dirname(f))
        if obj.key == s3_path or os.path.isfile(f):
            continue
        bucket.download_file(obj.key, f)
    logger.info(f"Downloaded {s3_path} to {local_abs_path}")


def download_and_extract_model(model_uri: str, local_dir: str):
    """Downloads model tarfile from model_uri (s3) and extracts it to download_location"""
    bucket, key = model_uri.replace("s3://", "").split("/", 1)
    exp_name = key.split("/")[1]
    local_model_dir = f"{local_dir}/{exp_name}"
    local_abs_model_dir = f"{PROJECT_DIR}/{local_model_dir}"
    os.makedirs(local_abs_model_dir, exist_ok=True)
    file_path = download_file(s3_path=key, bucket=bucket, local_dir=local_model_dir)
    with tarfile.open(file_path, mode="r") as tar:
        tar.extractall(local_abs_model_dir)
    return local_abs_model_dir


def upload_to_s3(local_path: str, s3_path: str, bucket: str):
    """
    Upload a file to s3 from a local directory # TODO: Add example usage
    Args:
        bucket (str): bucket name
        s3_path (str): s3 path to file (relative to bucket name)
        local_path (str): path of the local for upload # TODO: is this an absolute path?
    Returns:
        (str) remote path to file
    """
    client = boto3.client("s3")
    client.upload_file(local_path, bucket, s3_path)
    return f"s3://{bucket}/{s3_path}"


def get_params_from_config(pattern: str, text: str) -> Dict[str, str]:
    matches = re.findall(pattern, text)
    return {key: value.strip() for key, _, value in matches}


def extract_hyperparams(config: str = "tutorial_4_training.py") -> Dict[str, str]:
    """Extract the hyperparameters from file specified in config parameter"""
    with open(f"{PROJECT_DIR}/src/{config}", mode="r", encoding="utf-8") as f:
        text = f.read()
    pattern = r"cfg\.(.*)(\..*)? = ([^#^\n]*)"
    return get_params_from_config(pattern, text)


def create_encrypted_bucket(bucket_name: str) -> None:
    """Creates the default SM bucket, but encrypted"""
    s3 = boto3.client("s3")
    try:
        s3.create_bucket(Bucket=bucket_name)
    except ClientError as e:
        if e.response["Error"].get("Code") not in (
            "BucketAlreadyExists",
            "BucketAlreadyOwnedByYou",
        ):
            raise e

    s3.put_bucket_encryption(
        Bucket=bucket_name,
        ServerSideEncryptionConfiguration={
            "Rules": [{"ApplyServerSideEncryptionByDefault": {"SSEAlgorithm": "AES256"}}]
        },
    )

    logging.info(f"Successfully created encrypted bucket: {bucket_name}")
