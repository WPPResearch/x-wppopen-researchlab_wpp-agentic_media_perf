import os
import shutil
from datetime import datetime, timedelta, timezone

from google.api_core import exceptions
from google.cloud import storage

from src.utils.logger import logger


def upload_file_to_gcs(
    bucket_name: str,
    local_file_path: str,
    gcs_destination_blob_name: str,
) -> str | None:
    """
    Uploads a single file to a specific location within a GCS bucket.

    Args:
        bucket_name (str): The name of the GCS bucket.
        local_file_path (str): The path to the local file to upload.
        gcs_destination_blob_name (str): The destination blob name in the GCS bucket.

    Returns:
        Optional[str]: The GCS blob name if upload is successful, None otherwise.
    """
    # Ensure the local file exists
    if not os.path.isfile(local_file_path):
        logger.info(f"Error: Local file '{local_file_path}' not found.")
        return None

    try:
        # Initialize the GCS client
        storage_client = storage.Client()
        bucket = storage_client.get_bucket(bucket_name)

        logger.info(
            f"Starting upload of file '{local_file_path}' to 'gs://{bucket_name}/{gcs_destination_blob_name}'...",
        )

        # Upload the file
        blob = bucket.blob(gcs_destination_blob_name)
        blob.upload_from_filename(local_file_path)

        logger.info(
            f"Upload completed successfully: 'gs://{bucket_name}/{gcs_destination_blob_name}'",
        )
        return f"gs://{bucket_name}/{gcs_destination_blob_name}"

    except exceptions.NotFound:
        logger.error(f"Error: Bucket '{bucket_name}' not found.")
        return None
    except Exception as e:
        logger.error(f"An unexpected error occurred during upload: {e}")
        return None


def download_file_from_gcs(
    bucket_name: str,
    gcs_blob_name: str,
    local_destination_path: str,
) -> str | None:
    """
    Downloads a single file from a GCS bucket to a local path.

    Args:
        bucket_name (str): The name of the GCS bucket.
        gcs_blob_name (str): The blob name in the GCS bucket to download.
        local_destination_path (str): The local file path where the file will be saved.

    Returns:
        Optional[str]: The local file path if download is successful, None otherwise.
    """
    # Ensure the destination directory exists
    destination_dir = os.path.dirname(local_destination_path)
    if destination_dir and not os.path.exists(destination_dir):
        try:
            os.makedirs(destination_dir, exist_ok=True)
            logger.info(f"Created directory: '{destination_dir}'")
        except Exception as e:
            logger.error(f"Error creating directory '{destination_dir}': {e}")
            return None

    try:
        # Initialize the GCS client
        storage_client = storage.Client()
        bucket = storage_client.get_bucket(bucket_name)

        logger.info(
            f"Starting download of 'gs://{bucket_name}/{gcs_blob_name}' to '{local_destination_path}'...",
        )

        # Download the file
        blob = bucket.blob(gcs_blob_name)

        # Check if blob exists
        if not blob.exists():
            logger.error(
                f"Error: Blob '{gcs_blob_name}' not found in bucket '{bucket_name}'."
            )
            return None

        blob.download_to_filename(local_destination_path)

        logger.info(
            f"Download completed successfully: '{local_destination_path}'",
        )
        return local_destination_path

    except exceptions.NotFound:
        logger.error(f"Error: Bucket '{bucket_name}' not found.")
        return None
    except Exception as e:
        logger.error(f"An unexpected error occurred during download: {e}")
        return None


def upload_folder_to_gcs(
    bucket_name: str,
    local_folder_path: str,
    gcs_destination_folder: str = "",
) -> list[str]:
    """
    Uploads a folder and its content to a specific folder within a GCS bucket.

    Args:
        bucket_name (str): The name of the GCS bucket.
        local_folder_path (str): The path to the local folder to upload.
        gcs_destination_folder (str): The destination folder path within the GCS bucket.
                                      If empty, uploads to the root of the bucket.

    Returns:
        List[str]: A list of the GCS blob names for the uploaded files.
    """
    uploaded_blob_names = []
    # Ensure the local folder exists
    if not os.path.isdir(local_folder_path):
        logger.info(f"Error: Local folder '{local_folder_path}' not found.")
        return uploaded_blob_names

    try:
        # Initialize the GCS client
        storage_client = storage.Client()
        bucket = storage_client.get_bucket(bucket_name)

        logger.info(
            f"Starting upload of folder '{local_folder_path}' to 'gs://{bucket_name}/{gcs_destination_folder}'...",
        )

        # Walk through the local folder directory
        for local_root, _, filenames in os.walk(local_folder_path):
            for filename in filenames:
                local_file_path = os.path.join(local_root, filename)
                relative_path = os.path.relpath(local_file_path, local_folder_path)

                # Construct the full path for the object in the bucket
                if gcs_destination_folder:
                    # Use posixpath to ensure forward slashes, which GCS uses
                    gcs_blob_name = os.path.join(
                        gcs_destination_folder,
                        relative_path,
                    ).replace(os.path.sep, "/")
                else:
                    gcs_blob_name = relative_path.replace(os.path.sep, "/")

                # Upload the file
                blob = bucket.blob(gcs_blob_name)
                blob.upload_from_filename(local_file_path)
                uploaded_blob_names.append(gcs_blob_name)
                logger.info(
                    f"  - Uploaded '{local_file_path}' to 'gs://{bucket_name}/{gcs_blob_name}'",
                )

        logger.info("\nUpload completed successfully.")
        return uploaded_blob_names

    except exceptions.NotFound:
        logger.error(f"Error: Bucket '{bucket_name}' not found.")
        return []
    except Exception as e:
        logger.error(f"An unexpected error occurred during upload: {e}")
        return []


def generate_download_signed_url_v4(
    bucket_name: str,
    blob_name: str,
    key_json_path: str | os.PathLike,
) -> str:
    """Generates a v4 signed URL for downloading a blob.

    Note that this method requires a service account key file. You can not use this if you are using Application Default
    Credentials from Google Compute Engine or from the Google Cloud SDK.
    """

    storage_client = storage.Client.from_service_account_json(key_json_path)
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    url = blob.generate_signed_url(
        version="v4",
        # This URL is valid for 15 minutes
        expiration=timedelta(minutes=15),
        # Allow GET requests using this URL.
        method="GET",
    )
    logger.info(f"Generated GET signed URL: {url}")
    return url


def generate_signed_urls(
    bucket_name: str,
    blob_names: list[str] | None = None,
    expiration_minutes: int = 60,
) -> dict[str, str]:
    """
    Generates V4 signed URLs for specified objects in a GCS bucket.
    If no blob_names are provided, it generates URLs for all objects.

    Args:
        bucket_name (str): The name of the GCS bucket.
        blob_names (Optional[List[str]]): A list of blob names to generate URLs for.
        expiration_minutes (int): The duration in minutes for which the URL is valid.

    Returns:
        Dict[str, str]: A dictionary mapping object names to their signed URLs.
    """
    signed_urls = {}
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)

        logger.info(
            f"\nGenerating signed URLs for objects in bucket '{bucket_name}'...",
        )

        expiration_time = datetime.now(timezone.utc) + timedelta(
            minutes=expiration_minutes,
        )

        # If no specific blob names are provided, list all blobs in the bucket
        if blob_names is None:
            blobs = storage_client.list_blobs(bucket_name)
            target_blob_names = [blob.name for blob in blobs]
        else:
            target_blob_names = blob_names

        for blob_name in target_blob_names:
            try:
                blob = bucket.blob(blob_name)
                # Generate a v4 signed URL for reading the object
                url = blob.generate_signed_url(
                    version="v4",
                    expiration=expiration_time,
                    method="GET",
                )
                signed_urls[blob.name] = url
                logger.info(f"  - Generated URL for '{blob.name}'")
            except Exception as e:
                logger.error(f"    Could not generate signed URL for {blob.name}: {e}")

        logger.info(f"\nSigned URLs will expire in {expiration_minutes} minutes.")
        return signed_urls

    except exceptions.NotFound:
        logger.error(f"Error: Bucket '{bucket_name}' not found.")
        return {}
    except Exception as e:
        logger.error(f"An unexpected error occurred during URL generation: {e}")
        return {}


def delete_local_folder(local_folder_path: str):
    """
    Deletes a local folder and all of its contents.

    Args:
        local_folder_path (str): The path to the local folder to delete.
    """
    if not os.path.isdir(local_folder_path):
        logger.info(
            f"\nWarning: Local folder '{local_folder_path}' does not exist. Nothing to delete.",
        )
        return

    try:
        logger.info(f"\nDeleting local folder '{local_folder_path}'...")
        shutil.rmtree(local_folder_path)
        logger.error("Local folder deleted successfully.")
    except OSError as e:
        logger.error(f"Error deleting folder '{local_folder_path}': {e.strerror}")
    except Exception as e:
        logger.error(f"An unexpected error occurred during deletion: {e}")


if __name__ == "__main__":
    # --- Configuration ---
    GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME", "your-bucket-name")
    # Example: upload a local folder to GCS
    # upload_folder_to_gcs(GCS_BUCKET_NAME, "data/my-local-data", "project-files/uploads")
    # with open(os.path.join(LOCAL_FOLDER_TO_UPLOAD, "subfolder", "file2.txt"), "w") as f:
    #     f.write("This is the second file inside a subfolder.")
    # logger.info("Dummy files created.")
    #
    # # 2. Upload the folder to GCS and get the list of uploaded files
    # uploaded_files = upload_folder_to_gcs(
    #     GCS_BUCKET_NAME,
    #     LOCAL_FOLDER_TO_UPLOAD,
    #     GCS_DESTINATION_FOLDER,
    # )
    # Example: download a batch prediction output
    output_uri = "gs://your-bucket/agentic-media-perf/batch_jobs/predictions.jsonl"
    bucket_name = output_uri.split("gs://")[-1].split("/")[0]
    blob_path = "/".join(output_uri.split("gs://")[-1].split("/")[1:])
    local_path = download_file_from_gcs(
        bucket_name=bucket_name,
        gcs_blob_name=blob_path,
        local_destination_path="data/pred.jsonl",
    )
    print(f"Downloaded to: {local_path}")
