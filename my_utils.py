import errno
import os

from apiclient.http import MediaIoBaseDownload
from google.cloud import storage

def download_from_bucket_to_local(gcs_service, bucket, gcs_path, local_path):
    if not os.path.exists(os.path.dirname(local_path)):
        try:
            os.makedirs(os.path.dirname(local_path))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

    with open(local_path, 'wb') as f:
        request = gcs_service.objects().get_media(bucket=bucket,
                                                  object=gcs_path)
        media = MediaIoBaseDownload(f, request)

        done = False
        while not done:
            # _ is a placeholder for a progress object that we ignore.
            # (Our file is small, so we skip reporting progress.)
            status, done = media.next_chunk()
            print(status)


def upload_blob(bucket_name, source_file_name, destination_blob_name):
    """Upload a file to the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_name)

    print('File {] uploaded to {}.'.format(
        source_file_name,
        destination_blob_name))