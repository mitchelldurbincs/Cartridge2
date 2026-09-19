"""In-memory S3 client with immutable-create and ETag preconditions."""

import io

from trainer.storage.artifact_codec import sha256_bytes


class S3Error(Exception):
    def __init__(self, code):
        super().__init__(code)
        self.response = {"Error": {"Code": code}}


class FakeS3:
    def __init__(self):
        self.objects = {}
        self.puts = []
        self.fail_key = None

    def get_object(self, *, Bucket, Key):
        del Bucket
        if Key not in self.objects:
            raise S3Error("NoSuchKey")
        data = self.objects[Key]
        return {
            "Body": io.BytesIO(data),
            "ETag": f'"{sha256_bytes(data)}"',
        }

    def put_object(self, **kwargs):
        key = kwargs["Key"]
        if key == self.fail_key:
            raise RuntimeError("S3 unavailable")
        data = kwargs["Body"]
        if hasattr(data, "read"):
            data = data.read()
        if kwargs.get("IfNoneMatch") == "*" and key in self.objects:
            raise S3Error("PreconditionFailed")
        if "IfMatch" in kwargs:
            existing = self.objects.get(key)
            etag = f'"{sha256_bytes(existing)}"' if existing is not None else None
            if kwargs["IfMatch"] != etag:
                raise S3Error("PreconditionFailed")
        self.objects[key] = bytes(data)
        self.puts.append(kwargs)

    def list_objects_v2(self, *, Bucket, Prefix, ContinuationToken=None):
        del Bucket
        if ContinuationToken is not None:
            raise AssertionError("fake listing is not paginated")
        return {
            "Contents": [{"Key": key} for key in sorted(self.objects) if key.startswith(Prefix)],
            "IsTruncated": False,
        }
