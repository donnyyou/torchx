# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""A limited reimplementation of the TensorFlow FileIO API.

The TensorFlow version wraps the C++ FileSystem API.  Here we provide a
pure Python implementation, limited to the features required for
TensorBoard.  This allows running TensorBoard without depending on
TensorFlow for file operations.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import socket
from collections import namedtuple
from urllib import parse, request
import glob as py_glob
import io
import os
import shutil
import six
import sys
import tempfile
import functools
import requests

try:
    import botocore.exceptions
    import boto3

    S3_ENABLED = True
except ImportError:
    S3_ENABLED = False

try:
    import oss2

    OSS_ENABLED = True
except ImportError:
    OSS_ENABLED = False

if sys.version_info < (3, 0):
    # In Python 2 FileExistsError is not defined and the
    # error manifests it as OSError.
    FileExistsError = OSError

from ._utils import compat, errors

# A good default block size depends on the system in question.
# A somewhat conservative default chosen here.
_DEFAULT_BLOCK_SIZE = 128 * 1024 * 1024

# Default pool size for HTTP session pool.
_DEFAULT_CONNECTION_POOL_SIZE = 10
_DEFAULT_CONNECTION_RETRIES = 3

# Registry of filesystems by prefix.
#
# Currently supports "s3://" URLs for S3 based on boto3 and falls
# back to local filesystem.
_REGISTERED_FILESYSTEMS = {}


def get_path_prefix(filename):
    filename = compat.as_str_any(filename)
    prefix = ""
    index = filename.find("://")
    if index >= 0:
        prefix = filename[:index]
    return prefix


def register_filesystem(prefix, filesystem):
    if ":" in prefix:
        raise ValueError("Filesystem prefix cannot contain a :")
    _REGISTERED_FILESYSTEMS[prefix] = filesystem


def get_filesystem(filename):
    """Return the registered filesystem for the given file."""
    prefix = get_path_prefix(filename)
    fs = _REGISTERED_FILESYSTEMS.get(prefix, None)
    if fs is None:
        raise ValueError("No recognized filesystem for prefix %s" % prefix)
    return fs


# Data returned from the Stat call.
StatData = namedtuple("StatData", ["length"])


class LocalFileSystem(object):
    """Provides local fileystem access."""

    def exists(self, filename):
        """Determines whether a path exists or not."""
        return os.path.exists(compat.as_bytes(filename))

    def join(self, path, *paths):
        """Join paths with path delimiter."""
        return os.path.join(path, *paths)

    def read(self, filename, binary_mode=False, size=None, continue_from=None):
        """Reads contents of a file to a string.

        Args:
            filename: string, a path
            binary_mode: bool, read as binary if True, otherwise text
            size: int, number of bytes or characters to read, otherwise
                read all the contents of the file (from the continuation
                marker, if present).
            continue_from: An opaque value returned from a prior invocation of
                `read(...)` marking the last read position, so that reading
                may continue from there.  Otherwise read from the beginning.

        Returns:
            A tuple of `(data, continuation_token)` where `data' provides either
            bytes read from the file (if `binary_mode == true`) or the decoded
            string representation thereof (otherwise), and `continuation_token`
            is an opaque value that can be passed to the next invocation of
            `read(...) ' in order to continue from the last read position.
        """
        mode = "rb" if binary_mode else "r"
        encoding = None if binary_mode else "utf8"
        if not exists(filename):
            raise errors.NotFoundError(
                None, None, "Not Found: " + compat.as_text(filename)
            )
        offset = None
        if continue_from is not None:
            offset = continue_from.get("byte_offset", None)
        with io.open(filename, mode, encoding=encoding) as f:
            if offset is not None:
                f.seek(offset)
            data = f.read(size)
            # The new offset may not be `offset + len(data)`, due to decoding
            # and newline translation.
            # So, just measure it in whatever terms the underlying stream uses.
            continuation_token = {"byte_offset": f.tell()}
            return (data, continuation_token)

    def write(self, filename, file_content, binary_mode=False):
        """Writes string file contents to a file, overwriting any existing
        contents.

        Args:
            filename: string, a path
            file_content: string, the contents
            binary_mode: bool, write as binary if True, otherwise text

        Returns:
            The number of bytes actually written.
        """
        return self._write(filename, file_content, "wb" if binary_mode else "w")

    def append(self, filename, file_content, binary_mode=False):
        """Append string file contents to a file.

        Args:
            filename: string, a path
            file_content: string, the contents to append
            binary_mode: bool, write as binary if True, otherwise text
        """
        return self._write(filename, file_content, "ab" if binary_mode else "a")

    def _write(self, filename, file_content, mode):
        encoding = None if "b" in mode else "utf8"
        with io.open(filename, mode, encoding=encoding) as f:
            compatify = compat.as_bytes if "b" in mode else compat.as_text
            return f.write(compatify(file_content))

    def glob(self, filename):
        """Returns a list of files that match the given pattern(s)."""
        if isinstance(filename, six.string_types):
            return [
                # Convert the filenames to string from bytes.
                compat.as_str_any(matching_filename)
                for matching_filename in py_glob.glob(compat.as_bytes(filename))
            ]
        else:
            return [
                # Convert the filenames to string from bytes.
                compat.as_str_any(matching_filename)
                for single_filename in filename
                for matching_filename in py_glob.glob(
                    compat.as_bytes(single_filename)
                )
            ]

    def isdir(self, dirname):
        """Returns whether the path is a directory or not."""
        return os.path.isdir(compat.as_bytes(dirname))

    def listdir(self, dirname):
        """Returns a list of entries contained within a directory."""
        if not self.isdir(dirname):
            raise errors.NotFoundError(None, None, "Could not find directory")

        entries = os.listdir(compat.as_str_any(dirname))
        entries = [compat.as_str_any(item) for item in entries]
        return entries

    def makedirs(self, path, exist_ok=False):
        """Creates a directory and all parent/intermediate directories."""
        try:
            os.makedirs(path, exist_ok=exist_ok)
        except FileExistsError:
            raise errors.AlreadyExistsError(
                None, None, "Directory already exists"
            )

    def stat(self, filename):
        """Returns file statistics for a given path."""
        # NOTE: Size of the file is given by .st_size as returned from
        # os.stat(), but we convert to .length
        try:
            file_length = os.stat(compat.as_bytes(filename)).st_size
        except OSError:
            raise errors.NotFoundError(None, None, "Could not find file")
        return StatData(file_length)

    def copy(self, src, dst):
        shutil.copy2(src, dst)

    def remove(self, filename):
        """Remove a file"""
        try:
            os.remove(filename)
        except IsADirectoryError:
            raise errors.InvalidArgumentError(
                None, None, "{} Is an directory, not file".format(filename)
            )

    def rmtree(self, dirname):
        """Delete an entire directory tree"""
        shutil.rmtree(dirname)


class S3FileSystem(object):
    """Provides filesystem access to S3."""

    def __init__(self):
        if not boto3:
            raise ImportError("boto3 must be installed for S3 support.")
        self._s3_endpoint = os.environ.get("S3_ENDPOINT", None)

    def bucket_and_path(self, url):
        """Split an S3-prefixed URL into bucket and path."""
        url = compat.as_str_any(url)
        if url.startswith("s3://"):
            url = url[len("s3://"):]
        idx = url.index("/")
        bucket = url[:idx]
        path = url[(idx + 1):]
        return bucket, path

    def exists(self, filename):
        """Determines whether a path exists or not."""
        client = boto3.client("s3", endpoint_url=self._s3_endpoint)
        bucket, path = self.bucket_and_path(filename)
        r = client.list_objects(Bucket=bucket, Prefix=path, Delimiter="/")
        if r.get("Contents") or r.get("CommonPrefixes"):
            return True
        return False

    def join(self, path, *paths):
        """Join paths with a slash."""
        return "/".join((path,) + paths)

    def read(self, filename, binary_mode=False, size=None, continue_from=None):
        """Reads contents of a file to a string.

        Args:
            filename: string, a path
            binary_mode: bool, read as binary if True, otherwise text
            size: int, number of bytes or characters to read, otherwise
                read all the contents of the file (from the continuation
                marker, if present).
            continue_from: An opaque value returned from a prior invocation of
                `read(...)` marking the last read position, so that reading
                may continue from there.  Otherwise read from the beginning.

        Returns:
            A tuple of `(data, continuation_token)` where `data' provides either
            bytes read from the file (if `binary_mode == true`) or the decoded
            string representation thereof (otherwise), and `continuation_token`
            is an opaque value that can be passed to the next invocation of
            `read(...) ' in order to continue from the last read position.
        """
        s3 = boto3.resource("s3", endpoint_url=self._s3_endpoint)
        bucket, path = self.bucket_and_path(filename)
        args = {}

        # For the S3 case, we use continuation tokens of the form
        # {byte_offset: number}
        offset = 0
        if continue_from is not None:
            offset = continue_from.get("byte_offset", 0)

        endpoint = ""
        if size is not None:
            # TODO(orionr): This endpoint risks splitting a multi-byte
            # character or splitting \r and \n in the case of CRLFs,
            # producing decoding errors below.
            endpoint = offset + size

        if offset != 0 or endpoint != "":
            # Asked for a range, so modify the request
            args["Range"] = "bytes={}-{}".format(offset, endpoint)

        try:
            stream = s3.Object(bucket, path).get(**args)["Body"].read()
        except botocore.exceptions.ClientError as exc:
            if exc.response["Error"]["Code"] == "416":
                if size is not None:
                    # Asked for too much, so request just to the end. Do this
                    # in a second request so we don't check length in all cases.
                    client = boto3.client("s3", endpoint_url=self._s3_endpoint)
                    obj = client.head_object(Bucket=bucket, Key=path)
                    content_length = obj["ContentLength"]
                    endpoint = min(content_length, offset + size)
                if offset == endpoint:
                    # Asked for no bytes, so just return empty
                    stream = b""
                else:
                    args["Range"] = "bytes={}-{}".format(offset, endpoint)
                    stream = s3.Object(bucket, path).get(**args)["Body"].read()
            else:
                raise
        # `stream` should contain raw bytes here (i.e., there has been neither
        # decoding nor newline translation), so the byte offset increases by
        # the expected amount.
        continuation_token = {"byte_offset": (offset + len(stream))}
        if binary_mode:
            return (bytes(stream), continuation_token)
        else:
            return (stream.decode("utf-8"), continuation_token)

    def write(self, filename, file_content, binary_mode=False):
        """Writes string file contents to a file.

        Args:
            filename: string, a path
            file_content: string, the contents
            binary_mode: bool, write as binary if True, otherwise text

        Returns:
            The number of bytes actually written.
        """
        client = boto3.client("s3", endpoint_url=self._s3_endpoint)
        bucket, path = self.bucket_and_path(filename)
        # Always convert to bytes for writing
        if binary_mode:
            if not isinstance(file_content, six.binary_type):
                raise TypeError("File content type must be bytes")
        else:
            file_content = compat.as_bytes(file_content)
        client.put_object(Body=file_content, Bucket=bucket, Key=path)
        return len(file_content)

    def glob(self, filename):
        """Returns a list of files that match the given pattern(s)."""
        # Only support prefix with * at the end and no ? in the string
        star_i = filename.find("*")
        quest_i = filename.find("?")
        if quest_i >= 0:
            raise NotImplementedError(
                "{} not supported by compat glob".format(filename)
            )
        if star_i != len(filename) - 1:
            # Just return empty so we can use glob from directory watcher
            #
            # TODO: Remove and instead handle in GetLogdirSubdirectories.
            # However, we would need to handle it for all non-local registered
            # filesystems in some way.
            return []
        filename = filename[:-1]
        client = boto3.client("s3", endpoint_url=self._s3_endpoint)
        bucket, path = self.bucket_and_path(filename)
        p = client.get_paginator("list_objects")
        keys = []
        for r in p.paginate(Bucket=bucket, Prefix=path):
            for o in r.get("Contents", []):
                key = o["Key"][len(path):]
                if key:  # Skip the base dir, which would add an empty string
                    keys.append(filename + key)
        return keys

    def isdir(self, dirname):
        """Returns whether the path is a directory or not."""
        client = boto3.client("s3", endpoint_url=self._s3_endpoint)
        bucket, path = self.bucket_and_path(dirname)
        if not path.endswith("/"):
            path += "/"  # This will now only retrieve subdir content
        r = client.list_objects(Bucket=bucket, Prefix=path, Delimiter="/")
        if r.get("Contents") or r.get("CommonPrefixes"):
            return True
        return False

    def listdir(self, dirname):
        """Returns a list of entries contained within a directory."""
        client = boto3.client("s3", endpoint_url=self._s3_endpoint)
        bucket, path = self.bucket_and_path(dirname)
        p = client.get_paginator("list_objects")
        if not path.endswith("/"):
            path += "/"  # This will now only retrieve subdir content
        keys = []
        for r in p.paginate(Bucket=bucket, Prefix=path, Delimiter="/"):
            keys.extend(
                o["Prefix"][len(path): -1] for o in r.get("CommonPrefixes", [])
            )
            for o in r.get("Contents", []):
                key = o["Key"][len(path):]
                if key:  # Skip the base dir, which would add an empty string
                    keys.append(key)
        return keys

    def makedirs(self, dirname, exist_ok=False):
        """Creates a directory and all parent/intermediate directories."""
        if not exist_ok and self.exists(dirname):
            raise errors.AlreadyExistsError(
                None, None, "Directory already exists"
            )
        client = boto3.client("s3", endpoint_url=self._s3_endpoint)
        bucket, path = self.bucket_and_path(dirname)
        if not path.endswith("/"):
            path += "/"  # This will make sure we don't override a file
        client.put_object(Body="", Bucket=bucket, Key=path)

    def stat(self, filename):
        """Returns file statistics for a given path."""
        # NOTE: Size of the file is given by ContentLength from S3,
        # but we convert to .length
        client = boto3.client("s3", endpoint_url=self._s3_endpoint)
        bucket, path = self.bucket_and_path(filename)
        try:
            obj = client.head_object(Bucket=bucket, Key=path)
            return StatData(obj["ContentLength"])
        except botocore.exceptions.ClientError as exc:
            if exc.response["Error"]["Code"] == "404":
                raise errors.NotFoundError(None, None, "Could not find file")
            else:
                raise

    def remove(self, filename):
        raise NotImplementedError("Coming soon.")

    def rmtree(self, dirname):
        raise NotImplementedError("Coming soon.")


class OSSFileSystem(object):
    """Provides filesystem access to OSS."""

    def __init__(self):
        if not oss2:
            raise ImportError("oss2 must be installed for OSS support.")
        self._oss_endpoint = None
        self._oss_auth = None

        self._sessions = {}

    def bucket_and_path(self, url):
        url = compat.as_str_any(url)

        if url.startswith("oss://"):
            path = url[len("oss://"):]

        configs = {
            "endpoint": os.getenv("OSS_ENDPOINT", None),
            "accessKeyID": os.getenv("OSS_ACCESS_KEY_ID", None),
            "accessKeySecret": os.getenv("OSS_ACCESS_KEY_SECRET", None),
            "securityToken": os.getenv("OSS_SECURITY_TOKEN", None),
        }

        bucket, path = path.split('/', maxsplit=1)
        if '?' in bucket:
            bucket, config = bucket.split('?', maxsplit=1)
            for pair in config.split('&'):
                k, v = pair.split('=', maxsplit=1)
                configs[k] = v

        # Get or create cached Session object
        session = self._sessions.setdefault(
            f"{bucket}@{os.getpid()}",
            oss2.Session(),
        )

        # Create Bucket object
        if configs["accessKeyID"] is None or configs["accessKeySecret"] is None:
            auth = oss2.AnonymousAuth()
        elif configs["securityToken"] is not None:
            auth = oss2.StsAuth(
                configs["accessKeyID"],
                configs["accessKeySecret"],
                configs["securityToken"]
            )
        else:
            auth = oss2.Auth(configs["accessKeyID"], configs["accessKeySecret"])

        bucket = oss2.Bucket(auth, configs["endpoint"], bucket, session=session)

        return bucket, path

    def exists(self, filename):
        """Determines whether a path exists or not."""
        bucket, path = self.bucket_and_path(filename)
        r = bucket.list_objects(prefix=path, delimiter="/", max_keys=2)
        if r.object_list or r.prefix_list:
            return True
        return False

    def join(self, path, *paths):
        """Join paths with a slash."""
        return os.path.join(path, *paths)

    def read(self, filename, binary_mode=False, size=None, continue_from=None):
        """Reads contents of a file to a string.

        Args:
            filename: string, a path
            binary_mode: bool, read as binary if True, otherwise text
            size: int, number of bytes or characters to read, otherwise
                read all the contents of the file (from the continuation
                marker, if present).
            continue_from: An opaque value returned from a prior invocation of
                `read(...)` marking the last read position, so that reading
                may continue from there.  Otherwise read from the beginning.

        Returns:
            A tuple of `(data, continuation_token)` where `data' provides either
            bytes read from the file (if `binary_mode == true`) or the decoded
            string representation thereof (otherwise), and `continuation_token`
            is an opaque value that can be passed to the next invocation of
            `read(...) ' in order to continue from the last read position.
        """
        bucket, path = self.bucket_and_path(filename)
        byte_range = None

        # For the OSS case, we use continuation tokens of the form
        # {byte_offset: number}
        offset = 0
        if continue_from is not None:
            offset = continue_from.get("byte_offset", 0)

        endpoint = None
        if size is not None:
            # TODO(orionr): This endpoint risks splitting a multi-byte
            # character or splitting \r and \n in the case of CRLFs,
            # producing decoding errors below.
            endpoint = offset + size

        if offset != 0 or endpoint is not None:
            # Asked for a range, so pass the range
            byte_range = (offset, endpoint)

        try:
            # Ref: https://help.aliyun.com/document_detail/88443.html?spm=a2c4g.11186623.6.905.119d1df24rO6nc#d7e41
            headers = {'x-oss-range-behavior': 'standard'}
            stream = bucket.get_object(path, byte_range=byte_range, headers=headers).read()
        except oss2.exceptions.ServerError as e:
            if e.status == 416:
                # In these 2 cases, offset >= file length, so just return empty.
                #   1. already read to the file end
                #   2. seek to an invalid position
                stream = b''
            else:
                raise

        # `stream` should contain raw bytes here (i.e., there has been neither
        # decoding nor newline translation), so the byte offset increases by
        # the expected amount.
        continuation_token = {"byte_offset": (offset + len(stream))}
        if binary_mode:
            return bytes(stream), continuation_token
        else:
            return stream.decode("utf-8"), continuation_token

    def write(self, filename, file_content, binary_mode=False):
        """Writes string file contents to a file.

        Args:
            filename: string, a path
            file_content: string, the contents
            binary_mode: bool, write as binary if True, otherwise text

        Returns:
            The number of bytes actually written.
        """
        bucket, path = self.bucket_and_path(filename)
        # Always convert to bytes for writing
        if binary_mode:
            if not isinstance(file_content, six.binary_type):
                raise TypeError("File content type must be bytes")
        else:
            file_content = compat.as_bytes(file_content)
        bucket.put_object(path, file_content)
        return len(file_content)

    def glob(self, filename):
        """Returns a list of files that match the given pattern(s)."""
        # Only support prefix with * at the end and no ? in the string
        star_i = filename.find("*")
        quest_i = filename.find("?")
        if quest_i >= 0:
            raise NotImplementedError(
                "{} not supported by compat glob".format(filename)
            )
        postfix = filename[star_i + 1:]
        filename = filename[:star_i]
        bucket, path = self.bucket_and_path(filename)
        keys = []
        # The following instruction is copied from OSS document.
        # 通过delimiter和prefix两个参数可以模拟文件夹功能：
        # + 如果设置prefix为某个文件夹名称，则会列举以此prefix开头的文件，即该文件夹下所有的文件和子文件夹（目录）。
        # + 如果再设置delimiter为正斜线（/），则只列举该文件夹下的文件和子文件夹（目录）名称，子文件夹下的文件和文件夹不显示。
        for o in oss2.ObjectIterator(bucket, prefix=path):
            key = o.key[len(path):]
            # Since o.is_prefix() will always return False
            # if we set param `delimiter` of `ObjectIterator`
            # to empty, we have to use tailing '/' to identity
            # the directories.
            if key and key.endswith(postfix):
                keys.append(filename + key)
        return keys

    def isdir(self, dirname):
        """Returns whether the path is a directory or not."""
        bucket, path = self.bucket_and_path(dirname)
        if not path.endswith("/"):
            path += "/"  # This will now only retrieve subdir content
        r = bucket.list_objects(prefix=path, delimiter="/", max_keys=2)
        if r.object_list or r.prefix_list:
            return True
        return False

    def listdir(self, dirname):
        """Returns a list of entries contained within a directory."""
        bucket, path = self.bucket_and_path(dirname)
        if not path.endswith("/"):
            path += "/"  # This will now only retrieve subdir content
        keys = []
        # The following instruction is copied from OSS document.
        # 通过delimiter和prefix两个参数可以模拟文件夹功能：
        # + 如果设置prefix为某个文件夹名称，则会列举以此prefix开头的文件，即该文件夹下所有的文件和子文件夹（目录）。
        # + 如果再设置delimiter为正斜线（/），则只列举该文件夹下的文件和子文件夹（目录）名称，子文件夹下的文件和文件夹不显示。
        for o in oss2.ObjectIterator(bucket, prefix=path, delimiter='/'):
            key = o.key[len(path):]
            # Since we set param `delimiter` to '/', o.is_prefix() will
            # return True if it is a directory.
            if o.is_prefix():
                keys.append(key[:-1])
            elif key:
                keys.append(key)
        return keys

    def makedirs(self, dirname, exist_ok=False):
        """Creates a directory and all parent/intermediate directories."""
        if not exist_ok and self.exists(dirname):
            raise errors.AlreadyExistsError(
                None, None, "Directory already exists"
            )
        bucket, path = self.bucket_and_path(dirname)
        if not path.endswith("/"):
            path += "/"  # This will make sure we don't override a file
        bucket.put_object(path, "")

    def stat(self, filename):
        """Returns file statistics for a given path."""
        # NOTE: Size of the file is given by ContentLength from S3,
        # but we convert to .length
        bucket, path = self.bucket_and_path(filename)
        try:
            meta = bucket.get_object_meta(path)
            length = int(meta.headers['Content-Length'])
            return StatData(length)
        except oss2.exceptions.NoSuchKey:
            raise errors.NotFoundError(None, None, "Could not find file")

    def copy(self, src, dst):
        """Copy a file"""
        src_bucket, src_path = self.bucket_and_path(src)
        dst_bucket, dst_path = self.bucket_and_path(dst)

        dst_bucket.copy_object(src_bucket.bucket_name, src_path, dst_path)

    def remove(self, filename):
        """Remove a file"""
        bucket, path = self.bucket_and_path(filename)
        bucket.delete_object(path)

    def rmtree(self, dirname):
        """Delete an entire directory tree"""
        if not dirname.endswith("/"):
            dirname += "/"
        for filename in self.glob(dirname + "*"):
            self.remove(filename)
        self.remove(dirname)  # remove root dir


class HTTPFileSystem(object):
    def __init__(self):
        self.session = requests.Session()

        psize = _DEFAULT_CONNECTION_POOL_SIZE
        max_retries = _DEFAULT_CONNECTION_RETRIES
        for poctotal in ("http://", "https://"):
            adapter = requests.adapters.HTTPAdapter(
                pool_connections=psize,
                pool_maxsize=2 * psize,
                max_retries=max_retries
            )
            self.session.mount(poctotal, adapter)

    @staticmethod
    def _make_range_string(start=None, end=None):
        if start is None and end is None:
            return ''
        start_s = str(start) if start is not None else ''
        end_s = str(end) if end is not None else ''
        return 'bytes={}-{}'.format(start_s, end_s)

    @staticmethod
    def _download(filename, headers=None):
        filename = compat.as_str_any(filename)
        req = request.Request(filename)
        if headers is not None:
            for k, v in headers.items():
                req.add_header(k, v)
        # urlopen has better performance than requests
        try:
            resp = request.urlopen(req)
            return resp.status, resp.read()
        except request.HTTPError as e:
            return e.code, b''

    def read(self, filename, binary_mode=False, size=None, continue_from=None):
        """Reads contents of a HTTP url to a string.

        Args:
            filename: string, a path
            binary_mode: bool, read as binary if True, otherwise text
            size: int, number of bytes or characters to read, otherwise
                read all the contents of the file (from the continuation
                marker, if present).
            continue_from: An opaque value returned from a prior invocation of
                `read(...)` marking the last read position, so that reading
                may continue from there.  Otherwise read from the beginning.

        Returns:
            A tuple of `(data, continuation_token)` where `data' provides either
            bytes read from the file (if `binary_mode == true`) or the decoded
            string representation thereof (otherwise), and `continuation_token`
            is an opaque value that can be passed to the next invocation of
            `read(...) ' in order to continue from the last read position.
        """
        headers = None

        # For the OSS case, we use continuation tokens of the form
        # {byte_offset: number}
        offset = 0
        if continue_from is not None:
            offset = continue_from.get("byte_offset", 0)

        endpoint = None
        if size is not None:
            # TODO(orionr): This endpoint risks splitting a multi-byte
            # character or splitting \r and \n in the case of CRLFs,
            # producing decoding errors below.
            endpoint = offset + size

        if offset != 0 or endpoint is not None:
            # Asked for a range, so pass the range
            headers = {
                "Range": self._make_range_string(offset, endpoint),
                'x-oss-range-behavior': 'standard'
            }

        # resp = self.session.get(filename, headers=headers, stream=True)
        # content = resp.raw.read()
        status_code, content = self._download(filename, headers)

        # This part is very annoying, since different HTTP servers treat partial read differently.
        # We just assume all HTTP server strictly obey the standard HTTP protocol.
        stream = b''
        if status_code == 200:
            # Case 1: no range specified, (offset, endpoint) = (0, None)
            # Case 2: server doesn't support partial read, return partial
            stream = content[offset:endpoint]
        elif status_code == 206:
            # 206 Partial Content
            stream = content
        elif status_code == 416:
            if size is not None:
                # Asked for too much, so request just to the end. Do this
                # in a second request so we don't check length in all cases.
                try:
                    length = self.stat(filename).length
                    endpoint = min(length, offset + size)
                except errors.UnimplementedError:
                    # We assume offset is out-of-range.
                    endpoint = None
            if endpoint is None or offset >= endpoint:
                # offset exceeds file length
                stream = b''
            else:
                headers = {"Range": "bytes={}-{}".format(offset, endpoint)}
                _, stream = self._download(filename, headers=headers)
        else:
            raise errors.InternalError(None, None, "Server error: {}".format(status_code))

        # `stream` should contain raw bytes here (i.e., there has been neither
        # decoding nor newline translation), so the byte offset increases by
        # the expected amount.
        continuation_token = {"byte_offset": (offset + len(stream))}
        if binary_mode:
            return bytes(stream), continuation_token
        else:
            return stream.decode("utf-8"), continuation_token

    def stat(self, filename):
        """Returns file statistics for a given path."""
        resp = self.session.head(filename)

        # Check if file exists
        if resp.status_code == 404:
            raise errors.NotFoundError(None, None, "Could not find file")
        elif resp.status_code != 200:
            raise errors.InternalError(None, None, "Server error: {}".format(resp.reason))

        # Try some different key name
        for key in ('Content-Length', 'ContentLength'):
            try:
                length = int(resp.headers[key])
                return StatData(length)
            except KeyError:
                continue
            else:
                break
        else:
            raise errors.UnimplementedError(
                None, None, "Server does not support stat"
            )


class DFFileSystem(HTTPFileSystem):
    def __init__(self):
        super().__init__()
        self.endpoint = os.environ.get('DF_ENDPOINT', None)
        if self.endpoint is None:
            with open('/etc/hostinfo', 'r') as f:
                for line in f:
                    ip = line.strip()
                    try:
                        socket.inet_aton(ip)
                    except socket.error:
                        continue
                    else:
                        self.endpoint = f"http://{ip}:39999/api/v1/paths/"
                        break

    def read(self, filename, binary_mode=False, size=None, continue_from=None):
        """Reads contents of a HTTP url to a string.

        Args:
            filename: string, a path
            binary_mode: bool, read as binary if True, otherwise text
            size: int, number of bytes or characters to read, otherwise
                read all the contents of the file (from the continuation
                marker, if present).
            continue_from: An opaque value returned from a prior invocation of
                `read(...)` marking the last read position, so that reading
                may continue from there.  Otherwise read from the beginning.

        Returns:
            A tuple of `(data, continuation_token)` where `data' provides either
            bytes read from the file (if `binary_mode == true`) or the decoded
            string representation thereof (otherwise), and `continuation_token`
            is an opaque value that can be passed to the next invocation of
            `read(...) ' in order to continue from the last read position.
        """
        filename = compat.as_str_any(filename)
        if filename.startswith("df://"):
            filename = filename[len("df://"):]
        if filename[0] != "/":
            filename = "/" + filename
        url = self.endpoint + parse.quote(filename) + "/download-file"
        return super().read(url, binary_mode, size, continue_from)


DF_ENABLED = ("DF_ENDPOINT" in os.environ) or os.path.exists("/etc/hostinfo")

register_filesystem("", LocalFileSystem())
register_filesystem("http", HTTPFileSystem())
register_filesystem("https", HTTPFileSystem())
if DF_ENABLED:
    register_filesystem("df", DFFileSystem())
if S3_ENABLED:
    register_filesystem("s3", S3FileSystem())
if OSS_ENABLED:
    register_filesystem("oss", OSSFileSystem())


class GFileObject(object):
    def __init__(self, filename, mode, buff_chunk_size=_DEFAULT_BLOCK_SIZE):
        if mode not in ("r", "rb", "br", "w", "wb", "bw"):
            raise NotImplementedError(
                "mode {} not supported by compat GFile".format(mode)
            )
        self.filename = compat.as_bytes(filename)
        self.fs = get_filesystem(self.filename)
        self.fs_supports_append = hasattr(self.fs, "append")
        self.stat_data = None
        self.buff = None
        # The buffer offset and the buffer chunk size are measured in the
        # natural units of the underlying stream, i.e. bytes for binary mode,
        # or characters in text mode.
        self.buff_chunk_size = buff_chunk_size
        self.buff_offset = 0
        self.current_position = 0
        self.continuation_token = dict(opaque_offset=0, byte_offset=0)
        self.write_temp = None
        self.write_started = False
        self.binary_mode = "b" in mode
        self.write_mode = "w" in mode
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
        self.buff = None
        self.buff_offset = 0
        self.continuation_token = None

    def __iter__(self):
        return self

    def __len__(self):
        return self.size()

    def stat(self):
        if self.stat_data is None:
            self.stat_data = self.fs.stat(self.filename)
        return self.stat_data

    def size(self):
        return self.stat().length

    def read(self, n=None):
        """Reads contents of file to a string.

        Args:
            n: int, number of bytes or characters to read, otherwise
                read all the contents of the file

        Returns:
            Subset of the contents of the file as a string or bytes.
        """
        if self.write_mode:
            raise errors.PermissionDeniedError(
                None, None, "File not opened in read mode"
            )

        result = None
        # there are 4 cases that buffer contains data to read:
        #   1. buff_start <= data_start <= data_end <= buff_end
        #   2. buff_start <= data_start <= buff_end <= data_end
        #   3. data_start <= buff_start <= data_end <= buff_end
        #   4. data_start <= buff_start <= buff_end <= data_end
        # here we only process first 2 cases, since the other 2
        # are to complicated to deal with.
        read_offset = None
        offset = self.current_position - self.buff_offset
        if self.buff and 0 <= offset < len(self.buff):
            # read from local buffer
            if n is None or offset + n > len(self.buff):
                # Case 2
                chunk = self.buff[offset:]
                if n is not None:
                    n -= len(chunk)
                if n is None or n > 0:
                    read_offset = self.buff_offset + len(self.buff)
            else:
                # Case 1
                chunk = self.buff[offset:offset + n]
                n = 0
            result = chunk
        else:
            read_offset = self.current_position

        if read_offset is not None:
            # read from filesystem
            read_size = max(self.buff_chunk_size, n) if n is not None else None
            self.continuation_token = dict(opaque_offset=read_offset, byte_offset=read_offset)
            (self.buff, self.continuation_token) = self.fs.read(
                self.filename, self.binary_mode, read_size, self.continuation_token
            )
            self.buff_offset = self.continuation_token.get("byte_offset", 0) - len(self.buff)

        # add from filesystem
        offset = self.current_position - self.buff_offset + (len(result) if result else 0)
        if n is not None:
            chunk = self.buff[offset:min(offset + n, len(self.buff))]
        else:
            # add all local buffer and update offsets
            chunk = self.buff[offset:]

        result = result + chunk if result else chunk
        self.current_position += len(result)
        return result

    def seek(self, pos, whence=os.SEEK_SET):
        """Set the chunk’s current position. pos is relative to the position indicated by whence.

        Args:
            pos: int, number of bytes or characters of the position.
            whence: int, os.SEEK_SET(0) / os.SEEK_CUR(1) / os.SEEK_END(2), default: os.SEEK_SET
        """
        if self.write_mode:
            raise errors.PermissionDeniedError(
                None, None, "File not opened in read mode"
            )

        if whence == os.SEEK_SET:
            self.current_position = pos
        elif whence == os.SEEK_CUR:
            self.current_position += pos
        elif whence == os.SEEK_END:
            self.current_position = self.size() + pos
        else:
            raise errors.InvalidArgumentError(
                None, None, "Unknown whence value: {}".format(whence)
            )

        return self.current_position

    def tell(self):
        """Return the current position into the chunk.

        Returns:
            Current position into the chunk.
        """
        return self.current_position

    def write(self, file_content):
        """Writes string file contents to file, clearing contents of the file
        on first write and then appending on subsequent calls.

        Args:
            file_content: string, the contents

        Returns:
            The number of bytes actually written
        """
        if not self.write_mode:
            raise errors.PermissionDeniedError(
                None, None, "File not opened in write mode"
            )
        if self.closed:
            raise errors.FailedPreconditionError(
                None, None, "File already closed"
            )

        if self.fs_supports_append:
            if not self.write_started:
                # write the first chunk to truncate file if it already exists
                nbytes = self.fs.write(self.filename, file_content, self.binary_mode)
                self.write_started = True
            else:
                # append the later chunks
                nbytes = self.fs.append(self.filename, file_content, self.binary_mode)
        else:
            # add to temp file, but wait for flush to write to final filesystem
            if self.write_temp is None:
                mode = "w+b" if self.binary_mode else "w+"
                self.write_temp = tempfile.TemporaryFile(mode)

            compatify = compat.as_bytes if self.binary_mode else compat.as_text
            nbytes = self.write_temp.write(compatify(file_content))

        return nbytes

    def __next__(self):
        delimiter = "\n"
        if self.binary_mode:
            delimiter = delimiter.encode()

        line = None
        while True:
            """
            Since we support seek, we cannot make sure current position equals to buffer
            position. Still, there are 3 cases:
                1. cur_pos < buff_start
                2. buff_start <= cur_pos < buff_end
                3. cur_pos >= buff_end
            For case 1 & 3, we need to read from file, and for case 2 we can try read from buff
            """
            offset = self.current_position - self.buff_offset
            if not self.buff or not 0 <= offset < len(self.buff):  # Case 1 & 3
                # read one unit into the buffer
                line = self.read(1)
                if line and (line[-1] == delimiter or not self.buff):
                    return line
                if not self.buff:
                    raise StopIteration()
            else:  # Case 2
                index = self.buff.find(delimiter, offset)
                if index != -1:
                    # include line until now plus newline
                    chunk = self.read(index - offset + 1)
                    line = line + chunk if line else chunk
                    return line

                # read one unit past end of buffer
                n = len(self.buff) - offset + 1
                chunk = self.read(len(self.buff) - offset + 1)
                line = line + chunk if line else chunk
                if line and (line[-1] == delimiter or not self.buff):
                    return line
                if not self.buff:
                    raise StopIteration()

    def next(self):
        return self.__next__()

    def readline(self):
        return self.__next__()

    def readlines(self):
        return [line for line in self]

    def flush(self):
        if self.closed:
            raise errors.FailedPreconditionError(
                None, None, "File already closed"
            )

        if not self.fs_supports_append:
            if self.write_temp is not None:
                # read temp file from the beginning
                self.write_temp.flush()
                self.write_temp.seek(0)
                chunk = self.write_temp.read()
                if chunk is not None:
                    # write full contents and keep in temp file
                    self.fs.write(self.filename, chunk, self.binary_mode)
                    self.write_temp.seek(len(chunk))

    def close(self):
        self.flush()
        if self.write_temp is not None:
            self.write_temp.close()
            self.write_temp = None
            self.write_started = False
        self.closed = True


class GFile(object):
    def __init__(self, filename, mode, *args, **kwargs):
        if get_path_prefix(filename) == "":
            self.file = io.open(filename, mode, *args, **kwargs)
        else:
            self.file = GFileObject(filename, mode, *args, **kwargs)

    def __getattr__(self, name):
        # Attribute lookups are delegated to the underlying file
        # and cached for non-numeric results
        # (i.e. methods are cached, closed and friends are not)
        file = self.__dict__['file']
        a = getattr(file, name)
        if hasattr(a, '__call__'):
            func = a

            @functools.wraps(func)
            def func_wrapper(*args, **kwargs):
                return func(*args, **kwargs)

            a = func_wrapper
        if not isinstance(a, int):
            setattr(self, name, a)
        return a

    # The underlying __enter__ method returns the wrong object
    # (self.file) so override it to return the wrapper
    def __enter__(self):
        self.file.__enter__()
        return self

    # Need to trap __exit__ as well to ensure the file gets
    # deleted when used in a with statement
    def __exit__(self, exc, value, tb):
        result = self.file.__exit__(exc, value, tb)
        return result

    # iter() doesn't use __getattr__ to find the __iter__ method
    def __iter__(self):
        # Don't return iter(self.file), but yield from it to avoid closing
        # file as long as it's being used as iterator (see issue #23700).  We
        # can't use 'yield from' here because iter(file) returns the file
        # object itself, which has a close method, and thus the file would get
        # closed when the generator is finalized, due to PEP380 semantics.
        for line in self.file:
            yield line


def open(filename, mode="r"):
    """Open a GFile instance.

    Args:
      filename: string, a path
      mode: string, IO mode

    Returns:
        A built-in file or GFileObject object of path

    Raises:
      errors.OpError: Propagates any errors reported by the FileSystem API.
    """
    return GFile(filename, mode)


def exists(filename):
    """Determines whether a path exists or not.

    Args:
      filename: string, a path

    Returns:
      True if the path exists, whether its a file or a directory.
      False if the path does not exist and there are no filesystem errors.

    Raises:
      errors.OpError: Propagates any errors reported by the FileSystem API.
    """
    return get_filesystem(filename).exists(filename)


def glob(filename):
    """Returns a list of files that match the given pattern(s).

    Args:
      filename: string or iterable of strings. The glob pattern(s).

    Returns:
      A list of strings containing filenames that match the given pattern(s).

    Raises:
      errors.OpError: If there are filesystem / directory listing errors.
    """
    return get_filesystem(filename).glob(filename)


def isdir(dirname):
    """Returns whether the path is a directory or not.

    Args:
      dirname: string, path to a potential directory

    Returns:
      True, if the path is a directory; False otherwise
    """
    return get_filesystem(dirname).isdir(dirname)


def listdir(dirname):
    """Returns a list of entries contained within a directory.

    The list is in arbitrary order. It does not contain the special entries "."
    and "..".

    Args:
      dirname: string, path to a directory

    Returns:
      [filename1, filename2, ... filenameN] as strings

    Raises:
      errors.NotFoundError if directory doesn't exist
    """
    return get_filesystem(dirname).listdir(dirname)


def makedirs(path, exist_ok=False):
    """Creates a directory and all parent/intermediate directories.

    It succeeds if path already exists and is writable.

    Args:
      path: string, name of the directory to be created
      exist_ok: bool, whether raise error if target directory exists

    Raises:
      errors.AlreadyExistsError: If leaf directory already exists or
        cannot be created.
    """
    return get_filesystem(path).makedirs(path, exist_ok)


def walk(top, topdown=True, onerror=None):
    """Recursive directory tree generator for directories.

    Args:
      top: string, a Directory name
      topdown: bool, Traverse pre order if True, post order if False.
      onerror: optional handler for errors. Should be a function, it will be
        called with the error as argument. Rethrowing the error aborts the walk.

    Errors that happen while listing directories are ignored.

    Yields:
      Each yield is a 3-tuple:  the pathname of a directory, followed by lists
      of all its subdirectories and leaf files.
      (dirname, [subdirname, subdirname, ...], [filename, filename, ...])
      as strings
    """
    top = compat.as_str_any(top)
    fs = get_filesystem(top)
    try:
        listing = listdir(top)
    except errors.NotFoundError as err:
        if onerror:
            onerror(err)
        else:
            return

    files = []
    subdirs = []
    for item in listing:
        full_path = fs.join(top, compat.as_str_any(item))
        if isdir(full_path):
            subdirs.append(item)
        else:
            files.append(item)

    here = (top, subdirs, files)

    if topdown:
        yield here

    for subdir in subdirs:
        joined_subdir = fs.join(top, compat.as_str_any(subdir))
        for subitem in walk(joined_subdir, topdown, onerror=onerror):
            yield subitem

    if not topdown:
        yield here


def stat(filename):
    """Returns file statistics for a given path.

    Args:
      filename: string, path to a file

    Returns:
      FileStatistics struct that contains information about the path

    Raises:
      errors.OpError: If the operation fails.
    """
    return get_filesystem(filename).stat(filename)


def copy(src, dst):
    """Copy a file. Also support copy cross different filesystems.

    Args:
      src: string, path to source file
      dst: string, path to destination file

    Raises:
      errors.OpError: If the operation fails.
    """
    src_fs = get_filesystem(src)
    dst_fs = get_filesystem(dst)

    if not src_fs.exists(src):
        raise errors.OpError("Soruce file not exists!")

    if dst_fs.isdir(dst):
        filename = os.path.basename(src)
        dst = dst_fs.join(dst, filename)

    if src_fs == dst_fs and hasattr(src_fs, "copy"):
        src_fs.copy(src, dst)
    else:
        content, _ = src_fs.read(src, binary_mode=True)
        dst_fs.write(dst, content, binary_mode=True)

    return dst


def remove(filename):
    """Remove a file.

    Args:
      filename: string, path to a file

    Raises:
      errors.OpError: If the operation fails.
    """
    get_filesystem(filename).remove(filename)


def rmtree(dirname):
    """Delete an entire directory tree.

    Args:
      dirname: string, path to a potential directory
    """
    get_filesystem(dirname).rmtree(dirname)


# Used for tests only
def _write_string_to_file(filename, file_content):
    """Writes a string to a given file.

    Args:
      filename: string, path to a file
      file_content: string, contents that need to be written to the file

    Raises:
      errors.OpError: If there are errors during the operation.
    """
    with GFile(filename, mode="w") as f:
        f.write(compat.as_text(file_content))


# Used for tests only
def _read_file_to_string(filename, binary_mode=False):
    """Reads the entire contents of a file to a string.

    Args:
      filename: string, path to a file
      binary_mode: whether to open the file in binary mode or not. This changes
        the type of the object returned.

    Returns:
      contents of the file as a string or bytes.

    Raises:
      errors.OpError: Raises variety of errors that are subtypes e.g.
      `NotFoundError` etc.
    """
    if binary_mode:
        f = GFile(filename, mode="rb")
    else:
        f = GFile(filename, mode="r")
    return f.read()


GFile.open = staticmethod(open)
GFile.exists = staticmethod(exists)
GFile.glob = staticmethod(glob)
GFile.isdir = staticmethod(isdir)
GFile.listdir = staticmethod(listdir)
GFile.makedirs = staticmethod(makedirs)
GFile.walk = staticmethod(walk)
GFile.stat = staticmethod(stat)
GFile.copy = staticmethod(copy)
GFile.remove = staticmethod(remove)
GFile.rmtree = staticmethod(rmtree)
