"""
Utilities for working with the local dataset cache.
"""

import glob
import os
import logging
import tempfile
import json
from collections import defaultdict
from dataclasses import dataclass, asdict
from datetime import timedelta
from fnmatch import fnmatch
from os import PathLike
from urllib.parse import urlparse
from pathlib import Path
from typing import (
    Optional, Tuple, Union,
    IO, Callable, Set,
    List, Iterator, Iterable,
    Dict, NamedTuple,
)
from hashlib import sha256
from functools import wraps
from zipfile import ZipFile, is_zipfile
import tarfile
import shutil
import time

import boto3
import botocore
from botocore.exceptions import ClientError, EndpointConnectionError
import requests
from requests.adapters import HTTPAdapter
from requests.exceptions import ConnectionError


from tqdm import tqdm

logger = logging.getLogger(__name__)

# 不定位到其他地方，在该仓库内部进行处理
CACHE_ROOT = Path(Path('..').resolve(), "cache")
CACHE_DIRECTORY = str(CACHE_ROOT / "cache")
DEPRECATED_CACHE_DIRECTORY = str(CACHE_ROOT / "datasets")

# This variable was deprecated in 0.7.2 since we use a single folder for caching
# all types of files (datasets, models, etc.)
DATASET_CACHE = CACHE_DIRECTORY

# Warn if the user is still using the deprecated cache directory.
if os.path.exists(DEPRECATED_CACHE_DIRECTORY):
    logger.warning(
        f"Deprecated cache directory found ({DEPRECATED_CACHE_DIRECTORY}).  "
        f"Please remove this directory from your system to free up space."
    )


def _resource_to_filename(resource: str, etag: str = None) -> str:
    """
    Convert a `resource` into a hashed filename in a repeatable way.
    If `etag` is specified, append its hash to the resources's, delimited
    by a period.
    """
    resource_bytes = resource.encode("utf-8")
    resource_hash = sha256(resource_bytes)
    filename = resource_hash.hexdigest()

    if etag:
        etag_bytes = etag.encode("utf-8")
        etag_hash = sha256(etag_bytes)
        filename += "." + etag_hash.hexdigest()

    return filename


def filename_to_url(filename: str, cache_dir: Union[str, Path] = None) -> Tuple[str, str]:
    """
    Return the url and etag (which may be `None`) stored for `filename`.
    Raise `FileNotFoundError` if `filename` or its stored metadata do not exist.
    """
    if cache_dir is None:
        cache_dir = CACHE_DIRECTORY

    cache_path = os.path.join(cache_dir, filename)
    if not os.path.exists(cache_path):
        raise FileNotFoundError("file {} not found".format(cache_path))

    meta_path = cache_path + ".json"
    if not os.path.exists(meta_path):
        raise FileNotFoundError("file {} not found".format(meta_path))

    with open(meta_path) as meta_file:
        metadata = json.load(meta_file)
    url = metadata["url"]
    etag = metadata["etag"]

    return url, etag



def is_url_or_existing_file(url_or_filename: Union[str, Path, None]) -> bool:
    """
    Given something that might be a URL (or might be a local path),
    determine check if it's url or an existing file path.
    """
    if url_or_filename is None:
        return False
    url_or_filename = os.path.expanduser(str(url_or_filename))
    parsed = urlparse(url_or_filename)
    return parsed.scheme in ("http", "https", "s3") or os.path.exists(url_or_filename)


def _split_s3_path(url: str) -> Tuple[str, str]:
    """Split a full s3 path into the bucket name and path."""
    parsed = urlparse(url)
    if not parsed.netloc or not parsed.path:
        raise ValueError("bad s3 path {}".format(url))
    bucket_name = parsed.netloc
    s3_path = parsed.path
    # Remove '/' at beginning of path.
    if s3_path.startswith("/"):
        s3_path = s3_path[1:]
    return bucket_name, s3_path


def _s3_request(func: Callable):
    """
    Wrapper function for s3 requests in order to create more helpful error
    messages.
    """

    @wraps(func)
    def wrapper(url: str, *args, **kwargs):
        try:
            return func(url, *args, **kwargs)
        except ClientError as exc:
            if int(exc.response["Error"]["Code"]) == 404:
                raise FileNotFoundError("file {} not found".format(url))
            else:
                raise

    return wrapper


def _get_s3_resource():
    session = boto3.session.Session()
    if session.get_credentials() is None:
        # Use unsigned requests.
        s3_resource = session.resource(
            "s3", config=botocore.client.Config(signature_version=botocore.UNSIGNED)
        )
    else:
        s3_resource = session.resource("s3")
    return s3_resource


@_s3_request
def _s3_etag(url: str) -> Optional[str]:
    """Check ETag on S3 object."""
    s3_resource = _get_s3_resource()
    bucket_name, s3_path = _split_s3_path(url)
    s3_object = s3_resource.Object(bucket_name, s3_path)
    return s3_object.e_tag


@_s3_request
def _s3_get(url: str, temp_file: IO) -> None:
    """Pull a file directly from S3."""
    s3_resource = _get_s3_resource()
    bucket_name, s3_path = _split_s3_path(url)
    s3_resource.Bucket(bucket_name).download_fileobj(s3_path, temp_file)

def _find_latest_cached(url: str, cache_dir: Union[str, Path]) -> Optional[str]:
    filename = _resource_to_filename(url)
    cache_path = os.path.join(cache_dir, filename)
    candidates: List[Tuple[str, float]] = []
    for path in glob.glob(cache_path + "*"):
        if path.endswith(".json") or path.endswith("-extracted") or path.endswith(".lock"):
            continue
        mtime = os.path.getmtime(path)
        candidates.append((path, mtime))
    # Sort candidates by modification time, newest first.
    candidates.sort(key=lambda x: x[1], reverse=True)
    if candidates:
        return candidates[0][0]
    return None


class CacheFile:
    """
    This is a context manager that makes robust caching easier.

    On `__enter__`, an IO handle to a temporarily file is returned, which can
    be treated as if it's the actual cache file.

    On `__exit__`, the temporarily file is renamed to the cache file. If anything
    goes wrong while writing to the temporary file, it will be removed.
    """

    def __init__(
        self, cache_filename: Union[Path, str], mode: str = "w+b", suffix: str = ".tmp"
    ) -> None:
        self.cache_filename = (
            cache_filename if isinstance(cache_filename, Path) else Path(cache_filename)
        )
        self.cache_directory = os.path.dirname(self.cache_filename)
        self.mode = mode
        self.temp_file = tempfile.NamedTemporaryFile(
            self.mode, dir=self.cache_directory, delete=False, suffix=suffix
        )

    def __enter__(self):
        return self.temp_file

    def __exit__(self, exc_type, exc_value, traceback):
        self.temp_file.close()
        if exc_value is None:
            # Success.
            logger.debug(
                "Renaming temp file %s to cache at %s", self.temp_file.name, self.cache_filename
            )
            # Rename the temp file to the actual cache filename.
            os.replace(self.temp_file.name, self.cache_filename)
            return True
        # Something went wrong, remove the temp file.
        logger.debug("removing temp file %s", self.temp_file.name)
        os.remove(self.temp_file.name)
        return False


@dataclass
class _Meta:
    """
    Any resource that is downloaded to - or extracted in - the cache directory will
    have a meta JSON file written next to it, which corresponds to an instance
    of this class.

    In older versions of AllenNLP, this meta document just had two fields: 'url' and
    'etag'. The 'url' field is now the more general 'resource' field, but these old
    meta files are still compatible when a `_Meta` is instantiated with the `.from_path()`
    class method.
    """

    resource: str
    """
    URL or normalized path to the resource.
    """

    cached_path: str
    """
    Path to the corresponding cached version of the resource.
    """

    creation_time: float
    """
    The unix timestamp of when the corresponding resource was cached or extracted.
    """

    size: int = 0
    """
    The size of the corresponding resource, in bytes.
    """

    etag: Optional[str] = None
    """
    Optional ETag associated with the current cached version of the resource.
    """

    extraction_dir: bool = False
    """
    Does this meta corresponded to an extraction directory?
    """

    def to_file(self) -> None:
        with open(self.cached_path + ".json", "w") as meta_file:
            json.dump(asdict(self), meta_file)

    @classmethod
    def from_path(cls, path: Union[str, Path]) -> "_Meta":
        path = str(path)
        with open(path) as meta_file:
            data = json.load(meta_file)
            # For backwards compat:
            if "resource" not in data:
                data["resource"] = data.pop("url")
            if "creation_time" not in data:
                data["creation_time"] = os.path.getmtime(path[:-5])
            if "extraction_dir" not in data and path.endswith("-extracted.json"):
                data["extraction_dir"] = True
            if "cached_path" not in data:
                data["cached_path"] = path[:-5]
            if "size" not in data:
                data["size"] = _get_resource_size(data["cached_path"])
        return cls(**data)


def read_set_from_file(filename: str) -> Set[str]:
    """
    Extract a de-duped collection (set) of text from a file.
    Expected file format is one item per line.
    """
    collection = set()
    with open(filename, "r") as file_:
        for line in file_:
            collection.add(line.rstrip())
    return collection


def get_file_extension(path: str, dot=True, lower: bool = True):
    ext = os.path.splitext(path)[1]
    ext = ext if dot else ext[1:]
    return ext.lower() if lower else ext


def open_compressed(
    filename: Union[str, Path], mode: str = "rt", encoding: Optional[str] = "UTF-8", **kwargs
):
    if isinstance(filename, Path):
        filename = str(filename)
    open_fn: Callable = open

    if filename.endswith(".gz"):
        import gzip

        open_fn = gzip.open
    elif filename.endswith(".bz2"):
        import bz2

        open_fn = bz2.open
    return open_fn(filename, mode=mode, encoding=encoding, **kwargs)


def text_lines_from_file(filename: Union[str, Path], strip_lines: bool = True) -> Iterator[str]:
    with open_compressed(filename, "rt", encoding="UTF-8", errors="replace") as p:
        if strip_lines:
            for line in p:
                yield line.strip()
        else:
            yield from p


def json_lines_from_file(filename: Union[str, Path]) -> Iterable[Union[list, dict]]:
    return (json.loads(line) for line in text_lines_from_file(filename))


def _get_resource_size(path: str) -> int:
    """
    Get the size of a file or directory.
    """
    if os.path.isfile(path):
        return os.path.getsize(path)
    inodes: Set[int] = set()
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            # skip if it is symbolic link or the same as a file we've already accounted
            # for (this could happen with hard links).
            inode = os.stat(fp).st_ino
            if not os.path.islink(fp) and inode not in inodes:
                inodes.add(inode)
                total_size += os.path.getsize(fp)
    return total_size


class _CacheEntry(NamedTuple):
    regular_files: List[_Meta]
    extraction_dirs: List[_Meta]


def _find_entries(
    patterns: List[str] = None,
    cache_dir: Union[str, Path] = None,
) -> Tuple[int, Dict[str, _CacheEntry]]:
    """
    Find all cache entries, filtering ones that don't match any of the glob patterns given.

    Returns the total size of the matching entries and mapping or resource name to meta data.

    The values in the returned mapping are tuples because we seperate meta entries that
    correspond to extraction directories vs regular cache entries.
    """
    cache_dir = os.path.expanduser(cache_dir or CACHE_DIRECTORY)

    total_size: int = 0
    cache_entries: Dict[str, _CacheEntry] = defaultdict(lambda: _CacheEntry([], []))
    for meta_path in glob.glob(str(cache_dir) + "/*.json"):
        meta = _Meta.from_path(meta_path)
        if patterns and not any(fnmatch(meta.resource, p) for p in patterns):
            continue
        if meta.extraction_dir:
            cache_entries[meta.resource].extraction_dirs.append(meta)
        else:
            cache_entries[meta.resource].regular_files.append(meta)
        total_size += meta.size

    # Sort entries for each resource by creation time, newest first.
    for entry in cache_entries.values():
        entry.regular_files.sort(key=lambda meta: meta.creation_time, reverse=True)
        entry.extraction_dirs.sort(key=lambda meta: meta.creation_time, reverse=True)

    return total_size, cache_entries


def remove_cache_entries(patterns: List[str], cache_dir: Union[str, Path] = None) -> int:
    """
    Remove cache entries matching the given patterns.

    Returns the total reclaimed space in bytes.
    """
    total_size, cache_entries = _find_entries(patterns=patterns, cache_dir=cache_dir)
    for resource, entry in cache_entries.items():
        for meta in entry.regular_files:
            logger.info("Removing cached version of %s at %s", resource, meta.cached_path)
            os.remove(meta.cached_path)
            if os.path.exists(meta.cached_path + ".lock"):
                os.remove(meta.cached_path + ".lock")
            os.remove(meta.cached_path + ".json")
        for meta in entry.extraction_dirs:
            logger.info("Removing extracted version of %s at %s", resource, meta.cached_path)
            shutil.rmtree(meta.cached_path)
            if os.path.exists(meta.cached_path + ".lock"):
                os.remove(meta.cached_path + ".lock")
            os.remove(meta.cached_path + ".json")
    return total_size


def inspect_cache(patterns: List[str] = None, cache_dir: Union[str, Path] = None):
    """
    Print out useful information about the cache directory.
    """
    from .util import format_timedelta, format_size

    cache_dir = os.path.expanduser(cache_dir or CACHE_DIRECTORY)

    # Gather cache entries by resource.
    total_size, cache_entries = _find_entries(patterns=patterns, cache_dir=cache_dir)

    if patterns:
        print(f"Cached resources matching {patterns}:")
    else:
        print("Cached resources:")

    for resource, entry in sorted(
        cache_entries.items(),
        # Sort by creation time, latest first.
        key=lambda x: max(
            0 if not x[1][0] else x[1][0][0].creation_time,
            0 if not x[1][1] else x[1][1][0].creation_time,
        ),
        reverse=True,
    ):
        print("\n-", resource)
        if entry.regular_files:
            td = timedelta(seconds=time.time() - entry.regular_files[0].creation_time)
            n_versions = len(entry.regular_files)
            size = entry.regular_files[0].size
            print(
                f"  {n_versions} {'versions' if n_versions > 1 else 'version'} cached, "
                f"latest {format_size(size)} from {format_timedelta(td)} ago"
            )
        if entry.extraction_dirs:
            td = timedelta(seconds=time.time() - entry.extraction_dirs[0].creation_time)
            n_versions = len(entry.extraction_dirs)
            size = entry.extraction_dirs[0].size
            print(
                f"  {n_versions} {'versions' if n_versions > 1 else 'version'} extracted, "
                f"latest {format_size(size)} from {format_timedelta(td)} ago"
            )
    print(f"\nTotal size: {format_size(total_size)}")
