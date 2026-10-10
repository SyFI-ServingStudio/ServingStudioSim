"""Read flat columns of neuron-explorer Parquet exports without pyarrow.

The pinned vLLM Neuron image ships neither pyarrow nor fastparquet, so the
device-trace runner decodes the small subset that neuron-explorer's
parquet-go writer emits: one flat schema, SNAPPY or uncompressed pages (v1 or
v2), PLAIN / dictionary / DELTA_BINARY_PACKED values and RLE definition
levels. Anything else raises ``ParquetFormatError``; tests compare every
column this runner uses against pyarrow on real exports.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

_TYPE_BOOLEAN, _TYPE_INT32, _TYPE_INT64 = 0, 1, 2
_TYPE_FLOAT, _TYPE_DOUBLE, _TYPE_BYTE_ARRAY = 4, 5, 6
_PLAIN, _PLAIN_DICTIONARY, _RLE, _DELTA_BINARY_PACKED, _RLE_DICTIONARY = 0, 2, 3, 5, 8
_DATA_PAGE, _DICTIONARY_PAGE, _DATA_PAGE_V2 = 0, 2, 3
_UNCOMPRESSED, _SNAPPY = 0, 1
_OPTIONAL = 1


class ParquetFormatError(ValueError):
    """The file uses a Parquet feature this reader does not implement."""


# --- Thrift compact protocol (footer and page headers) ------------------------


class _Reader:
    def __init__(self, data: bytes | memoryview, offset: int = 0):
        self.data = memoryview(data)
        self.offset = offset

    def byte(self) -> int:
        value = self.data[self.offset]
        self.offset += 1
        return value

    def take(self, count: int) -> memoryview:
        if self.offset + count > len(self.data):
            raise ParquetFormatError("truncated Parquet data")
        view = self.data[self.offset : self.offset + count]
        self.offset += count
        return view

    def uvarint(self) -> int:
        shift = result = 0
        while True:
            byte = self.byte()
            result |= (byte & 0x7F) << shift
            if byte < 0x80:
                return result
            shift += 7

    def zigzag(self) -> int:
        value = self.uvarint()
        return (value >> 1) ^ -(value & 1)


def _thrift_value(reader: _Reader, kind: int):
    if kind in (1, 2):  # Boolean field: the value is in the type nibble.
        return kind == 1
    if kind == 3:
        return struct.unpack("b", reader.take(1))[0]
    if kind in (4, 5, 6):
        return reader.zigzag()
    if kind == 7:
        return struct.unpack("<d", reader.take(8))[0]
    if kind == 8:
        return bytes(reader.take(reader.uvarint()))
    if kind in (9, 10):
        header = reader.byte()
        size, element = header >> 4, header & 0x0F
        if size == 15:
            size = reader.uvarint()
        if element in (1, 2):  # List booleans are one byte each.
            return [reader.byte() == 1 for _ in range(size)]
        return [_thrift_value(reader, element) for _ in range(size)]
    if kind == 11:
        size = reader.uvarint()
        if not size:
            return {}
        types = reader.byte()
        return {
            _thrift_value(reader, types >> 4): _thrift_value(reader, types & 0x0F)
            for _ in range(size)
        }
    if kind == 12:
        return _thrift_struct(reader)
    raise ParquetFormatError(f"unsupported Thrift compact type {kind}")


def _thrift_struct(reader: _Reader) -> dict[int, object]:
    fields, last = {}, 0
    while True:
        header = reader.byte()
        if header == 0:
            return fields
        delta, kind = header >> 4, header & 0x0F
        last = last + delta if delta else reader.zigzag()
        fields[last] = _thrift_value(reader, kind)


# --- Value codecs ---------------------------------------------------------------


def snappy_decompress(data: bytes | memoryview) -> bytes:
    """Raw (unframed) Snappy block, as written into Parquet pages."""
    reader = _Reader(data)
    length = reader.uvarint()
    out = bytearray()
    source = reader.data
    while reader.offset < len(source):
        tag = reader.byte()
        kind = tag & 3
        if kind == 0:
            size = tag >> 2
            if size >= 60:
                width = size - 59
                size = int.from_bytes(reader.take(width), "little")
            out += reader.take(size + 1)
            continue
        if kind == 1:
            size = ((tag >> 2) & 7) + 4
            offset = ((tag >> 5) << 8) | reader.byte()
        elif kind == 2:
            size = (tag >> 2) + 1
            offset = int.from_bytes(reader.take(2), "little")
        else:
            size = (tag >> 2) + 1
            offset = int.from_bytes(reader.take(4), "little")
        if offset <= 0 or offset > len(out):
            raise ParquetFormatError("invalid Snappy copy offset")
        start = len(out) - offset
        if offset >= size:
            out += out[start : start + size]
        else:  # Overlapping copy repeats the last ``offset`` bytes.
            pattern = bytes(out[start:])
            out += (pattern * (size // offset + 1))[:size]
    if len(out) != length:
        raise ParquetFormatError("Snappy length mismatch")
    return bytes(out)


def _bit_unpack(data: memoryview, width: int, count: int) -> np.ndarray:
    """Little-endian bit-packed unsigned integers, as Parquet packs them."""
    if width == 0:
        return np.zeros(count, dtype=np.uint64)
    raw = np.frombuffer(data, dtype=np.uint8)
    bits = np.unpackbits(raw, bitorder="little")[: count * width].reshape(count, width)
    weights = np.left_shift(np.uint64(1), np.arange(width, dtype=np.uint64))
    return (bits.astype(np.uint64) * weights).sum(axis=1, dtype=np.uint64)


def _rle_hybrid(reader: _Reader, width: int, count: int) -> np.ndarray:
    values, produced = [], 0
    byte_width = (width + 7) // 8
    while produced < count:
        header = reader.uvarint()
        if header & 1:
            groups = header >> 1
            run = _bit_unpack(reader.take(groups * width), width, groups * 8)
        else:
            repeat = header >> 1
            value = int.from_bytes(reader.take(byte_width), "little")
            run = np.full(repeat, value, dtype=np.uint64)
        values.append(run)
        produced += len(run)
    return np.concatenate(values)[:count].astype(np.int64) if values else np.zeros(0, np.int64)


def _delta_binary_packed(reader: _Reader, count: int) -> np.ndarray:
    block_size = reader.uvarint()
    miniblocks = reader.uvarint()
    total = reader.uvarint()
    first = reader.zigzag()
    if total != count or block_size % 128 or miniblocks == 0 or block_size % miniblocks:
        raise ParquetFormatError("unexpected DELTA_BINARY_PACKED header")
    per_miniblock = block_size // miniblocks
    deltas = []
    remaining = total - 1
    while remaining > 0:
        min_delta = np.int64(reader.zigzag()).astype(np.uint64)
        widths = bytes(reader.take(miniblocks))
        for width in widths:
            if remaining <= 0:
                break
            packed = _bit_unpack(reader.take(per_miniblock * width // 8), width, per_miniblock)
            take = min(remaining, per_miniblock)
            deltas.append(packed[:take] + min_delta)  # uint64 wraps like int64.
            remaining -= take
    values = np.empty(total, dtype=np.int64)
    if total:
        values[0] = first
        if total > 1:
            with np.errstate(over="ignore"):
                values[1:] = np.cumsum(np.concatenate(deltas).view(np.int64)) + np.int64(first)
    return values


def _plain(reader: _Reader, physical: int, count: int):
    if physical == _TYPE_INT64:
        return np.frombuffer(reader.take(8 * count), dtype="<i8").astype(np.int64)
    if physical == _TYPE_INT32:
        return np.frombuffer(reader.take(4 * count), dtype="<i4").astype(np.int64)
    if physical == _TYPE_DOUBLE:
        return np.frombuffer(reader.take(8 * count), dtype="<f8").astype(np.float64)
    if physical == _TYPE_FLOAT:
        return np.frombuffer(reader.take(4 * count), dtype="<f4").astype(np.float64)
    if physical == _TYPE_BOOLEAN:
        return _bit_unpack(reader.take((count + 7) // 8), 1, count).astype(bool)
    if physical == _TYPE_BYTE_ARRAY:
        values = []
        for _ in range(count):
            size = struct.unpack("<i", reader.take(4))[0]
            values.append(bytes(reader.take(size)).decode())
        return values
    raise ParquetFormatError(f"unsupported physical type {physical}")


# --- File and column assembly -----------------------------------------------------


def _decompress(codec: int, data: memoryview) -> memoryview:
    if codec == _UNCOMPRESSED:
        return data
    if codec == _SNAPPY:
        return memoryview(snappy_decompress(data))
    raise ParquetFormatError(f"unsupported compression codec {codec}")


def _decode_values(reader, encoding, physical, count, dictionary):
    """One page's non-null values: a NumPy array, or a list for strings."""
    if encoding == _PLAIN:
        return _plain(reader, physical, count)
    if encoding == _DELTA_BINARY_PACKED and physical in (_TYPE_INT32, _TYPE_INT64):
        return _delta_binary_packed(reader, count)
    if encoding in (_RLE_DICTIONARY, _PLAIN_DICTIONARY):
        if dictionary is None:
            raise ParquetFormatError("dictionary-encoded page without a dictionary")
        width = reader.byte()
        indices = _rle_hybrid(reader, width, count)
        if isinstance(dictionary, list):
            return [dictionary[i] for i in indices]
        return dictionary[indices]
    raise ParquetFormatError(f"unsupported encoding {encoding} for type {physical}")


def _read_chunk(data: memoryview, meta: dict, physical: int, optional: bool):
    """Yield ``(values, defined)`` per data page; ``defined`` is None when required."""
    codec, total = meta[4], meta[5]
    start = meta.get(11) or meta[9]
    reader = _Reader(data, start)
    dictionary, produced = None, 0
    while produced < total:
        header = _thrift_struct(reader)
        page_type, compressed = header[1], header[3]
        body = reader.take(compressed)
        if page_type == _DICTIONARY_PAGE:
            page = _Reader(_decompress(codec, body))
            dictionary = _plain(page, physical, header[7][1])
            continue
        if page_type == _DATA_PAGE:
            info = header[5]
            count, encoding = info[1], info[2]
            page = _Reader(_decompress(codec, body))
            defined = None
            if optional:
                size = struct.unpack("<i", page.take(4))[0]
                defined = _rle_hybrid(_Reader(page.take(size)), 1, count)
        elif page_type == _DATA_PAGE_V2:
            info = header[8]
            count, encoding = info[1], info[4]
            def_bytes, rep_bytes = info[5], info[6]
            if rep_bytes:
                raise ParquetFormatError("repeated columns are unsupported")
            payload = body[def_bytes:]
            if info.get(7, True):
                payload = _decompress(codec, payload)
            page = _Reader(payload)
            defined = _rle_hybrid(_Reader(body[:def_bytes]), 1, count) if optional else None
        else:
            raise ParquetFormatError(f"unsupported page type {page_type}")
        present = count if defined is None else int(defined.sum())
        yield _decode_values(page, encoding, physical, present, dictionary), defined
        produced += count


def _chunks(path: Path, columns: list[str] | None):
    data = memoryview(Path(path).read_bytes())
    if bytes(data[:4]) != b"PAR1" or bytes(data[-4:]) != b"PAR1":
        raise ParquetFormatError(f"{path} is not a Parquet file")
    footer_size = struct.unpack("<i", data[-8:-4])[0]
    metadata = _thrift_struct(_Reader(data[-8 - footer_size : -8]))
    leaves = {}
    for element in metadata[2][1:]:
        if element.get(5):
            raise ParquetFormatError("nested Parquet schemas are unsupported")
        leaves[element[4].decode()] = (element[1], element.get(3, 0) == _OPTIONAL)
    wanted = list(leaves) if columns is None else list(columns)
    missing = [name for name in wanted if name not in leaves]
    if missing:
        raise ParquetFormatError(f"{path} lacks columns {missing}")
    pages = {name: [] for name in wanted}
    for row_group in metadata.get(4, []):
        for chunk in row_group[1]:
            meta = chunk[3]
            name = b".".join(meta[3]).decode()
            if name in pages:
                pages[name].extend(_read_chunk(data, meta, *leaves[name]))
    return pages


def read_columns(path: Path, columns: list[str] | None = None) -> dict[str, list]:
    """Return ``{column: python values}`` (None for nulls) of a flat Parquet table."""
    result = {}
    for name, pages in _chunks(path, columns).items():
        out = []
        for values, defined in pages:
            values = values if isinstance(values, list) else values.tolist()
            if defined is None:
                out.extend(values)
            else:
                iterator = iter(values)
                out.extend(next(iterator) if flag else None for flag in defined)
        result[name] = out
    return result


def read_int_arrays(path: Path, columns: list[str]) -> dict[str, np.ndarray]:
    """Return non-null integer columns as int64 arrays (large instruction tables)."""
    result = {}
    for name, pages in _chunks(path, columns).items():
        parts = []
        for values, defined in pages:
            if isinstance(values, list) or values.dtype != np.int64:
                raise ParquetFormatError(f"column {name} is not an integer column")
            if defined is not None and not defined.all():
                raise ParquetFormatError(f"column {name} has null values")
            parts.append(values)
        result[name] = np.concatenate(parts) if parts else np.zeros(0, dtype=np.int64)
    return result


def read_rows(path: Path, columns: list[str] | None = None) -> list[dict]:
    table = read_columns(path, columns)
    names = list(table)
    return [dict(zip(names, row)) for row in zip(*(table[name] for name in names))]
