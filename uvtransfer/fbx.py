"""
Minimal FBX reader: parses ASCII (6.x / 7.x) and binary FBX files into a tree
of `Node`s. It reads the file structure only; `uvtransfer.mesh` turns the
geometry nodes into meshes.

Every FBX record is a name, a list of properties (numbers, strings or arrays)
and child records, e.g. in ASCII:

    Geometry: 123, "Geometry::Body", "Mesh" {
        Vertices: *9 { a: 0,0,0, 1,0,0, 0,1,0 }
    }
"""

import re
import struct
import zlib

import numpy as np

BINARY_MAGIC = b"Kaydara FBX Binary  "
_BINARY_HEADER_SIZE = 27          # magic + 0x1A 0x00 + uint32 version
_VERSION_OFFSET = 23
_WIDE_RECORDS_FROM = 7500         # FBX 7.5+ uses 64-bit record headers

# Binary property type codes
_SCALAR_FORMATS = {b"Y": "<h", b"C": "<?", b"I": "<i", b"F": "<f", b"D": "<d", b"L": "<q"}
_ARRAY_DTYPES = {b"f": "<f4", b"d": "<f8", b"l": "<i8", b"i": "<i4", b"b": "u1"}
_ARRAY_ZLIB = 1


class Node:
    """One FBX record: `name`, `props` (list of values) and `children` (list of Nodes)."""

    __slots__ = ("name", "props", "children")

    def __init__(self, name, props, children):
        self.name, self.props, self.children = name, props, children

    def find(self, name):
        """First child called `name`, or None."""
        for c in self.children:
            if c.name == name:
                return c
        return None

    def find_all(self, name):
        """All children called `name`, in file order."""
        return [c for c in self.children if c.name == name]

    def array(self, name):
        """The array stored in child `name` as a 1-D numpy array, or None if missing.

        FBX 7 stores arrays as a single array property; FBX 6 ASCII writes them
        as a plain list of values, which is also handled here.
        """
        c = self.find(name)
        if c is None or not c.props:
            return None
        if len(c.props) == 1:
            return np.atleast_1d(np.asarray(c.props[0]))
        return np.asarray(c.props, dtype=np.float64)

    def string(self, name, default=""):
        """First property of child `name` as a str, or `default` if missing."""
        c = self.find(name)
        if c is None or not c.props:
            return default
        return as_str(c.props[0])


def as_str(value):
    """FBX string property (bytes in binary files, str in ASCII ones) as str."""
    return value.decode("utf-8", "replace") if isinstance(value, bytes) else str(value)


def read_fbx(path):
    """Parse an FBX file (ASCII or binary) into a root `Node`.

    Malformed or hostile files fail with an exception rather than hanging or
    exhausting memory: record and array sizes are checked against the file,
    and nesting depth and array sizes are capped.

    Raises:
        OSError: the file can't be read.
        ValueError, struct.error, zlib.error: the file is malformed.
    """
    with open(path, "rb") as f:
        data = f.read()
    if data.startswith(BINARY_MAGIC):
        return _read_binary(data)
    return _read_ascii(data.decode("utf-8", "replace"))


# --------------------------------------------------------------------------
# Binary
# --------------------------------------------------------------------------

MAX_DEPTH = 64                    # real files nest records about 6 deep
MAX_ARRAY_BYTES = 1 << 30         # one array; a 40-million-vertex mesh's positions fit
MAX_TOTAL_ARRAY_BYTES = 4 << 30   # all arrays in one file


def _read_binary(data):
    """Parse binary FBX bytes into a root `Node`."""
    return _BinaryReader(data).read()


class _BinaryReader:
    """Reads binary FBX records, refusing anything that doesn't fit the file.

    Every record must end after its own name and inside its parent, so
    reading always moves forward; compressed arrays are only inflated up to
    their declared size.
    """

    def __init__(self, data):
        self.data = data
        version = struct.unpack_from("<I", data, _VERSION_OFFSET)[0]
        # record header: end offset, property count, property bytes
        self.header_fmt = "<QQQ" if version >= _WIDE_RECORDS_FROM else "<III"
        self.header_size = struct.calcsize(self.header_fmt)
        self.array_bytes = 0              # running total, checked against MAX_TOTAL_ARRAY_BYTES

    def read(self):
        top = []
        off = _BINARY_HEADER_SIZE
        while off < len(self.data) - self.header_size:
            node, off = self._node(off, len(self.data), depth=1)
            if node is None:                      # null record ends the top level
                break
            top.append(node)
        return Node("root", [], top)

    def _node(self, off, limit, depth):
        """Read the record at `off`, which must end by `limit`.

        Returns (Node, or None for a null record; offset after the record).
        """
        if depth > MAX_DEPTH:
            raise ValueError(f"records are nested more than {MAX_DEPTH} deep")
        end, n_props, _props_len = struct.unpack_from(self.header_fmt, self.data, off)
        name_at = off + self.header_size
        if end == 0:
            return None, name_at + 1
        name_len = self.data[name_at]
        p = name_at + 1 + name_len
        if not p <= end <= limit:
            raise ValueError(f"record at byte {off} has an invalid length")
        name = self.data[name_at + 1:p].decode("ascii", "replace")
        props, p = self._props(p, n_props, end)
        children = []
        while p < end:
            child, p = self._node(p, end, depth + 1)
            if child is None:
                break
            children.append(child)
        return Node(name, props, children), end

    def _props(self, off, count, limit):
        """Read `count` properties starting at `off`, all inside `limit`."""
        props = []
        for _ in range(count):
            code = self.data[off:off + 1]
            off += 1
            if code in _SCALAR_FORMATS:
                fmt = _SCALAR_FORMATS[code]
                self._check_inside(off + struct.calcsize(fmt), limit)
                props.append(struct.unpack_from(fmt, self.data, off)[0])
                off += struct.calcsize(fmt)
            elif code in _ARRAY_DTYPES:
                value, off = self._array(code, off, limit)
                props.append(value)
            elif code in (b"S", b"R"):            # string / raw bytes
                length = struct.unpack_from("<I", self.data, off)[0]
                off += 4
                self._check_inside(off + length, limit)
                props.append(bytes(self.data[off:off + length]))
                off += length
            else:
                raise ValueError(f"unknown FBX property type {code!r} at byte {off - 1}")
        return props, off

    def _array(self, code, off, limit):
        """Read an array property whose header starts at `off`. Returns (array, next offset)."""
        length, encoding, byte_len = struct.unpack_from("<III", self.data, off)
        off += 12
        self._check_inside(off + byte_len, limit)
        dtype = np.dtype(_ARRAY_DTYPES[code])
        size = length * dtype.itemsize
        if size > MAX_ARRAY_BYTES or self.array_bytes + size > MAX_TOTAL_ARRAY_BYTES:
            raise ValueError(f"an array of {size / 2**20:,.0f} MB is larger than this reader allows")
        self.array_bytes += size
        raw = self.data[off:off + byte_len]
        if encoding == _ARRAY_ZLIB:
            raw = _inflate(raw, size)
        elif encoding != 0:
            raise ValueError(f"unknown array encoding {encoding} at byte {off - 8}")
        if len(raw) < size:
            raise ValueError(f"array at byte {off - 12} is shorter than declared")
        return np.frombuffer(raw, dtype=dtype, count=length).copy(), off + byte_len

    def _check_inside(self, end, limit):
        if end > limit:
            raise ValueError(f"data runs past the end of its record (byte {limit})")


def _inflate(raw, size):
    """Decompress a zlib array that must expand to exactly `size` bytes (never more)."""
    stream = zlib.decompressobj()
    out = stream.decompress(raw, size)
    if len(out) != size or stream.decompress(stream.unconsumed_tail, 1):
        raise ValueError("compressed array doesn't match its declared size")
    return out


# --------------------------------------------------------------------------
# ASCII
# --------------------------------------------------------------------------

_ASCII_TOKEN = re.compile(r"""
    \s*(?:
        (?P<comment> ;[^\n]* )
      | (?P<string>  "[^"]*" )                  # FBX writes quotes inside strings as &quot;
      | (?P<open>    \{ )
      | (?P<close>   \} )
      | (?P<count>   \*\d+ )                    # array length marker, e.g. *1024
      | (?P<key>     [A-Za-z_][\w|]*\s*: )      # record name, e.g. Vertices:
      | (?P<value>   [^\s,{}"]+ )
      | (?P<comma>   , )
    )""", re.S | re.X)


def _read_ascii(text):
    """Build the node tree from ASCII FBX text with a single tokenising pass."""
    root = Node("root", [], [])
    stack = [root]            # open records; the innermost is stack[-1]
    current = None            # record whose properties are being read (before its '{')
    pos, n = 0, len(text)
    while pos < n:
        m = _ASCII_TOKEN.match(text, pos)
        if not m:
            pos += 1
            continue
        pos = m.end()
        kind, tok = m.lastgroup, m.group(m.lastgroup)
        if kind in ("comment", "comma"):
            continue
        if kind == "key":
            name = tok[:-1].strip()
            owner = stack[-1]
            if name == "a" and owner.props and isinstance(owner.props[-1], str) \
                    and owner.props[-1].startswith("*"):
                # FBX 7 array body "a: v,v,v,..." - parse it in one go (fast path)
                end = text.index("}", pos)
                values = text[pos:end].replace("\n", "").split(",")
                owner.props[-1] = np.array([float(v) for v in values if v.strip()])
                pos = end
                continue
            current = Node(name, [], [])
            owner.children.append(current)
        elif kind == "open":
            if current is None:                   # anonymous block
                current = Node("", [], [])
                stack[-1].children.append(current)
            stack.append(current)
            current = None
        elif kind == "close":
            if len(stack) > 1:
                stack.pop()
            current = None
        elif current is not None:
            current.props.append(_ascii_value(kind, tok))
    return root


def _ascii_value(kind, tok):
    if kind == "string":
        return tok[1:-1]
    if kind == "count":
        return tok                                # replaced by the array once "a:" is read
    try:
        return float(tok) if any(ch in tok for ch in ".eE") else int(tok)
    except ValueError:
        return tok                                # bare words such as T, Y, N
