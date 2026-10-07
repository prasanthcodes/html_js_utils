'use strict';

/* =====================================================================
 * Parsing: .npy header (Python literal), dtypes, values, .npz (zip)
 * ===================================================================== */

// Minimal parser for the Python dict literal stored in a .npy header,
// e.g. {'descr': '<f8', 'fortran_order': False, 'shape': (3, 4), }
function parsePyLiteral(src) {
  let i = 0;
  const fail = (msg) => { throw new Error(`Cannot parse .npy header (${msg}) near: ${src.slice(i, i + 30)}`); };
  const ws = () => { while (i < src.length && /\s/.test(src[i])) i++; };

  function seq(close) {
    const out = [];
    i++; ws();
    while (src[i] !== close) {
      if (i >= src.length) fail('unterminated sequence');
      out.push(value()); ws();
      if (src[i] === ',') { i++; ws(); } else if (src[i] !== close) fail(`expected , or ${close}`);
    }
    i++;
    return out;
  }

  function value() {
    ws();
    const c = src[i];
    if (c === '{') {
      const obj = {};
      i++; ws();
      while (src[i] !== '}') {
        if (i >= src.length) fail('unterminated dict');
        const k = value(); ws();
        if (src[i] !== ':') fail('expected :');
        i++;
        obj[k] = value(); ws();
        if (src[i] === ',') { i++; ws(); } else if (src[i] !== '}') fail('expected , or }');
      }
      i++;
      return obj;
    }
    if (c === '(') return seq(')');
    if (c === '[') return seq(']');
    if (c === "'" || c === '"') {
      let s = '';
      i++;
      while (src[i] !== c) {
        if (i >= src.length) fail('unterminated string');
        if (src[i] === '\\') i++;
        s += src[i++];
      }
      i++;
      return s;
    }
    const m = /^[^\s,:)\]}]+/.exec(src.slice(i));
    if (!m) fail('unexpected token');
    i += m[0].length;
    const t = m[0];
    if (t === 'True') return true;
    if (t === 'False') return false;
    if (t === 'None') return null;
    const n = Number(t.replace(/L$/, ''));
    if (Number.isNaN(n)) fail(`unknown token ${t}`);
    return n;
  }

  return value();
}

function parseDtype(descr) {
  if (Array.isArray(descr)) {
    // Structured dtype: list of (name, dtype[, shape]) tuples.
    const fields = [];
    let offset = 0;
    for (const f of descr) {
      let [name, sub, shape] = f;
      if (Array.isArray(name)) name = name[1]; // (title, name)
      if (shape && shape.length) throw new Error(`Structured field "${name}" has a sub-array shape ${JSON.stringify(shape)}, which is not supported`);
      const t = parseDtype(sub);
      if (t.kind === 'struct') throw new Error('Nested structured dtypes are not supported');
      if (name !== '') fields.push({ name, type: t, offset }); // '' = padding
      offset += t.size;
    }
    return {
      kind: 'struct', fields, size: offset,
      str: '[' + fields.map(f => `('${f.name}', '${f.type.str}')`).join(', ') + ']',
    };
  }

  const m = /^([<>|=])?([a-zA-Z])(\d*)(?:\[(\w+)\])?$/.exec(descr);
  if (!m) throw new Error(`Unsupported dtype "${descr}"`);
  const [, endian = '<', kind, num, unit] = m;
  const n = num ? parseInt(num, 10) : 0;
  const t = { kind, little: endian !== '>', str: descr, size: n };
  const need = (ok, what) => { if (!ok.includes(n)) throw new Error(`Unsupported ${what} size in dtype "${descr}"`); };

  switch (kind) {
    case 'b': t.size = 1; break;
    case 'i': case 'u': need([1, 2, 4, 8], 'integer'); break;
    case 'f': need([2, 4, 8], 'float'); break;
    case 'c': need([8, 16], 'complex'); break;
    case 'U': t.size = n * 4; break;
    case 'S': case 'a': t.kind = 'S'; break;
    case 'V': break;
    case 'M': case 'm': t.size = 8; t.unit = unit || ''; break;
    case 'O': throw new Error('Object arrays (dtype=object) hold pickled Python objects and cannot be read in the browser');
    default: throw new Error(`Unsupported dtype "${descr}"`);
  }
  t.num = numReader(t);
  return t;
}

// Fast reader returning a plain Number (for stats and plots); null for non-numeric kinds.
function numReader(t) {
  const le = t.little;
  switch (t.kind) {
    case 'b': return (dv, o) => dv.getUint8(o) ? 1 : 0;
    case 'i':
      if (t.size === 1) return (dv, o) => dv.getInt8(o);
      if (t.size === 2) return (dv, o) => dv.getInt16(o, le);
      if (t.size === 4) return (dv, o) => dv.getInt32(o, le);
      return (dv, o) => Number(dv.getBigInt64(o, le));
    case 'u':
      if (t.size === 1) return (dv, o) => dv.getUint8(o);
      if (t.size === 2) return (dv, o) => dv.getUint16(o, le);
      if (t.size === 4) return (dv, o) => dv.getUint32(o, le);
      return (dv, o) => Number(dv.getBigUint64(o, le));
    case 'f':
      if (t.size === 2) return (dv, o) => half(dv.getUint16(o, le));
      if (t.size === 4) return (dv, o) => dv.getFloat32(o, le);
      return (dv, o) => dv.getFloat64(o, le);
    case 'm': return (dv, o) => { const v = dv.getBigInt64(o, le); return v === NAT ? NaN : Number(v); };
    default: return null;
  }
}

const NUMERIC_KINDS = new Set(['b', 'i', 'u', 'f', 'c', 'm']);
const NAT = -(2n ** 63n);
const utf8 = new TextDecoder('utf-8');

function half(h) {
  const s = (h & 0x8000) ? -1 : 1, e = (h >> 10) & 0x1f, f = h & 0x3ff;
  if (e === 0) return s * Math.pow(2, -14) * (f / 1024);
  if (e === 31) return f ? NaN : s * Infinity;
  return s * Math.pow(2, e - 15) * (1 + f / 1024);
}

const pad = (x, n = 2) => String(x).padStart(n, '0');

function fmtDatetime(v, unit) {
  if (v === NAT) return 'NaT';
  try {
    const n = Number(v);
    switch (unit) {
      case 'Y': return String(1970 + n);
      case 'M': return `${1970 + Math.floor(n / 12)}-${pad(((n % 12) + 12) % 12 + 1)}`;
      case 'W': return new Date(n * 7 * 864e5).toISOString().slice(0, 10);
      case 'D': return new Date(n * 864e5).toISOString().slice(0, 10);
      case 'h': return new Date(n * 3600e3).toISOString().slice(0, 13);
      case 'm': return new Date(n * 60e3).toISOString().slice(0, 16);
      case 's': case 'ms': case 'us': case 'ns': {
        const digits = { s: 0, ms: 3, us: 6, ns: 9 }[unit];
        const per = 10n ** BigInt(digits);
        let sec = v / per, rem = v % per;
        if (rem < 0n) { sec -= 1n; rem += per; }
        const iso = new Date(Number(sec) * 1000).toISOString().slice(0, 19);
        return digits ? `${iso}.${pad(rem, digits)}` : iso;
      }
    }
  } catch { /* out of JS Date range: fall through */ }
  return unit ? `${v} [${unit}]` : String(v);
}

function readValue(dv, off, t) {
  const le = t.little;
  switch (t.kind) {
    case 'b': return dv.getUint8(off) !== 0;
    case 'i':
      switch (t.size) {
        case 1: return dv.getInt8(off);
        case 2: return dv.getInt16(off, le);
        case 4: return dv.getInt32(off, le);
        default: return dv.getBigInt64(off, le);
      }
    case 'u':
      switch (t.size) {
        case 1: return dv.getUint8(off);
        case 2: return dv.getUint16(off, le);
        case 4: return dv.getUint32(off, le);
        default: return dv.getBigUint64(off, le);
      }
    case 'f':
      if (t.size === 2) return half(dv.getUint16(off, le));
      return t.size === 4 ? dv.getFloat32(off, le) : dv.getFloat64(off, le);
    case 'c':
      return t.size === 8
        ? [dv.getFloat32(off, le), dv.getFloat32(off + 4, le)]
        : [dv.getFloat64(off, le), dv.getFloat64(off + 8, le)];
    case 'U': {
      let s = '';
      for (let k = 0; k < t.size; k += 4) {
        const cp = dv.getUint32(off + k, le);
        if (cp === 0) break;
        s += cp <= 0x10ffff ? String.fromCodePoint(cp) : '�';
      }
      return s;
    }
    case 'S': {
      const bytes = new Uint8Array(dv.buffer, dv.byteOffset + off, t.size);
      let end = bytes.length;
      while (end > 0 && bytes[end - 1] === 0) end--;
      return utf8.decode(bytes.subarray(0, end));
    }
    case 'V': {
      let s = '';
      for (let k = 0; k < t.size; k++) s += pad(dv.getUint8(off + k).toString(16));
      return s;
    }
    case 'M': return fmtDatetime(dv.getBigInt64(off, le), t.unit);
    case 'js': return t.values[off];
    case 'm': {
      const v = dv.getBigInt64(off, le);
      return v === NAT ? 'NaT' : t.unit ? `${v} ${t.unit}` : String(v);
    }
  }
  return '?';
}

function fmtFloat(v, size, dec) {
  if (Number.isNaN(v)) return 'nan';
  if (!Number.isFinite(v)) return v > 0 ? 'inf' : '-inf';
  if (dec != null) return v.toFixed(dec);
  if (size === 4) {
    // shortest decimal that round-trips through float32
    for (let p = 1; p < 10; p++) {
      const s = v.toPrecision(p);
      if (Math.fround(parseFloat(s)) === v) return String(parseFloat(s));
    }
  }
  if (size === 2) return String(parseFloat(v.toPrecision(5)));
  return String(v);
}

function fmtJs(v, dec) {
  if (v == null) return '';
  switch (typeof v) {
    case 'string': return v;
    case 'boolean': return v ? 'True' : 'False';
    case 'number': return Number.isInteger(v) ? String(v) : fmtFloat(v, 8, dec);
    case 'bigint': return String(v);
  }
  const plain = ArrayBuffer.isView(v) ? Array.from(v) : v;
  try {
    return JSON.stringify(plain, (k, x) => typeof x === 'bigint' ? String(x) : ArrayBuffer.isView(x) ? Array.from(x) : x);
  } catch { return String(v); }
}

function formatValue(v, t, dec) {
  switch (t.kind) {
    case 'b': return v ? 'True' : 'False';
    case 'f': return fmtFloat(v, t.size, dec);
    case 'js': return fmtJs(v, dec);
    case 'c': {
      const half = t.size / 2, [re, im] = v;
      const sign = (im < 0 || Object.is(im, -0)) ? '-' : '+';
      return `${fmtFloat(re, half, dec)}${sign}${fmtFloat(Math.abs(im), half, dec)}j`;
    }
    default: return String(v);
  }
}

function parseNpy(u8, label) {
  const magic = [0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59]; // \x93NUMPY
  if (u8.length < 10 || magic.some((b, k) => u8[k] !== b)) throw new Error(`${label}: not a .npy file`);
  const major = u8[6];
  const dv0 = new DataView(u8.buffer, u8.byteOffset, u8.byteLength);
  let hlen, hstart;
  if (major === 1) { hlen = dv0.getUint16(8, true); hstart = 10; }
  else { hlen = dv0.getUint32(8, true); hstart = 12; }

  const header = new TextDecoder(major >= 3 ? 'utf-8' : 'latin1').decode(u8.subarray(hstart, hstart + hlen));
  const h = parsePyLiteral(header);
  const type = parseDtype(h.descr);
  const shape = (h.shape || []).map(Number);
  const fortran = !!h.fortran_order;
  const dataStart = hstart + hlen;
  const size = shape.reduce((a, b) => a * b, 1);
  const nbytes = size * type.size;
  if (u8.byteLength - dataStart < nbytes) {
    throw new Error(`${label}: file is truncated (expected ${nbytes} data bytes, found ${u8.byteLength - dataStart})`);
  }
  const dv = new DataView(u8.buffer, u8.byteOffset + dataStart, nbytes);
  return { ...buildArray({ shape, type, fortran, dv }), version: `${major}.${u8[7]}` };
}

// Wrap raw element storage into the array object the viewer works with.
// `type.size` is the byte step between elements (1 for 'js' value arrays).
function buildArray({ shape, type, fortran = false, dv, dimNames }) {
  const size = shape.reduce((a, b) => a * b, 1);
  const nbytes = type.kind === 'js' ? (type.bytes ?? 0) : size * type.size;

  // byte strides for each axis
  const byteStrides = new Array(shape.length);
  let s = type.size;
  if (fortran) for (let k = 0; k < shape.length; k++) { byteStrides[k] = s; s *= shape[k]; }
  else for (let k = shape.length - 1; k >= 0; k--) { byteStrides[k] = s; s *= shape[k]; }

  // Structured arrays get an extra virtual "fields" axis.
  const dims = shape.slice();
  dimNames = dimNames ? dimNames.slice() : shape.map((_, k) => `axis ${k}`);
  let fieldDim = -1;
  if (type.kind === 'struct') {
    fieldDim = dims.length;
    dims.push(type.fields.length);
    dimNames.push('fields');
    byteStrides.push(0);
  }

  return { shape, dims, dimNames, fieldDim, type, fortran, dv, byteStrides, size, nbytes };
}

async function inflateRaw(u8) {
  const stream = new Blob([u8]).stream().pipeThrough(new DecompressionStream('deflate-raw'));
  return new Uint8Array(await new Response(stream).arrayBuffer());
}

async function parseNpz(u8) {
  const dv = new DataView(u8.buffer, u8.byteOffset, u8.byteLength);

  // End of central directory record
  let eocd = -1;
  for (let p = u8.length - 22; p >= Math.max(0, u8.length - 22 - 65535); p--) {
    if (dv.getUint32(p, true) === 0x06054b50) { eocd = p; break; }
  }
  if (eocd < 0) throw new Error('Not a valid .npz file (zip directory not found)');
  let count = dv.getUint16(eocd + 10, true);
  let cdOffset = dv.getUint32(eocd + 16, true);

  // Zip64
  const loc = eocd - 20;
  if (loc >= 0 && dv.getUint32(loc, true) === 0x07064b50) {
    const z = Number(dv.getBigUint64(loc + 8, true));
    if (dv.getUint32(z, true) === 0x06064b50) {
      count = Number(dv.getBigUint64(z + 32, true));
      cdOffset = Number(dv.getBigUint64(z + 48, true));
    }
  }

  const entries = [];
  let p = cdOffset;
  for (let e = 0; e < count; e++) {
    if (dv.getUint32(p, true) !== 0x02014b50) throw new Error('Corrupt .npz central directory');
    const method = dv.getUint16(p + 10, true);
    let compSize = dv.getUint32(p + 20, true);
    let size = dv.getUint32(p + 24, true);
    const nameLen = dv.getUint16(p + 28, true);
    const extraLen = dv.getUint16(p + 30, true);
    const commentLen = dv.getUint16(p + 32, true);
    let localOffset = dv.getUint32(p + 42, true);
    const name = utf8.decode(u8.subarray(p + 46, p + 46 + nameLen));

    // Zip64 extended information extra field
    let x = p + 46 + nameLen;
    const xEnd = x + extraLen;
    while (x + 4 <= xEnd) {
      const id = dv.getUint16(x, true), len = dv.getUint16(x + 2, true);
      if (id === 0x0001) {
        let q = x + 4;
        if (size === 0xffffffff) { size = Number(dv.getBigUint64(q, true)); q += 8; }
        if (compSize === 0xffffffff) { compSize = Number(dv.getBigUint64(q, true)); q += 8; }
        if (localOffset === 0xffffffff) { localOffset = Number(dv.getBigUint64(q, true)); q += 8; }
      }
      x += 4 + len;
    }
    entries.push({ name, method, compSize, size, localOffset });
    p = xEnd + commentLen;
  }

  const arrays = [];
  for (const en of entries) {
    if (en.name.endsWith('/')) continue;
    const label = en.name.replace(/\.npy$/i, '');
    try {
      const lo = en.localOffset;
      if (dv.getUint32(lo, true) !== 0x04034b50) throw new Error('corrupt local header');
      const start = lo + 30 + dv.getUint16(lo + 26, true) + dv.getUint16(lo + 28, true);
      const raw = u8.subarray(start, start + en.compSize);
      let data;
      if (en.method === 0) data = raw;
      else if (en.method === 8) data = await inflateRaw(raw);
      else throw new Error(`unsupported zip compression method ${en.method}`);
      arrays.push({ name: label, ...parseNpy(data, en.name) });
    } catch (err) {
      arrays.push({ name: label, error: err.message });
    }
  }
  if (!arrays.length) throw new Error('The .npz archive contains no arrays');
  return arrays;
}

/* =====================================================================
 * Slicing / table access
 * ===================================================================== */

// Python-style range normalisation; empty fields mean defaults.
function normRange(r, n) {
  const fix = (x, d) => {
    if (x === '' || x == null || Number.isNaN(+x)) return d;
    x = Math.trunc(+x);
    if (x < 0) x += n;
    return Math.min(Math.max(x, 0), n);
  };
  const start = fix(r.start, 0);
  const stop = fix(r.stop, n);
  const step = Math.max(1, Math.trunc(+r.step) || 1);
  const count = stop > start ? Math.ceil((stop - start) / step) : 0;
  return { start, stop, step, count, at: (i) => start + i * step };
}

function normIndex(x, n) {
  if (n === 0) return -1;
  x = Math.trunc(+x) || 0;
  if (x < 0) x += n;
  return Math.min(Math.max(x, 0), n - 1);
}

const SINGLE = { start: 0, stop: 1, step: 1, count: 1, at: () => 0 };

function makeAccessor(arr, view, dec) {
  const { dims, fieldDim: fd, byteStrides, dv, type } = arr;
  const { rowDim, colDim } = view;
  const R = rowDim >= 0 ? normRange(view.ranges[rowDim], dims[rowDim]) : { ...SINGLE };
  const C = colDim >= 0 ? normRange(view.ranges[colDim], dims[colDim]) : { ...SINGLE };

  let base = 0, field = 0, empty = false;
  const fixedIdx = [];
  for (let k = 0; k < dims.length; k++) {
    if (k === rowDim || k === colDim) continue;
    const idx = normIndex(view.fixed[k], dims[k]);
    fixedIdx[k] = idx;
    if (idx < 0) { empty = true; continue; }
    if (k === fd) field = idx;
    else base += idx * byteStrides[k];
  }
  if (empty) { R.count = 0; C.count = 0; }

  const rs = rowDim >= 0 && rowDim !== fd ? byteStrides[rowDim] : 0;
  const cs = colDim >= 0 && colDim !== fd ? byteStrides[colDim] : 0;

  let cell;
  if (fd < 0) {
    cell = (r, c) => formatValue(readValue(dv, base + r * rs + c * cs, type), type, dec);
  } else {
    cell = (r, c) => {
      const f = type.fields[rowDim === fd ? r : colDim === fd ? c : field];
      return formatValue(readValue(dv, base + r * rs + c * cs + f.offset, f.type), f.type, dec);
    };
  }
  // numeric value; `x` is an extra byte offset (used to step along an RGB axis)
  let num = null;
  if (fd < 0) {
    const nr = type.num;
    if (nr) num = (r, c, x = 0) => nr(dv, base + r * rs + c * cs + x);
  } else if (type.fields.some((f) => f.type.num)) {
    num = (r, c, x = 0) => {
      const f = type.fields[rowDim === fd ? r : colDim === fd ? c : field];
      return f.type.num ? f.type.num(dv, base + r * rs + c * cs + x + f.offset) : undefined;
    };
  }

  const label = (dim, i) => dim < 0 ? '' : dim === fd ? type.fields[i].name
    : arr.axisLabels && arr.axisLabels[dim] ? String(arr.axisLabels[dim][i]) : String(i);
  // full index tuple of a cell in the original array
  const index = (r, c) => dims.map((_, k) => {
    const v = k === rowDim ? r : k === colDim ? c : fixedIdx[k];
    return k === fd ? `'${type.fields[v].name}'` : v;
  });

  return {
    R, C, cell, num, index, fixedIdx,
    rowLabel: (r) => label(rowDim, r),
    colLabel: (c) => colDim < 0 ? 'value' : label(colDim, c),
    corner: [rowDim >= 0 ? arr.dimNames[rowDim] + ' ↓' : '', colDim >= 0 ? arr.dimNames[colDim] + ' →' : '']
      .filter(Boolean).join(' / '),
  };
}

function sliceExpression(arr, view, acc) {
  const parts = [];
  let prefix = '';
  for (let k = 0; k < arr.dims.length; k++) {
    let s;
    if (k === view.rowDim || k === view.colDim) {
      const r = k === view.rowDim ? acc.R : acc.C;
      s = `${r.start}:${r.stop}${r.step > 1 ? ':' + r.step : ''}`;
    } else {
      s = String(acc.fixedIdx[k]);
    }
    if (k === arr.fieldDim) {
      if (k !== view.rowDim && k !== view.colDim && acc.fixedIdx[k] >= 0) {
        prefix = `['${arr.type.fields[acc.fixedIdx[k]].name}']`;
      }
    } else {
      parts.push(s);
    }
  }
  return `${arr.name}${prefix}${parts.length ? '[' + parts.join(', ') + ']' : ''}`;
}

/* =====================================================================
 * Statistics
 * ===================================================================== */

class Stats {
  constructor() {
    this.n = 0; this.nan = 0; this.sum = 0; this.mean = 0; this.m2 = 0;
    this.min = Infinity; this.max = -Infinity; this.minAt = -1; this.maxAt = -1;
  }
  add(v, at) {
    if (v === undefined) return;          // non-numeric struct field
    if (v !== v) { this.nan++; return; }  // NaN / NaT
    this.n++;
    this.sum += v;
    if (this.n === 1 || v < this.min) { this.min = v; this.minAt = at; }
    if (this.n === 1 || v > this.max) { this.max = v; this.maxAt = at; }
    const d = v - this.mean;               // Welford, for a stable std
    this.mean += d / this.n;
    this.m2 += d * (v - this.mean);
  }
  get std() { return Math.sqrt(this.m2 / this.n); }
}

const tick = () => new Promise((r) => setTimeout(r, 0));
const fmtIdx = (a) => a.length ? `[${a.join(', ')}]` : '()';

// memory-order flat index -> multi-index
function unravel(arr, flat) {
  const sh = arr.shape, idx = new Array(sh.length);
  if (arr.fortran) for (let k = 0; k < sh.length; k++) { idx[k] = flat % sh[k]; flat = Math.floor(flat / sh[k]); }
  else for (let k = sh.length - 1; k >= 0; k--) { idx[k] = flat % sh[k]; flat = Math.floor(flat / sh[k]); }
  return idx;
}

// Stats over every element (one result per numeric field for structured arrays).
async function computeWholeStats(arr, onProgress) {
  const { type, dv, size } = arr;
  const targets = type.kind === 'struct'
    ? type.fields.filter((f) => f.type.num).map((f) => ({ label: f.name, nr: f.type.num, off: f.offset, field: f.name }))
    : type.num ? [{ label: 'Whole array', nr: type.num, off: 0 }] : [];
  const stats = targets.map(() => new Stats());
  const isz = type.size, CH = 1 << 19;
  for (let s = 0; s < size && targets.length; s += CH) {
    const e = Math.min(size, s + CH);
    targets.forEach((t, ti) => {
      const st = stats[ti], nr = t.nr, off = t.off;
      for (let i = s; i < e; i++) st.add(nr(dv, i * isz + off), i);
    });
    if (e < size) { onProgress(e / size); await tick(); }
  }
  return targets.map((t, i) => {
    stats[i].loc = (at) => fmtIdx([...unravel(arr, at), ...(t.field ? [`'${t.field}'`] : [])]);
    return { label: t.label, stats: stats[i] };
  });
}

/* =====================================================================
 * UI
 * ===================================================================== */

if (typeof document !== 'undefined') {
  const $ = (id) => document.getElementById(id);
  const el = Object.fromEntries([
    'fileInput', 'fileName', 'sidebar', 'arrayList', 'error', 'dropZone', 'viewer', 'arrayInfo', 'dimTable',
    'swapBtn', 'decimals', 'pageSize', 'includeLabels', 'exportBtn', 'sliceExpr', 'sliceSize', 'notice',
    'tableWrap', 'pageInfo', 'firstBtn', 'prevBtn', 'nextBtn', 'lastBtn', 'statsBody', 'statsStatus', 'tabs',
    'tablePane', 'linePane', 'imagePane', 'lineLegend', 'lineBox', 'lineCanvas', 'lineNote', 'imgMode', 'cmap',
    'vmin', 'vmax', 'aspect', 'imgBox', 'imgCanvas', 'imgMsg', 'colorbar', 'cbRamp', 'cbMin', 'cbMid',
    'cbMax', 'imgNote', 'tooltip',
  ].map((id) => [id, $(id)]));

  const MAX_RENDER_COLS = 1000;
  const MAX_SERIES = 8;
  const IMG_MAX = 2048;
  const S = { fileName: '', arrays: [], arr: null, view: null, acc: null, page: 0, tab: 'table', sel: null, selToken: 0 };

  const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
  const fmtBytes = (n) => n < 1024 ? `${n} B` : n < 1048576 ? `${(n / 1024).toFixed(1)} KB` : `${(n / 1048576).toFixed(1)} MB`;
  const shapeStr = (shape) => `(${shape.join(', ')}${shape.length === 1 ? ',' : ''})`;
  const decimals = () => el.decimals.value === '' ? null : Math.min(20, Math.max(0, parseInt(el.decimals.value, 10) || 0));
  const cssVar = (n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
  const seriesColor = (j) => cssVar(`--series-${j + 1}`);
  const darkMQ = window.matchMedia('(prefers-color-scheme: dark)');

  function fmtStat(v) {
    if (Number.isNaN(v)) return 'nan';
    if (!Number.isFinite(v)) return v > 0 ? 'inf' : '-inf';
    if (Number.isInteger(v)) return String(v);
    const dec = decimals();
    return dec != null ? v.toFixed(dec) : String(+v.toPrecision(8));
  }
  function fmtTick(v) {
    const a = Math.abs(v);
    if (a !== 0 && (a >= 1e6 || a < 1e-3)) return v.toExponential(1);
    return String(+v.toPrecision(6));
  }
  function niceTicks(lo, hi, n) {
    const raw = (hi - lo) / Math.max(1, n);
    const mag = 10 ** Math.floor(Math.log10(raw));
    const f = raw / mag;
    const step = (f >= 7.5 ? 10 : f >= 3.5 ? 5 : f >= 1.5 ? 2 : 1) * mag;
    const out = [];
    for (let v = Math.floor(lo / step) * step; v <= Math.ceil(hi / step) * step + step / 2; v += step) out.push(+v.toPrecision(12));
    return out;
  }

  function showError(msg) {
    el.error.textContent = msg;
    el.error.classList.toggle('hidden', !msg);
  }

  /* ---------- loading ---------- */

  async function loadFile(file) {
    showError('');
    el.fileName.textContent = `Loading ${file.name}…`;
    try {
      const buf = new Uint8Array(await file.arrayBuffer());
      let arrays;
      const isHdf5 = buf[0] === 0x89 && buf[1] === 0x48 && buf[2] === 0x44 && buf[3] === 0x46
        || /\.(h5|hdf5|he5|hdf)$/i.test(file.name);
      if (isHdf5) {
        if (typeof window.loadHdf5 !== 'function') throw new Error('This is an HDF5 file — open it with h5.html.');
        arrays = await window.loadHdf5(buf, file.name);
      } else if (buf[0] === 0x93 && buf[1] === 0x4e) {
        arrays = [{ name: file.name.replace(/\.npy$/i, ''), ...parseNpy(buf, file.name) }];
      } else if (buf[0] === 0x50 && buf[1] === 0x4b) {
        arrays = await parseNpz(buf);
      } else {
        throw new Error('Unrecognised file format: expected a NumPy .npy or .npz file');
      }
      S.fileName = file.name;
      S.arrays = arrays;
      el.fileName.textContent = `${file.name} · ${fmtBytes(file.size)}`;
      el.dropZone.classList.add('hidden');
      buildSidebar(arrays.length > 1 || arrays.some((a) => a.group));
      const pref = arrays.findIndex((a) => a.preferred);
      const first = pref >= 0 ? pref : arrays.findIndex((a) => !a.error && !a.group);
      if (first >= 0) selectArray(first);
      else { el.viewer.classList.add('hidden'); showError('No readable arrays in this file.'); }
    } catch (err) {
      console.error(err);
      el.fileName.textContent = '';
      showError(`Could not open ${file.name}:\n${err.message}`);
    }
  }

  function buildSidebar(show) {
    el.sidebar.classList.toggle('hidden', !show);
    el.arrayList.innerHTML = S.arrays.map((a, i) => {
      const indent = `style="padding-left:${16 + (a.depth || 0) * 14}px"`;
      if (a.group) {
        return `<li class="group" ${indent} title="${esc(a.tip || a.path || '')}"><div class="name">▾ ${esc(a.label || a.name)}</div>${a.meta ? `<div class="meta">${esc(a.meta)}</div>` : ''}</li>`;
      }
      const meta = a.error ? '⚠ ' + a.error
        : a.meta || shapeStr(a.shape) + ' ' + (a.type.kind === 'struct' ? 'structured' : a.type.str);
      return `<li data-i="${i}" class="${a.error ? 'bad' : ''}" ${indent} title="${esc(a.error || a.path || '')}">
        <div class="name">${esc(a.label || a.name)}</div>
        <div class="meta">${esc(meta)}</div>
      </li>`;
    }).join('');
  }

  async function selectArray(i) {
    let arr = S.arrays[i];
    if (!arr || arr.group) return;
    if (arr.lazy) {  // HDF5 datasets are read on first selection
      for (const li of el.arrayList.children) li.classList.toggle('loading', +li.dataset.i === i);
      try {
        Object.assign(arr, await arr.lazy());
      } catch (err) {
        console.error(err);
        arr.error = err.message;
        buildSidebar(true);
      }
      delete arr.lazy;
      for (const li of el.arrayList.children) li.classList.remove('loading');
    }
    if (arr.error) { showError(`${arr.name}: ${arr.error}`); return; }
    showError('');
    for (const li of el.arrayList.children) li.classList.toggle('active', +li.dataset.i === i);
    const nd = arr.dims.length;
    S.arr = arr;
    S.page = 0;
    S.view = {
      rowDim: nd >= 2 ? nd - 2 : nd - 1,
      colDim: nd >= 2 ? nd - 1 : -1,
      ranges: arr.dims.map(() => ({ start: '', stop: '', step: '' })),
      fixed: arr.dims.map(() => 0),
    };
    el.vmin.value = el.vmax.value = '';
    el.viewer.classList.remove('hidden');
    el.arrayInfo.innerHTML = `
      <span class="title">${esc(arr.name)}</span>
      <span class="kv"><span>shape</span> ${esc(shapeStr(arr.shape))}</span>
      <span class="kv"><span>dtype</span> ${esc(arr.type.str)}</span>
      <span class="kv"><span>size</span> ${arr.size.toLocaleString()} (${fmtBytes(arr.nbytes)})</span>
      <span class="kv"><span>order</span> ${arr.fortran ? 'Fortran' : 'C'}</span>
      ${arr.extraInfo ? arr.extraInfo.map(([k, v]) => `<span class="kv"><span>${esc(k)}</span> ${esc(v)}</span>`).join('') : ''}
      ${arr.note ? `<div class="note">${esc(arr.note)}</div>` : ''}
      ${arr.attrs && arr.attrs.length ? `
        <details class="attrs"><summary>Attributes (${arr.attrs.length})</summary>
          <table>${arr.attrs.map(([k, v]) => `<tr><th>${esc(k)}</th><td>${esc(v)}</td></tr>`).join('')}</table>
        </details>` : ''}`;
    startWholeStats(arr);
    buildDimTable();
    render();
  }

  /* ---------- dimension controls ---------- */

  function roleOf(k) {
    return S.view.rowDim === k ? 'rows' : S.view.colDim === k ? 'cols' : 'fixed';
  }

  function buildDimTable() {
    const { arr, view } = S;
    updateImgModes();
    if (!arr.dims.length) {
      el.dimTable.innerHTML = '<tr><td class="muted">0-dimensional array (a single scalar value)</td></tr>';
      return;
    }
    const rows = arr.dims.map((n, k) => {
      const role = roleOf(k);
      const ranged = role !== 'fixed';
      const r = view.ranges[k];
      const opt = (v, t) => `<option value="${v}"${role === v ? ' selected' : ''}>${t}</option>`;
      return `
        <tr class="role-${role}">
          <td class="axis">${esc(arr.dimNames[k])}</td>
          <td>${n}</td>
          <td><select data-k="${k}" data-f="role">${opt('rows', 'Rows')}${opt('cols', 'Columns')}${opt('fixed', 'Fixed index')}</select></td>
          <td><input type="number" data-k="${k}" data-f="start" value="${r.start}" placeholder="0" ${ranged ? '' : 'disabled'}></td>
          <td><input type="number" data-k="${k}" data-f="stop" value="${r.stop}" placeholder="${n}" ${ranged ? '' : 'disabled'}></td>
          <td><input type="number" data-k="${k}" data-f="step" value="${r.step}" placeholder="1" min="1" ${ranged ? '' : 'disabled'}></td>
          <td>
            <input type="number" data-k="${k}" data-f="index" value="${view.fixed[k]}" min="${-n}" max="${n - 1}" ${ranged ? 'disabled' : ''}>
            <input type="range" data-k="${k}" data-f="slider" value="${normIndex(view.fixed[k], n)}" min="0" max="${Math.max(0, n - 1)}" ${ranged || n <= 1 ? 'disabled' : ''}>
            ${k === arr.fieldDim && !ranged && n ? `<span class="muted">${esc(arr.type.fields[normIndex(view.fixed[k], n)].name)}</span>` : ''}
          </td>
        </tr>`;
    }).join('');
    el.dimTable.innerHTML = `
      <thead><tr><th>Axis</th><th>Size</th><th>Show as</th><th>Start</th><th>Stop</th><th>Step</th><th>Index (fixed axes)</th></tr></thead>
      <tbody>${rows}</tbody>`;
  }

  function setRole(k, role) {
    const v = S.view;
    const old = roleOf(k);
    if (role === old) return;
    const holder = role === 'rows' ? v.rowDim : role === 'cols' ? v.colDim : -1;
    if (old === 'rows') v.rowDim = -1; else if (old === 'cols') v.colDim = -1;
    // the axis that previously held the requested role takes over k's old role
    if (holder >= 0) { if (old === 'rows') v.rowDim = holder; else if (old === 'cols') v.colDim = holder; }
    if (role === 'rows') v.rowDim = k; else if (role === 'cols') v.colDim = k;
    S.page = 0;
    buildDimTable();
    render();
  }

  /* ---------- rendering ---------- */

  let renderTimer = 0;
  function scheduleRender() {
    clearTimeout(renderTimer);
    renderTimer = setTimeout(render, 120);
  }

  function render() {
    const { arr, view } = S;
    if (!arr) return;
    const acc = makeAccessor(arr, view, decimals());
    S.acc = acc;
    el.sliceExpr.textContent = sliceExpression(arr, view, acc);
    el.sliceSize.textContent = `→ ${acc.R.count.toLocaleString()} × ${acc.C.count.toLocaleString()} table`;
    renderView();
    renderSelStats(acc);
  }

  function renderView() {
    const acc = S.acc;
    if (!acc) return;
    if (S.tab === 'table') renderTable(acc);
    else if (S.tab === 'line') { lineModel = buildLineModel(acc); renderLegend(); drawLine(); }
    else renderImage(acc);
  }

  function renderTable(acc) {
    const { arr } = S;
    const { R, C } = acc;
    const pageSize = +el.pageSize.value;
    const pages = Math.max(1, Math.ceil(R.count / pageSize));
    S.page = Math.min(Math.max(0, S.page), pages - 1);
    const r0 = S.page * pageSize, r1 = Math.min(R.count, r0 + pageSize);
    const nCols = Math.min(C.count, MAX_RENDER_COLS);

    if (C.count > MAX_RENDER_COLS) {
      el.notice.textContent = `Showing the first ${MAX_RENDER_COLS.toLocaleString()} of ${C.count.toLocaleString()} columns. Narrow the column range to see others — CSV export always includes the full selection.`;
      el.notice.classList.remove('hidden');
    } else {
      el.notice.classList.add('hidden');
    }

    if (R.count === 0 || C.count === 0) {
      el.tableWrap.innerHTML = '<div class="empty">The selection is empty.</div>';
    } else {
      const numeric = arr.type.kind === 'struct'
        ? arr.type.fields.every((f) => NUMERIC_KINDS.has(f.type.kind))
        : NUMERIC_KINDS.has(arr.type.kind) || (arr.type.kind === 'js' && !!arr.type.num);
      const html = [];
      html.push(`<table class="data${numeric ? ' numeric' : ''}"><thead><tr><th class="corner">${esc(acc.corner)}</th>`);
      for (let j = 0; j < nCols; j++) html.push(`<th>${esc(acc.colLabel(C.at(j)))}</th>`);
      html.push('</tr></thead><tbody>');
      for (let i = r0; i < r1; i++) {
        const r = R.at(i);
        html.push(`<tr><th>${esc(acc.rowLabel(r))}</th>`);
        for (let j = 0; j < nCols; j++) html.push(`<td>${esc(acc.cell(r, C.at(j)))}</td>`);
        html.push('</tr>');
      }
      html.push('</tbody></table>');
      el.tableWrap.innerHTML = html.join('');
    }

    el.pageInfo.textContent = R.count
      ? `Rows ${(r0 + 1).toLocaleString()}–${r1.toLocaleString()} of ${R.count.toLocaleString()} · page ${S.page + 1}/${pages}`
      : 'No rows';
    el.firstBtn.disabled = el.prevBtn.disabled = S.page === 0;
    el.lastBtn.disabled = el.nextBtn.disabled = S.page >= pages - 1;
  }

  /* ---------- statistics ---------- */

  const STAT_ROWS = [
    ['Count', (s) => s.n.toLocaleString()],
    ['NaN', (s) => s.nan.toLocaleString()],
    ['Min', (s) => s.n ? fmtStat(s.min) : '—'],
    ['Min at', (s) => s.n ? s.loc(s.minAt) : '—'],
    ['Max', (s) => s.n ? fmtStat(s.max) : '—'],
    ['Max at', (s) => s.n ? s.loc(s.maxAt) : '—'],
    ['Mean', (s) => s.n ? fmtStat(s.sum / s.n) : '—'],
    ['Std', (s) => s.n ? fmtStat(s.std) : '—'],
    ['Sum', (s) => s.n ? fmtStat(s.sum) : '—'],
  ];

  function startWholeStats(arr) {
    if (arr._whole) return;
    const w = arr._whole = { progress: 0, result: null };
    computeWholeStats(arr, (p) => { w.progress = p; if (S.arr === arr) renderStatsTable(); })
      .then((res) => { w.result = res; if (S.arr === arr) renderStatsTable(); });
  }

  async function renderSelStats(acc) {
    const token = ++S.selToken;
    if (!acc.num) { S.sel = { na: true }; renderStatsTable(); return; }
    const { R, C } = acc;
    const total = R.count * C.count, CH = 1 << 19;
    const st = new Stats();
    let k = 0, next = CH;
    for (let i = 0; i < R.count; i++) {
      const r = R.at(i);
      for (let j = 0; j < C.count; j++) st.add(acc.num(r, C.at(j)), k++);
      if (k >= next) {
        next = k + CH;
        S.sel = { progress: k / total };
        renderStatsTable();
        await tick();
        if (token !== S.selToken) return;
      }
    }
    st.loc = (at) => fmtIdx(acc.index(R.at(Math.floor(at / C.count)), C.at(at % C.count)));
    S.sel = { stats: st };
    renderStatsTable();
  }

  function renderStatsTable() {
    const arr = S.arr;
    if (!arr) return;
    const cols = [{ title: 'Current selection', data: S.sel || { progress: 0 } }];
    const w = arr._whole;
    if (w && w.result) {
      if (!w.result.length) cols.push({ title: 'Whole array', data: { na: true } });
      for (const x of w.result) {
        cols.push({ title: arr.type.kind === 'struct' ? `Whole array · ${x.label}` : 'Whole array', data: { stats: x.stats } });
      }
    } else {
      cols.push({ title: 'Whole array', data: { progress: w ? w.progress : 0 } });
    }

    if (cols.every((c) => c.data.na)) {
      el.statsBody.innerHTML = '<div class="muted">Statistics are available for numeric data only.</div>';
      el.statsStatus.textContent = '';
      return;
    }
    const val = (d, f) => d.na ? 'n/a' : d.stats ? f(d.stats) : '…';
    el.statsBody.innerHTML = `
      <table class="stats">
        <thead><tr><th></th>${cols.map((c) => `<th>${esc(c.title)}</th>`).join('')}</tr></thead>
        <tbody>${STAT_ROWS.map(([name, f]) => `<tr><th>${name}</th>${cols.map((c) => `<td>${esc(val(c.data, f))}</td>`).join('')}</tr>`).join('')}</tbody>
      </table>`;
    const busy = cols.filter((c) => c.data.progress != null);
    el.statsStatus.textContent = busy.length
      ? 'computing… ' + busy.map((c) => `${c.title.toLowerCase()} ${Math.round(c.data.progress * 100)}%`).join(', ')
      : 'NaN values are excluded · std is the population std (ddof=0)';
  }

  /* ---------- tooltip ---------- */

  function showTip(e, html) {
    const t = el.tooltip;
    t.innerHTML = html;
    t.classList.remove('hidden');
    const r = t.getBoundingClientRect(), pad = 14;
    let x = e.clientX + pad, y = e.clientY + pad;
    if (x + r.width > window.innerWidth - 8) x = e.clientX - r.width - pad;
    if (y + r.height > window.innerHeight - 8) y = e.clientY - r.height - pad;
    t.style.left = `${Math.max(4, x)}px`;
    t.style.top = `${Math.max(4, y)}px`;
  }
  const hideTip = () => el.tooltip.classList.add('hidden');

  /* ---------- line plot ---------- */

  let lineModel = null;

  // Each column of the selection becomes a line over the row axis
  // (or a single line over the column axis when no axis is on rows).
  function buildLineModel(acc) {
    const { arr, view } = S;
    if (!acc.num) return { error: 'Line plots need numeric data.' };
    let X = acc.R, Ser = acc.C, get = acc.num, xDim = view.rowDim, sLabel = acc.colLabel;
    let fmt = (x, s) => acc.cell(x, s);
    if (view.rowDim < 0 && view.colDim >= 0) {
      X = acc.C; Ser = acc.R; xDim = view.colDim;
      get = (x, s) => acc.num(s, x);
      fmt = (x, s) => acc.cell(s, x);
      sLabel = () => 'value';
    }
    if (!X.count || !Ser.count) return { error: 'The selection is empty.' };

    const ns = Math.min(Ser.count, MAX_SERIES);
    const nb = Math.min(X.count, 2000);   // pixel-ish buckets; min/max envelope when decimating
    const dec = nb < X.count;
    let lo = Infinity, hi = -Infinity;
    const series = [];
    for (let j = 0; j < ns; j++) {
      const s = Ser.at(j);
      const mn = new Float64Array(nb).fill(NaN), mx = new Float64Array(nb).fill(NaN);
      for (let i = 0; i < X.count; i++) {
        const v = get(X.at(i), s);
        if (v === undefined || !Number.isFinite(v)) continue;
        const b = dec ? Math.floor(i * nb / X.count) : i;
        if (!(v >= mn[b])) mn[b] = v;
        if (!(v <= mx[b])) mx[b] = v;
        if (v < lo) lo = v;
        if (v > hi) hi = v;
      }
      series.push({ label: sLabel(s), s, mn, mx });
    }
    const xLabels = xDim >= 0 && arr.axisLabels ? arr.axisLabels[xDim] : null;
    return { X, xName: xDim >= 0 ? arr.dimNames[xDim] : 'index', xLabels, series, nb, dec, lo, hi, get, fmt, total: Ser.count };
  }

  function renderLegend() {
    const m = lineModel;
    if (!m || m.error) { el.lineLegend.innerHTML = ''; el.lineNote.textContent = ''; return; }
    el.lineLegend.innerHTML = m.series.length > 1
      ? m.series.map((s, j) => `<span class="lg"><span class="lg-key" style="background:${seriesColor(j)}"></span>${esc(s.label)}</span>`).join('')
      : `<span class="lg-title">${esc(el.sliceExpr.textContent)}</span>`;
    const notes = [`x axis: ${m.xName}`];
    if (m.series.length > 1) notes.push(`one line per column (${S.view.colDim >= 0 ? S.arr.dimNames[S.view.colDim] : ''})`);
    if (m.total > MAX_SERIES) notes.push(`showing the first ${MAX_SERIES} of ${m.total.toLocaleString()} columns — narrow the column range or swap rows/columns`);
    if (m.dec) notes.push(`${m.X.count.toLocaleString()} points drawn as a min/max envelope`);
    el.lineNote.textContent = notes.join(' · ');
  }

  function drawLine(hover = null) {
    const m = lineModel;
    const cv = el.lineCanvas, ctx = cv.getContext('2d');
    const W = el.lineBox.clientWidth, H = 420, dpr = window.devicePixelRatio || 1;
    if (cv.width !== Math.round(W * dpr) || cv.height !== Math.round(H * dpr)) {
      cv.width = Math.round(W * dpr); cv.height = Math.round(H * dpr);
      cv.style.width = `${W}px`; cv.style.height = `${H}px`;
    }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, W, H);
    const muted = cssVar('--muted');
    if (!m || m.error) {
      ctx.fillStyle = muted;
      ctx.font = '14px system-ui, sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText(m ? m.error : '', W / 2, H / 2);
      if (m) m.layout = null;
      return;
    }

    let lo = m.lo, hi = m.hi;
    if (!Number.isFinite(lo)) { lo = 0; hi = 1; } else if (lo === hi) { const d = Math.abs(lo) * 0.1 || 1; lo -= d; hi += d; }
    const yt = niceTicks(lo, hi, 6);
    const y0 = yt[0], y1 = yt[yt.length - 1];
    ctx.font = `11px ${cssVar('--mono')}`;
    const yLabels = yt.map(fmtTick);
    const P = { l: Math.max(...yLabels.map((t) => ctx.measureText(t).width)) + 16, r: 18, t: 12, b: 46 };
    const pw = W - P.l - P.r, ph = H - P.t - P.b;
    const n = m.X.count;
    const xOf = (pos) => P.l + (n === 1 ? pw / 2 : (pos / (n - 1)) * pw);
    const yOf = (v) => P.t + (1 - (v - y0) / (y1 - y0)) * ph;

    // grid + y labels
    ctx.lineWidth = 1;
    ctx.strokeStyle = cssVar('--grid');
    ctx.fillStyle = muted;
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    yt.forEach((v, k) => {
      const y = Math.round(yOf(v)) + 0.5;
      ctx.beginPath(); ctx.moveTo(P.l, y); ctx.lineTo(W - P.r, y); ctx.stroke();
      ctx.fillText(yLabels[k], P.l - 8, y);
    });
    // x ticks (actual array indices)
    const a = m.X.at(0), b = m.X.at(n - 1);
    const xText = (t) => m.xLabels ? String(m.xLabels[t]) : String(t);
    const tickW = Math.max(ctx.measureText(xText(a)).width, ctx.measureText(xText(b)).width) + 36;
    const xt = n > 1
      ? niceTicks(a, b, Math.max(2, Math.floor(pw / Math.max(90, tickW)))).filter((t) => t >= a && t <= b && Number.isInteger(t))
      : [a];
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    ctx.strokeStyle = cssVar('--border');
    for (const t of xt) {
      const x = Math.round(xOf((t - m.X.start) / m.X.step)) + 0.5;
      ctx.beginPath(); ctx.moveTo(x, P.t + ph); ctx.lineTo(x, P.t + ph + 4); ctx.stroke();
      const txt = xText(t), half = ctx.measureText(txt).width / 2;  // keep edge labels inside the canvas
      ctx.fillText(txt, Math.min(Math.max(x, half + 2), W - half - 2), P.t + ph + 8);
    }
    ctx.fillText(m.xName, P.l + pw / 2, H - 16);

    // series
    ctx.save();
    ctx.beginPath(); ctx.rect(P.l - 4, P.t - 4, pw + 8, ph + 8); ctx.clip();
    ctx.lineWidth = 2; ctx.lineJoin = 'round'; ctx.lineCap = 'round';
    m.series.forEach((s, j) => {
      const color = seriesColor(j);
      ctx.strokeStyle = color;
      ctx.beginPath();
      let pen = false;
      const lone = [];
      for (let k = 0; k < m.nb; k++) {
        if (Number.isNaN(s.mn[k])) { pen = false; continue; }
        const x = xOf(m.dec ? ((k + 0.5) * n) / m.nb - 0.5 : k);
        if (!pen) {
          ctx.moveTo(x, yOf(s.mn[k]));
          if (k + 1 >= m.nb || Number.isNaN(s.mn[k + 1])) lone.push([x, s.mn[k]]);
          pen = true;
        } else ctx.lineTo(x, yOf(s.mn[k]));
        if (s.mx[k] !== s.mn[k]) ctx.lineTo(x, yOf(s.mx[k]));
      }
      ctx.stroke();
      ctx.fillStyle = color;  // isolated points get a dot so they stay visible
      for (const [x, v] of lone) { ctx.beginPath(); ctx.arc(x, yOf(v), 4, 0, Math.PI * 2); ctx.fill(); }
    });
    ctx.restore();

    // hover crosshair + markers
    if (hover != null) {
      const x = Math.round(xOf(hover)) + 0.5;
      ctx.strokeStyle = muted; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(x, P.t); ctx.lineTo(x, P.t + ph); ctx.stroke();
      const ring = cssVar('--surface');
      m.series.forEach((s, j) => {
        const v = m.get(m.X.at(hover), s.s);
        if (v === undefined || !Number.isFinite(v)) return;
        ctx.beginPath(); ctx.arc(x, yOf(v), 4, 0, Math.PI * 2);
        ctx.fillStyle = seriesColor(j); ctx.fill();
        ctx.lineWidth = 2; ctx.strokeStyle = ring; ctx.stroke();
      });
    }
    m.layout = { P, pw, n };
  }

  el.lineCanvas.addEventListener('mousemove', (e) => {
    const m = lineModel;
    if (!m || m.error || !m.layout) return;
    const { P, pw, n } = m.layout;
    const pos = n === 1 ? 0 : Math.round(Math.min(Math.max((e.offsetX - P.l) / pw, 0), 1) * (n - 1));
    drawLine(pos);
    const xv = m.X.at(pos);
    const rows = m.series.map((s, j) => `
      <div class="tt-row"><span class="lg-key" style="background:${seriesColor(j)}"></span>
      <span class="tt-label">${esc(s.label)}</span><b>${esc(m.fmt(xv, s.s))}</b></div>`).join('');
    showTip(e, `<div class="tt-head">${esc(m.xName)} = ${xv}${m.xLabels ? ` · ${esc(m.xLabels[xv])}` : ''}</div>${rows}`);
  });
  el.lineCanvas.addEventListener('mouseleave', () => { hideTip(); if (lineModel) drawLine(); });

  /* ---------- image ---------- */

  let imgModel = null;

  function cmapStops() {
    const dark = darkMQ.matches;
    switch (el.cmap.value) {
      case 'gray': return ['#000000', '#ffffff'];
      case 'diverging': return dark
        ? ['#9ec5f4', '#3987e5', '#1c5cab', '#383835', '#a83a3a', '#e66767', '#f4b0a6']
        : ['#0d366b', '#2a78d6', '#9ec5f4', '#f0efec', '#f4b0a6', '#e34948', '#8a1f1f'];
      default: {
        // one hue; low values recede toward the surface, high values stand out
        const ramp = ['#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#256abf', '#184f95', '#0d366b'];
        return dark ? ramp.slice().reverse() : ramp;
      }
    }
  }

  function buildLut(stops) {
    const rgb = stops.map((h) => [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16)));
    const lut = new Uint8Array(256 * 3);
    for (let i = 0; i < 256; i++) {
      const t = (i / 255) * (rgb.length - 1), k = Math.min(rgb.length - 2, Math.floor(t)), f = t - k;
      for (let c = 0; c < 3; c++) lut[i * 3 + c] = Math.round(rgb[k][c] + (rgb[k + 1][c] - rgb[k][c]) * f);
    }
    return lut;
  }

  // Fixed axes of length 3 or 4 can be used as colour channels.
  function updateImgModes() {
    const { arr, view } = S;
    const cur = el.imgMode.value;
    const opts = ['<option value="map">Colormap (value per cell)</option>'];
    arr.dims.forEach((n, k) => {
      if (k !== view.rowDim && k !== view.colDim && k !== arr.fieldDim && (n === 3 || n === 4)) {
        opts.push(`<option value="${k}">${n === 3 ? 'RGB' : 'RGBA'} channels from ${esc(arr.dimNames[k])}</option>`);
      }
    });
    el.imgMode.innerHTML = opts.join('');
    el.imgMode.value = [...el.imgMode.options].some((o) => o.value === cur) ? cur : 'map';
  }

  function renderImage(acc) {
    const { arr, view } = S;
    imgModel = null;
    const fail = (msg) => {
      el.imgMsg.textContent = msg;
      el.imgMsg.classList.remove('hidden');
      el.imgCanvas.classList.add('hidden');
      el.colorbar.classList.add('hidden');
      el.imgNote.textContent = '';
    };
    if (!acc.num) return fail('Images need numeric data.');
    const { R, C } = acc;
    if (!R.count || !C.count) return fail('The selection is empty.');
    el.imgMsg.classList.add('hidden');
    el.imgCanvas.classList.remove('hidden');

    const rgbK = el.imgMode.value === 'map' ? -1 : +el.imgMode.value;
    el.cmap.disabled = rgbK >= 0;
    const sy = Math.ceil(R.count / IMG_MAX), sx = Math.ceil(C.count / IMG_MAX);
    const h = Math.ceil(R.count / sy), w = Math.ceil(C.count / sx);
    const nch = rgbK >= 0 ? arr.dims[rgbK] : 1;
    const offs = rgbK >= 0
      ? Array.from({ length: nch }, (_, ch) => (ch - acc.fixedIdx[rgbK]) * arr.byteStrides[rgbK])
      : [0];

    const vals = new Float32Array(w * h * nch);
    let lo = Infinity, hi = -Infinity;
    for (let y = 0; y < h; y++) {
      const r = R.at(y * sy);
      for (let x = 0; x < w; x++) {
        const c = C.at(x * sx);
        for (let ch = 0; ch < nch; ch++) {
          let v = acc.num(r, c, offs[ch]);
          if (v === undefined) v = NaN;
          vals[(y * w + x) * nch + ch] = v;
          if (Number.isFinite(v)) { if (v < lo) lo = v; if (v > hi) hi = v; }
        }
      }
    }

    // colour range: auto (selection min/max) unless overridden
    let amin = lo, amax = hi;
    if (!Number.isFinite(lo)) { amin = 0; amax = 1; }
    if (rgbK >= 0 && arr.type.kind === 'u' && arr.type.size === 1) { amin = 0; amax = 255; }
    if (rgbK < 0 && el.cmap.value === 'diverging') { const m = Math.max(Math.abs(amin), Math.abs(amax)); amin = -m; amax = m; }
    el.vmin.placeholder = `auto ${fmtTick(amin)}`;
    el.vmax.placeholder = `auto ${fmtTick(amax)}`;
    const vmin = el.vmin.value !== '' ? +el.vmin.value : amin;
    const vmax = el.vmax.value !== '' ? +el.vmax.value : amax;
    const scale = 255 / (vmax - vmin || 1);

    const img = new ImageData(w, h), px = img.data;
    if (rgbK < 0) {
      const lut = buildLut(cmapStops());
      for (let p = 0; p < w * h; p++) {
        const v = vals[p];
        if (Number.isNaN(v)) continue;  // NaN stays transparent
        let t = Math.round((v - vmin) * scale);
        t = t < 0 ? 0 : t > 255 ? 255 : t;
        px[p * 4] = lut[t * 3]; px[p * 4 + 1] = lut[t * 3 + 1]; px[p * 4 + 2] = lut[t * 3 + 2]; px[p * 4 + 3] = 255;
      }
    } else {
      for (let p = 0; p < w * h; p++) {
        for (let ch = 0; ch < 3; ch++) px[p * 4 + ch] = (vals[p * nch + ch] - vmin) * scale;
        px[p * 4 + 3] = nch === 4 ? (vals[p * nch + 3] - vmin) * scale : 255;
      }
    }
    el.imgCanvas.width = w;
    el.imgCanvas.height = h;
    el.imgCanvas.getContext('2d').putImageData(img, 0, 0);
    imgModel = { acc, w, h, sx, sy, rgbK, offs };
    sizeImage();

    el.colorbar.classList.toggle('hidden', rgbK >= 0);
    if (rgbK < 0) {
      el.cbRamp.style.background = `linear-gradient(to right, ${cmapStops().join(', ')})`;
      el.cbMin.textContent = fmtTick(vmin);
      el.cbMid.textContent = fmtTick((vmin + vmax) / 2);
      el.cbMax.textContent = fmtTick(vmax);
    }
    const name = (d, none) => d >= 0 ? arr.dimNames[d] : none;
    el.imgNote.textContent = `y ↓ ${name(view.rowDim, 'single row')} · x → ${name(view.colDim, 'single column')}`
      + (sx > 1 || sy > 1 ? ` · ${R.count.toLocaleString()} × ${C.count.toLocaleString()} sampled to ${h} × ${w} for display` : '')
      + (rgbK < 0 ? ' · NaN cells are transparent' : '');
  }

  function sizeImage() {
    if (!imgModel) return;
    const { w, h } = imgModel;
    const availW = el.imgBox.clientWidth - 2;
    const maxH = Math.max(320, window.innerHeight * 0.7);
    // auto: square pixels unless the selection is very elongated (e.g. 20 x 2000)
    const ratio = w / h;
    const square = el.aspect.value === 'square' || (el.aspect.value === 'auto' && ratio <= 4 && ratio >= 0.25);
    let dw, dh;
    if (square) { const s = Math.min(availW / w, maxH / h); dw = w * s; dh = h * s; }
    else { dw = availW; dh = Math.min(maxH, 480); }
    el.imgCanvas.style.width = `${Math.max(1, Math.floor(dw))}px`;
    el.imgCanvas.style.height = `${Math.max(1, Math.floor(dh))}px`;
  }

  el.imgCanvas.addEventListener('mousemove', (e) => {
    const m = imgModel;
    if (!m) return;
    const rect = el.imgCanvas.getBoundingClientRect();
    const x = Math.min(m.w - 1, Math.max(0, Math.floor(((e.clientX - rect.left) / rect.width) * m.w)));
    const y = Math.min(m.h - 1, Math.max(0, Math.floor(((e.clientY - rect.top) / rect.height) * m.h)));
    const r = m.acc.R.at(y * m.sy), c = m.acc.C.at(x * m.sx);
    const idx = m.acc.index(r, c);
    let val;
    if (m.rgbK < 0) val = m.acc.cell(r, c);
    else { idx[m.rgbK] = ':'; val = m.offs.map((o) => fmtStat(m.acc.num(r, c, o))).join(', '); }
    const lbl = S.arr.axisLabels ? [m.acc.rowLabel(r), m.acc.colLabel(c)].filter((x) => x && x !== 'value').join(' · ') : '';
    showTip(e, `<div class="tt-head">${esc(S.arr.name + fmtIdx(idx))}</div>${lbl ? `<div class="tt-label">${esc(lbl)}</div>` : ''}<div><b>${esc(val)}</b></div>`);
  });
  el.imgCanvas.addEventListener('mouseleave', hideTip);

  /* ---------- CSV export ---------- */

  function csvField(s) {
    return /[",\r\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
  }

  function exportCsv() {
    const { arr, view } = S;
    if (!arr) return;
    const acc = makeAccessor(arr, view, decimals());
    const { R, C } = acc;
    const labels = el.includeLabels.checked;
    const chunks = [];
    let lines = [];
    const flush = () => { if (lines.length) { chunks.push(lines.join('\r\n') + '\r\n'); lines = []; } };

    if (labels) {
      const head = [acc.corner];
      for (let j = 0; j < C.count; j++) head.push(acc.colLabel(C.at(j)));
      lines.push(head.map(csvField).join(','));
    }
    const row = new Array(C.count + (labels ? 1 : 0));
    for (let i = 0; i < R.count; i++) {
      const r = R.at(i);
      let p = 0;
      if (labels) row[p++] = csvField(acc.rowLabel(r));
      for (let j = 0; j < C.count; j++) row[p++] = csvField(acc.cell(r, C.at(j)));
      lines.push(row.join(','));
      if (lines.length >= 2000) flush();
    }
    flush();

    const base = S.fileName.replace(/\.(npz|npy)$/i, '');
    const sliceTag = sliceExpression(arr, view, acc).slice(arr.name.length).replace(/[^\w.-]+/g, '_').replace(/^_+|_+$/g, '');
    const name = (S.arrays.length > 1 || base !== arr.name ? `${base}_${arr.name}` : base) + (sliceTag ? `_${sliceTag}` : '');
    const url = URL.createObjectURL(new Blob(chunks, { type: 'text/csv;charset=utf-8' }));
    const a = document.createElement('a');
    a.href = url;
    a.download = `${name.replace(/[\\/:*?"<>|]+/g, '_')}.csv`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 5000);
  }

  /* ---------- events ---------- */

  el.fileInput.addEventListener('change', () => {
    const f = el.fileInput.files[0];
    if (f) loadFile(f);
    el.fileInput.value = '';
  });

  el.arrayList.addEventListener('click', (e) => {
    const li = e.target.closest('li');
    if (li) selectArray(+li.dataset.i);
  });

  el.dimTable.addEventListener('change', (e) => {
    const t = e.target;
    if (t.dataset.f === 'role') setRole(+t.dataset.k, t.value);
  });

  el.dimTable.addEventListener('input', (e) => {
    const t = e.target, k = +t.dataset.k, f = t.dataset.f;
    if (!f || f === 'role') return;
    const v = S.view;
    if (f === 'index' || f === 'slider') {
      v.fixed[k] = t.value === '' ? 0 : +t.value;
      const tr = t.closest('tr');
      if (f === 'index') tr.querySelector('[data-f="slider"]').value = normIndex(v.fixed[k], S.arr.dims[k]);
      else tr.querySelector('[data-f="index"]').value = t.value;
      if (k === S.arr.fieldDim) {
        const lbl = tr.querySelector('.muted');
        if (lbl) lbl.textContent = S.arr.type.fields[normIndex(v.fixed[k], S.arr.dims[k])].name;
      }
      if (f === 'slider') { render(); return; }
    } else {
      v.ranges[k][f] = t.value;
      S.page = 0;
    }
    scheduleRender();
  });

  el.swapBtn.addEventListener('click', () => {
    const v = S.view;
    if (!v) return;
    [v.rowDim, v.colDim] = [v.colDim, v.rowDim];
    S.page = 0;
    buildDimTable();
    render();
  });

  el.tabs.addEventListener('click', (e) => {
    const b = e.target.closest('[data-tab]');
    if (!b) return;
    S.tab = b.dataset.tab;
    for (const x of el.tabs.children) {
      x.classList.toggle('active', x === b);
      x.setAttribute('aria-selected', String(x === b));
    }
    el.tablePane.classList.toggle('hidden', S.tab !== 'table');
    el.linePane.classList.toggle('hidden', S.tab !== 'line');
    el.imagePane.classList.toggle('hidden', S.tab !== 'image');
    hideTip();
    renderView();
  });

  el.decimals.addEventListener('input', scheduleRender);
  el.pageSize.addEventListener('change', () => { S.page = 0; renderView(); });
  el.exportBtn.addEventListener('click', exportCsv);
  el.firstBtn.addEventListener('click', () => { S.page = 0; renderView(); });
  el.prevBtn.addEventListener('click', () => { S.page--; renderView(); });
  el.nextBtn.addEventListener('click', () => { S.page++; renderView(); });
  el.lastBtn.addEventListener('click', () => { S.page = Infinity; renderView(); });

  let imgTimer = 0;
  const imgRefresh = () => { clearTimeout(imgTimer); imgTimer = setTimeout(() => S.acc && renderImage(S.acc), 150); };
  el.imgMode.addEventListener('change', imgRefresh);
  el.cmap.addEventListener('change', imgRefresh);
  el.vmin.addEventListener('input', imgRefresh);
  el.vmax.addEventListener('input', imgRefresh);
  el.aspect.addEventListener('change', sizeImage);

  // redraw plots on resize / theme change
  new ResizeObserver(() => {
    if (S.tab === 'line' && lineModel) drawLine();
    else if (S.tab === 'image') sizeImage();
  }).observe(el.viewer);
  darkMQ.addEventListener('change', () => { if (S.tab !== 'table') renderView(); });

  // drag & drop anywhere on the page
  let dragDepth = 0;
  window.addEventListener('dragenter', (e) => { e.preventDefault(); dragDepth++; document.body.classList.add('dragging'); });
  window.addEventListener('dragleave', () => { if (--dragDepth <= 0) { dragDepth = 0; document.body.classList.remove('dragging'); } });
  window.addEventListener('dragover', (e) => e.preventDefault());
  window.addEventListener('drop', (e) => {
    e.preventDefault();
    dragDepth = 0;
    document.body.classList.remove('dragging');
    const f = e.dataTransfer.files[0];
    if (f) loadFile(f);
  });
}
