'use strict';

/* =====================================================================
 * HDF5 support (h5.html) — built on h5wasm, the HDF5 C library compiled
 * to WebAssembly. Datasets are turned into the same array objects the
 * .npy/.npz viewer uses, so table, stats, plots and CSV export all apply.
 * ===================================================================== */

(function () {
  // HDF5 datatype classes (H5T_class_t)
  const H5T = { INTEGER: 0, FLOAT: 1, STRING: 3, COMPOUND: 6, REFERENCE: 7, ENUM: 8, VLEN: 9, ARRAY: 10 };

  let openFile = null, openPath = null;

  const shapeStr = (shape) => `(${shape.join(', ')}${shape.length === 1 ? ',' : ''})`;
  const clip = (s, n = 300) => s.length > n ? s.slice(0, n) + '…' : s;

  function readAttrs(obj) {
    const out = [];
    let attrs;
    try { attrs = obj.attrs; } catch { return out; }
    for (const k of Object.keys(attrs)) {
      try { out.push([k, clip(fmtJs(attrs[k].value, null))]); } catch (e) { out.push([k, `<unreadable: ${e.message}>`]); }
    }
    return out;
  }

  function attrValue(obj, name) {
    try { const a = obj.attrs[name]; return a ? a.value : undefined; } catch { return undefined; }
  }

  // numpy-style description of an HDF5 datatype
  function describe(md) {
    const bo = md.littleEndian ? '<' : '>';
    switch (md.type) {
      case H5T.INTEGER: return `${bo}${md.signed ? 'i' : 'u'}${md.size}`;
      case H5T.FLOAT: return `${bo}f${md.size}`;
      case H5T.STRING: return md.vlen ? 'str (variable length)' : `|S${md.size}`;
      case H5T.COMPOUND: return `compound {${md.compound_type.members.map((m) => `${m.name}: ${describe(m)}`).join(', ')}}`;
      case H5T.ENUM: {
        const names = Object.keys(md.enum_type.members);
        return names.length === 2 && md.enum_type.members.FALSE === 0 && md.enum_type.members.TRUE === 1
          ? 'bool' : `enum {${names.join(', ')}}`;
      }
      case H5T.ARRAY: return `array ${describe(md.array_type)} ${JSON.stringify(md.array_type.shape)}`;
      case H5T.VLEN: return 'vlen sequence';
      case H5T.REFERENCE: return 'reference';
      default: return `HDF5 type class ${md.type}`;
    }
  }

  function readDataset(ds) {
    const md = ds.metadata;
    const shape = md.shape.map(Number);
    const size = shape.reduce((a, b) => a * b, 1);

    // Integers and floats: keep the raw typed array (fast, no copies).
    if ((md.type === H5T.INTEGER || md.type === H5T.FLOAT) && shape.length) {
      const value = ds.value;
      if (ArrayBuffer.isView(value)) {
        const kind = md.type === H5T.FLOAT ? 'f' : md.signed ? 'i' : 'u';
        // pandas/PyTables tag datetime columns with kind='datetime64' (int64 nanoseconds)
        const isDate = kind === 'i' && md.size === 8 && String(attrValue(ds, 'kind') ?? '') === 'datetime64';
        const type = parseDtype(`${md.littleEndian ? '<' : '>'}${isDate ? 'M8[ns]' : kind + md.size}`);
        return buildArray({ shape, type, dv: new DataView(value.buffer, value.byteOffset, value.byteLength) });
      }
    }

    // Everything else (strings, bools, compound rows, scalars, …): a plain JS value per element.
    const v = md.type === H5T.ENUM ? ds.json_value : ds.value;
    const values = shape.length === 0 ? [v]
      : Array.isArray(v) || ArrayBuffer.isView(v) ? Array.from(v) : [v];
    if (values.length !== size) throw new Error(`Cannot display datatype ${describe(md)} element by element`);
    const first = values.find((x) => x != null);
    const numeric = ['number', 'bigint', 'boolean'].includes(typeof first);
    const type = {
      kind: 'js', size: 1, little: true, str: describe(md), values, bytes: md.size * size,
      num: numeric ? (dv, o) => {
        const x = values[o];
        return typeof x === 'number' ? x : typeof x === 'bigint' ? Number(x) : typeof x === 'boolean' ? +x : undefined;
      } : null,
    };
    return buildArray({ shape, type, dv: new DataView(new ArrayBuffer(0)) });
  }

  // pandas HDFStore ("fixed" format) index → readable labels
  function indexLabels(ds) {
    const kind = String(attrValue(ds, 'kind') ?? '');
    const values = ds.value;
    if (kind === 'datetime64') {
      return Array.from(values, (x) => fmtDatetime(BigInt(x), 'ns').replace('T', ' ').replace(/\.0+$/, ''));
    }
    return Array.from(values, (x) => fmtJs(x, null));
  }

  function applyPandas(arr, p) {
    if (arr.shape.length !== 2) return;
    const index = indexLabels(openFile.get(p.index));
    const items = Array.from(openFile.get(p.items).value, (x) => fmtJs(x, null));
    const [n0, n1] = arr.shape;
    if (n0 === index.length && n1 === items.length) {
      arr.axisLabels = [index, items];
      arr.dimNames = ['index', 'columns'];
    } else if (n1 === index.length && n0 === items.length) {
      arr.axisLabels = [items, index];
      arr.dimNames = ['columns', 'index'];
    } else return;
    const kind = String(attrValue(openFile.get(p.index), 'kind') ?? '');
    arr.note = `pandas DataFrame block (${p.group}): ${index.length.toLocaleString()} rows × ${items.length.toLocaleString()} columns. `
      + `Rows are labelled by the frame's index${kind ? ` (${kind})` : ''}, columns by its column names; labels are used in the table, plots and CSV.`;
  }

  window.loadHdf5 = async function (u8) {
    if (typeof h5wasm === 'undefined') {
      throw new Error('The HDF5 reader (h5wasm) did not load. It is fetched from cdn.jsdelivr.net — check your internet connection and reload the page.');
    }
    const Module = await h5wasm.ready;
    if (openFile) {
      try { openFile.close(); Module.FS.unlink(openPath); } catch { /* already gone */ }
    }
    openPath = `/file_${Date.now()}.h5`;
    Module.FS.writeFile(openPath, u8);
    const f = openFile = new h5wasm.File(openPath, 'r');
    const entries = [];

    function datasetEntry(ds, label, path, depth) {
      const base = { name: path.slice(1), label, path, depth };
      let md;
      try { md = ds.metadata; } catch (e) { return { ...base, error: e.message }; }
      if (!md.shape) return { ...base, error: 'empty dataset (null dataspace)' };
      const shape = md.shape.map(Number);
      const entry = {
        ...base, shape, size: shape.reduce((a, b) => a * b, 1),
        meta: `${shapeStr(shape)} ${describe(md)}`,
        lazy: async () => {
          const arr = readDataset(ds);
          arr.attrs = readAttrs(ds);
          arr.extraInfo = [['path', path]];
          try {
            const filters = ds.filters.map((x) => x.name || `filter ${x.id}`);
            if (filters.length) arr.extraInfo.push(['filters', filters.join(', ')]);
          } catch { /* no filter info */ }
          if (md.chunks) arr.extraInfo.push(['chunks', shapeStr(md.chunks.map(Number))]);
          if (entry.pandas) applyPandas(arr, entry.pandas);
          return arr;
        },
      };
      return entry;
    }

    (function walk(g, depth) {
      let keys;
      try { keys = g.keys(); } catch { return; }
      for (const name of keys) {
        const path = g.path === '/' ? `/${name}` : `${g.path}/${name}`;
        let obj;
        try { obj = g.get(name); } catch (e) {
          entries.push({ name: path.slice(1), label: name, path, depth, error: e.message });
          continue;
        }
        if (!obj) continue;
        if (obj.type === 'Group') {
          const attrs = readAttrs(obj);
          const pandas = attrValue(obj, 'pandas_type');
          entries.push({
            group: true, name: path, label: name, path, depth,
            meta: pandas ? `pandas ${pandas === 'frame' ? 'DataFrame' : pandas}` : '',
            tip: [path, ...attrs.map(([k, v]) => `${k} = ${v}`)].join('\n'),
            pandasType: pandas,
          });
          walk(obj, depth + 1);
        } else if (obj.type === 'Dataset') {
          entries.push(datasetEntry(obj, name, path, depth));
        } else {
          entries.push({ name: path.slice(1), label: name, path, depth, error: `${obj.type || 'link'} — not a dataset` });
        }
      }
    })(f, 0);

    // pandas HDFStore frames: label rows with the index and columns with the column names
    for (const g of entries.filter((e) => e.group && e.pandasType === 'frame')) {
      const nblocks = Number(attrValue(f.get(g.path), 'nblocks') ?? 0);
      for (let k = 0; k < nblocks; k++) {
        const e = entries.find((x) => x.path === `${g.path}/block${k}_values`);
        if (e && !e.error) {
          e.pandas = { group: g.path, index: `${g.path}/axis1`, items: `${g.path}/block${k}_items` };
          e.meta += ' · DataFrame values';
        }
      }
    }

    const datasets = entries.filter((e) => !e.group && !e.error);
    if (!datasets.length) throw new Error('This HDF5 file contains no readable datasets.');
    // open the biggest dataset first — usually the main data
    datasets.reduce((a, b) => (b.size > a.size ? b : a)).preferred = true;
    return entries;
  };
})();
