//! Pure-Rust ports of the HDF5 N-bit (filter id 5) and Scale-offset
//! (filter id 6) filters.
//!
//! Both ports are byte-exact with libhdf5's `H5Znbit.c` and
//! `H5Zscaleoffset.c`. The bit-packing helpers mirror the C routines
//! line-for-line so that crate-decoded chunks match libhdf5 element-exact.

use crate::format::{FormatError, FormatResult};

// ===========================================================================
//  N-bit filter (H5Z_FILTER_NBIT, id 5)
// ===========================================================================

// Datatype class codes used in the nbit parameter tree.
const NBIT_ATOMIC: u32 = 1;
const NBIT_ARRAY: u32 = 2;
const NBIT_COMPOUND: u32 = 3;
const NBIT_NOOPTYPE: u32 = 4;
const NBIT_ORDER_LE: u32 = 0;
const NBIT_ORDER_BE: u32 = 1;

/// Parameters describing one atomic element for the nbit packer.
#[derive(Clone, Copy)]
struct NbitAtomic {
    size: u32,
    order: u32,
    precision: u32,
    offset: u32,
}

/// A bit cursor over a packed nbit buffer (`j` = byte index,
/// `buf_len` = remaining unread bits in the current byte).
struct NbitCursor {
    j: usize,
    buf_len: usize,
}

impl NbitCursor {
    fn next_byte(&mut self) {
        self.j += 1;
        self.buf_len = 8;
    }
}

/// `~((unsigned)(~0) << n)` over the low 32 bits.
fn mask_u32(n: usize) -> u32 {
    if n >= 32 {
        u32::MAX
    } else {
        !(u32::MAX << n)
    }
}

/// Decompress one atomic byte, mirroring `H5Z__nbit_decompress_one_byte`.
#[allow(clippy::too_many_arguments)]
fn nbit_decompress_one_byte(
    data: &mut [u8],
    data_offset: usize,
    k: u32,
    begin_i: u32,
    end_i: u32,
    buffer: &[u8],
    cur: &mut NbitCursor,
    p: &NbitAtomic,
    datatype_len: u32,
) -> FormatResult<()> {
    if cur.j >= buffer.len() {
        return Err(FormatError::InvalidData("nbit: buffer too short".into()));
    }
    let mut val = buffer[cur.j];
    let mut dat_offset: usize = 0;
    let mut dat_len: usize;

    if begin_i != end_i {
        if k == begin_i {
            dat_len = 8 - ((datatype_len - p.precision - p.offset) % 8) as usize;
        } else if k == end_i {
            dat_len = 8 - (p.offset % 8) as usize;
            dat_offset = 8 - dat_len;
        } else {
            dat_len = 8;
        }
    } else {
        dat_offset = (p.offset % 8) as usize;
        dat_len = p.precision as usize;
    }

    let idx = data_offset + k as usize;
    if cur.buf_len > dat_len {
        data[idx] =
            (((val >> (cur.buf_len - dat_len)) as u32 & mask_u32(dat_len)) << dat_offset) as u8;
        cur.buf_len -= dat_len;
    } else {
        data[idx] =
            (((val as u32 & mask_u32(cur.buf_len)) << (dat_len - cur.buf_len)) << dat_offset) as u8;
        dat_len -= cur.buf_len;
        cur.next_byte();
        if dat_len == 0 {
            return Ok(());
        }
        if cur.j >= buffer.len() {
            return Err(FormatError::InvalidData("nbit: buffer too short".into()));
        }
        val = buffer[cur.j];
        data[idx] |=
            (((val >> (cur.buf_len - dat_len)) as u32 & mask_u32(dat_len)) << dat_offset) as u8;
        cur.buf_len -= dat_len;
    }
    Ok(())
}

/// Compress one atomic byte, mirroring `H5Z__nbit_compress_one_byte`.
#[allow(clippy::too_many_arguments)]
fn nbit_compress_one_byte(
    data: &[u8],
    data_offset: usize,
    k: u32,
    begin_i: u32,
    end_i: u32,
    buffer: &mut [u8],
    cur: &mut NbitCursor,
    p: &NbitAtomic,
    datatype_len: u32,
) {
    let mut val = data[data_offset + k as usize];
    let mut dat_len: usize;

    if begin_i != end_i {
        if k == begin_i {
            dat_len = 8 - ((datatype_len - p.precision - p.offset) % 8) as usize;
        } else if k == end_i {
            dat_len = 8 - (p.offset % 8) as usize;
            val >>= 8 - dat_len;
        } else {
            dat_len = 8;
        }
    } else {
        val >>= p.offset % 8;
        dat_len = p.precision as usize;
    }

    if cur.buf_len > dat_len {
        buffer[cur.j] |= ((val as u32 & mask_u32(dat_len)) << (cur.buf_len - dat_len)) as u8;
        cur.buf_len -= dat_len;
    } else {
        buffer[cur.j] |= ((val as u32 >> (dat_len - cur.buf_len)) & mask_u32(cur.buf_len)) as u8;
        dat_len -= cur.buf_len;
        cur.next_byte();
        if dat_len == 0 {
            return;
        }
        buffer[cur.j] = ((val as u32 & mask_u32(dat_len)) << (cur.buf_len - dat_len)) as u8;
        cur.buf_len -= dat_len;
    }
}

/// Decompress one nooptype element, mirroring `H5Z__nbit_decompress_one_nooptype`.
fn nbit_decompress_one_nooptype(
    data: &mut [u8],
    data_offset: usize,
    buffer: &[u8],
    cur: &mut NbitCursor,
    size: u32,
) -> FormatResult<()> {
    for i in 0..size as usize {
        if cur.j >= buffer.len() {
            return Err(FormatError::InvalidData("nbit: buffer too short".into()));
        }
        let mut val = buffer[cur.j];
        let mut dat_len: usize = 8;
        data[data_offset + i] =
            ((val as u32 & mask_u32(cur.buf_len)) << (dat_len - cur.buf_len)) as u8;
        dat_len -= cur.buf_len;
        cur.next_byte();
        if dat_len == 0 {
            continue;
        }
        if cur.j >= buffer.len() {
            return Err(FormatError::InvalidData("nbit: buffer too short".into()));
        }
        val = buffer[cur.j];
        data[data_offset + i] |=
            ((val >> (cur.buf_len - dat_len)) as u32 & mask_u32(dat_len)) as u8;
        cur.buf_len -= dat_len;
    }
    Ok(())
}

/// Compress one nooptype element, mirroring `H5Z__nbit_compress_one_nooptype`.
fn nbit_compress_one_nooptype(
    data: &[u8],
    data_offset: usize,
    buffer: &mut [u8],
    cur: &mut NbitCursor,
    size: u32,
) {
    for i in 0..size as usize {
        let val = data[data_offset + i];
        let mut dat_len: usize = 8;
        buffer[cur.j] |= ((val as u32 >> (dat_len - cur.buf_len)) & mask_u32(cur.buf_len)) as u8;
        dat_len -= cur.buf_len;
        cur.next_byte();
        if dat_len == 0 {
            continue;
        }
        buffer[cur.j] = ((val as u32 & mask_u32(dat_len)) << (cur.buf_len - dat_len)) as u8;
        cur.buf_len -= dat_len;
    }
}

/// Decompress one atomic element, mirroring `H5Z__nbit_decompress_one_atomic`.
fn nbit_decompress_one_atomic(
    data: &mut [u8],
    data_offset: usize,
    buffer: &[u8],
    cur: &mut NbitCursor,
    p: &NbitAtomic,
) -> FormatResult<()> {
    let datatype_len = p.size * 8;
    if p.order == NBIT_ORDER_LE {
        let begin_i = if (p.precision + p.offset) % 8 != 0 {
            (p.precision + p.offset) / 8
        } else {
            (p.precision + p.offset) / 8 - 1
        };
        let end_i = p.offset / 8;
        let mut k = begin_i as i64;
        while k >= end_i as i64 {
            nbit_decompress_one_byte(
                data,
                data_offset,
                k as u32,
                begin_i,
                end_i,
                buffer,
                cur,
                p,
                datatype_len,
            )?;
            k -= 1;
        }
    } else {
        let begin_i = (datatype_len - p.precision - p.offset) / 8;
        let end_i = if p.offset % 8 != 0 {
            (datatype_len - p.offset) / 8
        } else {
            (datatype_len - p.offset) / 8 - 1
        };
        for k in begin_i..=end_i {
            nbit_decompress_one_byte(
                data,
                data_offset,
                k,
                begin_i,
                end_i,
                buffer,
                cur,
                p,
                datatype_len,
            )?;
        }
    }
    Ok(())
}

/// Compress one atomic element, mirroring `H5Z__nbit_compress_one_atomic`.
fn nbit_compress_one_atomic(
    data: &[u8],
    data_offset: usize,
    buffer: &mut [u8],
    cur: &mut NbitCursor,
    p: &NbitAtomic,
) {
    let datatype_len = p.size * 8;
    if p.order == NBIT_ORDER_LE {
        let begin_i = if (p.precision + p.offset) % 8 != 0 {
            (p.precision + p.offset) / 8
        } else {
            (p.precision + p.offset) / 8 - 1
        };
        let end_i = p.offset / 8;
        let mut k = begin_i as i64;
        while k >= end_i as i64 {
            nbit_compress_one_byte(
                data,
                data_offset,
                k as u32,
                begin_i,
                end_i,
                buffer,
                cur,
                p,
                datatype_len,
            );
            k -= 1;
        }
    } else {
        let begin_i = (datatype_len - p.precision - p.offset) / 8;
        let end_i = if p.offset % 8 != 0 {
            (datatype_len - p.offset) / 8
        } else {
            (datatype_len - p.offset) / 8 - 1
        };
        for k in begin_i..=end_i {
            nbit_compress_one_byte(
                data,
                data_offset,
                k,
                begin_i,
                end_i,
                buffer,
                cur,
                p,
                datatype_len,
            );
        }
    }
}

/// Read an atomic parameter group starting at `parms[idx]` (after the class
/// code has already been consumed): `size, order, precision, offset`.
fn read_atomic(parms: &[u32], idx: &mut usize) -> FormatResult<NbitAtomic> {
    if *idx + 4 > parms.len() {
        return Err(FormatError::InvalidData(
            "nbit: parameter list truncated".into(),
        ));
    }
    let p = NbitAtomic {
        size: parms[*idx],
        order: parms[*idx + 1],
        precision: parms[*idx + 2],
        offset: parms[*idx + 3],
    };
    *idx += 4;
    Ok(p)
}

/// Decompress one array element, mirroring `H5Z__nbit_decompress_one_array`.
fn nbit_decompress_one_array(
    data: &mut [u8],
    data_offset: usize,
    buffer: &[u8],
    cur: &mut NbitCursor,
    parms: &[u32],
    parms_index: &mut usize,
) -> FormatResult<()> {
    if *parms_index + 2 > parms.len() {
        return Err(FormatError::InvalidData(
            "nbit: parameter list truncated".into(),
        ));
    }
    let total_size = parms[*parms_index];
    let base_class = parms[*parms_index + 1];
    *parms_index += 2;

    match base_class {
        NBIT_ATOMIC => {
            let p = read_atomic(parms, parms_index)?;
            let n = total_size / p.size;
            for i in 0..n as usize {
                nbit_decompress_one_atomic(
                    data,
                    data_offset + i * p.size as usize,
                    buffer,
                    cur,
                    &p,
                )?;
            }
        }
        NBIT_ARRAY => {
            let base_size = parms[*parms_index];
            let n = total_size / base_size;
            let begin_index = *parms_index;
            for i in 0..n as usize {
                *parms_index = begin_index;
                nbit_decompress_one_array(
                    data,
                    data_offset + i * base_size as usize,
                    buffer,
                    cur,
                    parms,
                    parms_index,
                )?;
            }
        }
        NBIT_COMPOUND => {
            let base_size = parms[*parms_index];
            let n = total_size / base_size;
            let begin_index = *parms_index;
            for i in 0..n as usize {
                *parms_index = begin_index;
                nbit_decompress_one_compound(
                    data,
                    data_offset + i * base_size as usize,
                    buffer,
                    cur,
                    parms,
                    parms_index,
                )?;
            }
        }
        NBIT_NOOPTYPE => {
            *parms_index += 1; // skip size of no-op type
            nbit_decompress_one_nooptype(data, data_offset, buffer, cur, total_size)?;
        }
        _ => {
            return Err(FormatError::InvalidData(format!(
                "nbit: bad base class {}",
                base_class
            )))
        }
    }
    Ok(())
}

/// Decompress one compound element, mirroring `H5Z__nbit_decompress_one_compound`.
fn nbit_decompress_one_compound(
    data: &mut [u8],
    data_offset: usize,
    buffer: &[u8],
    cur: &mut NbitCursor,
    parms: &[u32],
    parms_index: &mut usize,
) -> FormatResult<()> {
    if *parms_index + 2 > parms.len() {
        return Err(FormatError::InvalidData(
            "nbit: parameter list truncated".into(),
        ));
    }
    *parms_index += 1; // skip compound size
    let nmembers = parms[*parms_index];
    *parms_index += 1;

    for _ in 0..nmembers {
        if *parms_index + 2 > parms.len() {
            return Err(FormatError::InvalidData(
                "nbit: parameter list truncated".into(),
            ));
        }
        let member_offset = parms[*parms_index] as usize;
        let member_class = parms[*parms_index + 1];
        *parms_index += 2;

        match member_class {
            NBIT_ATOMIC => {
                let p = read_atomic(parms, parms_index)?;
                nbit_decompress_one_atomic(data, data_offset + member_offset, buffer, cur, &p)?;
            }
            NBIT_ARRAY => {
                nbit_decompress_one_array(
                    data,
                    data_offset + member_offset,
                    buffer,
                    cur,
                    parms,
                    parms_index,
                )?;
            }
            NBIT_COMPOUND => {
                nbit_decompress_one_compound(
                    data,
                    data_offset + member_offset,
                    buffer,
                    cur,
                    parms,
                    parms_index,
                )?;
            }
            NBIT_NOOPTYPE => {
                let size = parms[*parms_index];
                *parms_index += 1;
                nbit_decompress_one_nooptype(data, data_offset + member_offset, buffer, cur, size)?;
            }
            _ => {
                return Err(FormatError::InvalidData(format!(
                    "nbit: bad member class {}",
                    member_class
                )))
            }
        }
    }
    Ok(())
}

/// Compress one array element, mirroring `H5Z__nbit_compress_one_array`.
fn nbit_compress_one_array(
    data: &[u8],
    data_offset: usize,
    buffer: &mut [u8],
    cur: &mut NbitCursor,
    parms: &[u32],
    parms_index: &mut usize,
) -> FormatResult<()> {
    if *parms_index + 2 > parms.len() {
        return Err(FormatError::InvalidData(
            "nbit: parameter list truncated".into(),
        ));
    }
    let total_size = parms[*parms_index];
    let base_class = parms[*parms_index + 1];
    *parms_index += 2;

    match base_class {
        NBIT_ATOMIC => {
            let p = read_atomic(parms, parms_index)?;
            let n = total_size / p.size;
            for i in 0..n as usize {
                nbit_compress_one_atomic(data, data_offset + i * p.size as usize, buffer, cur, &p);
            }
        }
        NBIT_ARRAY => {
            let base_size = parms[*parms_index];
            let n = total_size / base_size;
            let begin_index = *parms_index;
            for i in 0..n as usize {
                *parms_index = begin_index;
                nbit_compress_one_array(
                    data,
                    data_offset + i * base_size as usize,
                    buffer,
                    cur,
                    parms,
                    parms_index,
                )?;
            }
        }
        NBIT_COMPOUND => {
            let base_size = parms[*parms_index];
            let n = total_size / base_size;
            let begin_index = *parms_index;
            for i in 0..n as usize {
                *parms_index = begin_index;
                nbit_compress_one_compound(
                    data,
                    data_offset + i * base_size as usize,
                    buffer,
                    cur,
                    parms,
                    parms_index,
                )?;
            }
        }
        NBIT_NOOPTYPE => {
            *parms_index += 1;
            nbit_compress_one_nooptype(data, data_offset, buffer, cur, total_size);
        }
        _ => {
            return Err(FormatError::InvalidData(format!(
                "nbit: bad base class {}",
                base_class
            )))
        }
    }
    Ok(())
}

/// Compress one compound element, mirroring `H5Z__nbit_compress_one_compound`.
fn nbit_compress_one_compound(
    data: &[u8],
    data_offset: usize,
    buffer: &mut [u8],
    cur: &mut NbitCursor,
    parms: &[u32],
    parms_index: &mut usize,
) -> FormatResult<()> {
    if *parms_index + 2 > parms.len() {
        return Err(FormatError::InvalidData(
            "nbit: parameter list truncated".into(),
        ));
    }
    *parms_index += 1;
    let nmembers = parms[*parms_index];
    *parms_index += 1;

    for _ in 0..nmembers {
        if *parms_index + 2 > parms.len() {
            return Err(FormatError::InvalidData(
                "nbit: parameter list truncated".into(),
            ));
        }
        let member_offset = parms[*parms_index] as usize;
        let member_class = parms[*parms_index + 1];
        *parms_index += 2;

        match member_class {
            NBIT_ATOMIC => {
                let p = read_atomic(parms, parms_index)?;
                nbit_compress_one_atomic(data, data_offset + member_offset, buffer, cur, &p);
            }
            NBIT_ARRAY => {
                nbit_compress_one_array(
                    data,
                    data_offset + member_offset,
                    buffer,
                    cur,
                    parms,
                    parms_index,
                )?;
            }
            NBIT_COMPOUND => {
                nbit_compress_one_compound(
                    data,
                    data_offset + member_offset,
                    buffer,
                    cur,
                    parms,
                    parms_index,
                )?;
            }
            NBIT_NOOPTYPE => {
                let size = parms[*parms_index];
                *parms_index += 1;
                nbit_compress_one_nooptype(data, data_offset + member_offset, buffer, cur, size);
            }
            _ => {
                return Err(FormatError::InvalidData(format!(
                    "nbit: bad member class {}",
                    member_class
                )))
            }
        }
    }
    Ok(())
}

/// Apply the HDF5 N-bit filter.
///
/// `cd_values` follows `H5Znbit.c`'s schema:
/// `[0]` = number of parameters, `[1]` = need-not-compress flag,
/// `[2]` = element count, `[3..]` = the datatype parameter tree.
///
/// On compress, `data` is the raw element buffer; on decompress, `data`
/// is the packed buffer and the result is the unpacked element buffer.
pub fn apply_nbit(data: &[u8], cd_values: &[u32], compress: bool) -> FormatResult<Vec<u8>> {
    if cd_values.len() < 4 {
        return Err(FormatError::InvalidData("nbit: cd_values too short".into()));
    }
    // cd_values[1] != 0 -> data is full-precision, filter is a pass-through.
    if cd_values[1] != 0 {
        return Ok(data.to_vec());
    }

    let d_nelmts = cd_values[2] as usize;
    let dtype_size = cd_values[4] as usize;
    if dtype_size == 0 {
        return Err(FormatError::InvalidData("nbit: zero datatype size".into()));
    }
    let unpacked_size = d_nelmts * dtype_size;

    if compress {
        if data.len() != unpacked_size {
            return Err(FormatError::InvalidData(format!(
                "nbit: input size {} != expected {}",
                data.len(),
                unpacked_size
            )));
        }
        // Worst case the packed buffer is the same size as the unpacked one.
        let mut buffer = vec![0u8; unpacked_size + 1];
        let mut cur = NbitCursor { j: 0, buf_len: 8 };
        match cd_values[3] {
            NBIT_ATOMIC => {
                let mut idx = 4;
                let p = read_atomic(cd_values, &mut idx)?;
                for i in 0..d_nelmts {
                    nbit_compress_one_atomic(data, i * p.size as usize, &mut buffer, &mut cur, &p);
                }
            }
            NBIT_ARRAY => {
                let size = cd_values[4] as usize;
                for i in 0..d_nelmts {
                    let mut idx = 4;
                    nbit_compress_one_array(
                        data,
                        i * size,
                        &mut buffer,
                        &mut cur,
                        cd_values,
                        &mut idx,
                    )?;
                }
            }
            NBIT_COMPOUND => {
                let size = cd_values[4] as usize;
                for i in 0..d_nelmts {
                    let mut idx = 4;
                    nbit_compress_one_compound(
                        data,
                        i * size,
                        &mut buffer,
                        &mut cur,
                        cd_values,
                        &mut idx,
                    )?;
                }
            }
            other => {
                return Err(FormatError::InvalidData(format!(
                    "nbit: unsupported top class {}",
                    other
                )))
            }
        }
        // libhdf5 reports new_size + 1 (any hanging bits round up).
        buffer.truncate(cur.j + 1);
        Ok(buffer)
    } else {
        let mut out = vec![0u8; unpacked_size];
        let mut cur = NbitCursor { j: 0, buf_len: 8 };
        match cd_values[3] {
            NBIT_ATOMIC => {
                let mut idx = 4;
                let p = read_atomic(cd_values, &mut idx)?;
                if p.precision > p.size * 8 || p.precision + p.offset > p.size * 8 {
                    return Err(FormatError::InvalidData(
                        "nbit: invalid precision/offset".into(),
                    ));
                }
                for i in 0..d_nelmts {
                    nbit_decompress_one_atomic(&mut out, i * p.size as usize, data, &mut cur, &p)?;
                }
            }
            NBIT_ARRAY => {
                let size = cd_values[4] as usize;
                for i in 0..d_nelmts {
                    let mut idx = 4;
                    nbit_decompress_one_array(
                        &mut out,
                        i * size,
                        data,
                        &mut cur,
                        cd_values,
                        &mut idx,
                    )?;
                }
            }
            NBIT_COMPOUND => {
                let size = cd_values[4] as usize;
                for i in 0..d_nelmts {
                    let mut idx = 4;
                    nbit_decompress_one_compound(
                        &mut out,
                        i * size,
                        data,
                        &mut cur,
                        cd_values,
                        &mut idx,
                    )?;
                }
            }
            other => {
                return Err(FormatError::InvalidData(format!(
                    "nbit: unsupported top class {}",
                    other
                )))
            }
        }
        Ok(out)
    }
}

// ===========================================================================
//  Scale-offset filter (H5Z_FILTER_SCALEOFFSET, id 6)
// ===========================================================================

// cd_values index layout (H5Zscaleoffset.c).
const SO_PARM_SCALETYPE: usize = 0;
const SO_PARM_SCALEFACTOR: usize = 1;
const SO_PARM_NELMTS: usize = 2;
const SO_PARM_CLASS: usize = 3;
const SO_PARM_SIZE: usize = 4;
const SO_PARM_SIGN: usize = 5;
const SO_PARM_ORDER: usize = 6;
const SO_PARM_FILAVAIL: usize = 7;

const SO_CLS_INTEGER: u32 = 0;
const SO_CLS_FLOAT: u32 = 1;
const SO_ORDER_LE: u32 = 0;
const SO_FILL_DEFINED: u32 = 1;
// Float scale type: 0 = variable-minimum-bits (D-scale); 1 = E-scale (unsupported).
const SO_FLOAT_DSCALE: u32 = 0;

/// 21-byte parameter header stored in front of every scale-offset chunk.
const SO_BUF_OFFSET: usize = 21;

/// Decompress one scale-offset byte, mirroring
/// `H5Z__scaleoffset_decompress_one_byte`.
#[allow(clippy::too_many_arguments)]
fn so_decompress_one_byte(
    data: &mut [u8],
    data_offset: usize,
    k: u32,
    begin_i: u32,
    buffer: &[u8],
    cur: &mut NbitCursor,
    minbits: u32,
    dtype_len: u32,
) -> FormatResult<()> {
    if cur.j >= buffer.len() {
        return Err(FormatError::InvalidData(
            "scaleoffset: buffer too short".into(),
        ));
    }
    let mut val = buffer[cur.j];
    let mut bits_to_copy: usize = if k == begin_i {
        8 - ((dtype_len - minbits) % 8) as usize
    } else {
        8
    };

    let idx = data_offset + k as usize;
    if cur.buf_len > bits_to_copy {
        data[idx] = ((val >> (cur.buf_len - bits_to_copy)) as u32 & mask_u32(bits_to_copy)) as u8;
        cur.buf_len -= bits_to_copy;
    } else {
        data[idx] = ((val as u32 & mask_u32(cur.buf_len)) << (bits_to_copy - cur.buf_len)) as u8;
        bits_to_copy -= cur.buf_len;
        cur.next_byte();
        if bits_to_copy == 0 {
            return Ok(());
        }
        if cur.j >= buffer.len() {
            return Err(FormatError::InvalidData(
                "scaleoffset: buffer too short".into(),
            ));
        }
        val = buffer[cur.j];
        data[idx] |= ((val >> (cur.buf_len - bits_to_copy)) as u32 & mask_u32(bits_to_copy)) as u8;
        cur.buf_len -= bits_to_copy;
    }
    Ok(())
}

/// Decompress one scale-offset atomic element, mirroring
/// `H5Z__scaleoffset_decompress_one_atomic`.
fn so_decompress_one_atomic(
    data: &mut [u8],
    data_offset: usize,
    buffer: &[u8],
    cur: &mut NbitCursor,
    size: u32,
    minbits: u32,
    order: u32,
) -> FormatResult<()> {
    let dtype_len = size * 8;
    if order == SO_ORDER_LE {
        let begin_i = size - 1 - (dtype_len - minbits) / 8;
        let mut k = begin_i as i64;
        while k >= 0 {
            so_decompress_one_byte(
                data,
                data_offset,
                k as u32,
                begin_i,
                buffer,
                cur,
                minbits,
                dtype_len,
            )?;
            k -= 1;
        }
    } else {
        let begin_i = (dtype_len - minbits) / 8;
        for k in begin_i..=(size - 1) {
            so_decompress_one_byte(
                data,
                data_offset,
                k,
                begin_i,
                buffer,
                cur,
                minbits,
                dtype_len,
            )?;
        }
    }
    Ok(())
}

/// Read a little-/big-endian integer of `size` bytes from `data` at `offset`.
fn read_uint(data: &[u8], offset: usize, size: usize, order: u32) -> u64 {
    let mut v: u64 = 0;
    if order == SO_ORDER_LE {
        for i in 0..size {
            v |= (data[offset + i] as u64) << (i * 8);
        }
    } else {
        for i in 0..size {
            v = (v << 8) | data[offset + i] as u64;
        }
    }
    v
}

/// Write a little-/big-endian integer of `size` bytes into `data` at `offset`.
fn write_uint(data: &mut [u8], offset: usize, size: usize, order: u32, v: u64) {
    if order == SO_ORDER_LE {
        for i in 0..size {
            data[offset + i] = (v >> (i * 8)) as u8;
        }
    } else {
        for i in 0..size {
            data[offset + i] = (v >> ((size - 1 - i) * 8)) as u8;
        }
    }
}

/// Reverse the HDF5 scale-offset filter (decompress only).
///
/// `cd_values` follows `H5Zscaleoffset.c`'s 20-entry schema. The output is
/// the raw element buffer in the dataset datatype's byte order.
pub fn reverse_scaleoffset(data: &[u8], cd_values: &[u32]) -> FormatResult<Vec<u8>> {
    if cd_values.len() < 8 {
        return Err(FormatError::InvalidData(
            "scaleoffset: cd_values too short".into(),
        ));
    }
    let scale_type = cd_values[SO_PARM_SCALETYPE];
    let scale_factor = cd_values[SO_PARM_SCALEFACTOR] as i32;
    let d_nelmts = cd_values[SO_PARM_NELMTS] as usize;
    let dtype_class = cd_values[SO_PARM_CLASS];
    let size = cd_values[SO_PARM_SIZE] as usize;
    let dtype_sign = cd_values[SO_PARM_SIGN];
    let order = cd_values[SO_PARM_ORDER];
    let filavail = cd_values[SO_PARM_FILAVAIL];

    if size == 0 || size > 8 {
        return Err(FormatError::InvalidData(format!(
            "scaleoffset: unsupported datatype size {}",
            size
        )));
    }
    if filavail == SO_FILL_DEFINED {
        // Reconstructing a fill value from cd_values is not implemented;
        // fail loudly rather than emit wrong data.
        return Err(FormatError::UnsupportedFeature(
            "scaleoffset with a defined fill value is not supported".into(),
        ));
    }
    if dtype_class == SO_CLS_FLOAT && scale_type != SO_FLOAT_DSCALE {
        return Err(FormatError::UnsupportedFeature(
            "scaleoffset E-scaling method is not supported".into(),
        ));
    }

    let size_out = d_nelmts * size;

    // For integer types, scale_factor < 0 is reset to 0 by the library.
    let int_scalefactor = if scale_factor < 0 { 0 } else { scale_factor };
    if dtype_class == SO_CLS_INTEGER && int_scalefactor as usize == size * 8 {
        // No processing: payload after the header is the raw data.
        if data.len() < SO_BUF_OFFSET + size_out {
            return Err(FormatError::InvalidData(
                "scaleoffset: buffer too short".into(),
            ));
        }
        return Ok(data[SO_BUF_OFFSET..SO_BUF_OFFSET + size_out].to_vec());
    }

    // Read minbits + minval from the 21-byte header (always little-endian).
    if data.len() < SO_BUF_OFFSET {
        return Err(FormatError::InvalidData(
            "scaleoffset: buffer too short for header".into(),
        ));
    }
    let mut minbits: u32 = 0;
    for i in 0..4 {
        minbits |= (data[i] as u32) << (i * 8);
    }
    if minbits as usize > size * 8 {
        return Err(FormatError::InvalidData(
            "scaleoffset: minbits exceeds datatype size".into(),
        ));
    }
    let minval_size = std::cmp::min(8usize, data[4] as usize);
    let mut minval: u64 = 0;
    for i in 0..minval_size {
        minval |= (data[5 + i] as u64) << (i * 8);
    }

    // Special case: full precision -> payload copied verbatim.
    if minbits as usize == size * 8 {
        if data.len() < SO_BUF_OFFSET + size_out {
            return Err(FormatError::InvalidData(
                "scaleoffset: buffer too short".into(),
            ));
        }
        return Ok(data[SO_BUF_OFFSET..SO_BUF_OFFSET + size_out].to_vec());
    }

    let mut out = vec![0u8; size_out];

    if minbits != 0 {
        if data.len() < SO_BUF_OFFSET {
            return Err(FormatError::InvalidData(
                "scaleoffset: buffer too short".into(),
            ));
        }
        let payload = &data[SO_BUF_OFFSET..];
        let mut cur = NbitCursor { j: 0, buf_len: 8 };
        for i in 0..d_nelmts {
            so_decompress_one_atomic(
                &mut out,
                i * size,
                payload,
                &mut cur,
                size as u32,
                minbits,
                order,
            )?;
        }
    }
    // minbits == 0: out stays all-zero (all elements identical, no fill value).

    // Postprocess: add back minval (and apply float scaling).
    postdecompress(
        &mut out,
        d_nelmts,
        size,
        order,
        dtype_class,
        dtype_sign,
        minval,
        scale_factor,
    );

    Ok(out)
}

/// Postprocess decompressed scale-offset data (no fill value defined).
#[allow(clippy::too_many_arguments)]
fn postdecompress(
    out: &mut [u8],
    d_nelmts: usize,
    size: usize,
    order: u32,
    dtype_class: u32,
    dtype_sign: u32,
    minval: u64,
    scale_factor: i32,
) {
    if dtype_class == SO_CLS_INTEGER {
        // buf[i] = buf[i] + minval (wrapping in the datatype width).
        let width_mask: u64 = if size >= 8 {
            u64::MAX
        } else {
            (1u64 << (size * 8)) - 1
        };
        for i in 0..d_nelmts {
            let off = i * size;
            let v = read_uint(out, off, size, order);
            let sum = v.wrapping_add(minval) & width_mask;
            write_uint(out, off, size, order, sum);
        }
        let _ = dtype_sign;
    } else {
        // Float D-scale: value = (decompressed_int) / 10^D + min,
        // where `min` reinterprets `minval`'s low bits as the float type.
        let d_val = scale_factor as f64;
        let divisor = 10f64.powf(d_val);
        if size == 4 {
            let min = f32::from_bits(minval as u32);
            for i in 0..d_nelmts {
                let off = i * size;
                let raw = read_uint(out, off, size, order) as u32;
                let val = (raw as f32) / (divisor as f32) + min;
                write_uint(out, off, size, order, val.to_bits() as u64);
            }
        } else if size == 8 {
            let min = f64::from_bits(minval);
            for i in 0..d_nelmts {
                let off = i * size;
                let raw = read_uint(out, off, size, order);
                let val = (raw as f64) / divisor + min;
                write_uint(out, off, size, order, val.to_bits());
            }
        }
    }
}

// ===========================================================================
//  Tests
// ===========================================================================
#[cfg(test)]
mod tests {
    use super::*;

    /// Build an nbit cd_values list for an unsigned little-endian atomic int.
    fn nbit_atomic_cd(d_nelmts: u32, size: u32, precision: u32, offset: u32) -> Vec<u32> {
        // [0]=nparms [1]=need_not_compress [2]=d_nelmts [3]=class [4]=size
        // [5]=order [6]=precision [7]=offset
        let need_not_compress = if offset == 0 && precision == size * 8 {
            1
        } else {
            0
        };
        vec![
            8,
            need_not_compress,
            d_nelmts,
            NBIT_ATOMIC,
            size,
            NBIT_ORDER_LE,
            precision,
            offset,
        ]
    }

    #[test]
    fn nbit_roundtrip_u16_precision12() {
        // 16-bit storage, 12-bit precision, offset 0.
        let values: Vec<u16> = (0..40u16).map(|i| (i * 71) & 0x0FFF).collect();
        let mut raw = Vec::new();
        for &v in &values {
            raw.extend_from_slice(&v.to_le_bytes());
        }
        let cd = nbit_atomic_cd(values.len() as u32, 2, 12, 0);
        let packed = apply_nbit(&raw, &cd, true).unwrap();
        assert!(packed.len() <= raw.len());
        let unpacked = apply_nbit(&packed, &cd, false).unwrap();
        assert_eq!(unpacked, raw);
    }

    #[test]
    fn nbit_roundtrip_u32_precision20_offset4() {
        let values: Vec<u32> = (0..32u32).map(|i| ((i * 9999) & 0xFFFFF) << 4).collect();
        let mut raw = Vec::new();
        for &v in &values {
            raw.extend_from_slice(&v.to_le_bytes());
        }
        let cd = nbit_atomic_cd(values.len() as u32, 4, 20, 4);
        let packed = apply_nbit(&raw, &cd, true).unwrap();
        let unpacked = apply_nbit(&packed, &cd, false).unwrap();
        assert_eq!(unpacked, raw);
    }

    #[test]
    fn nbit_passthrough_full_precision() {
        let raw: Vec<u8> = (0..64).collect();
        let cd = nbit_atomic_cd(16, 4, 32, 0); // full precision -> need_not_compress
        let packed = apply_nbit(&raw, &cd, true).unwrap();
        assert_eq!(packed, raw);
        let unpacked = apply_nbit(&packed, &cd, false).unwrap();
        assert_eq!(unpacked, raw);
    }

    #[test]
    fn nbit_roundtrip_big_endian() {
        let values: Vec<u16> = (0..24u16).map(|i| (i * 53) & 0x03FF).collect();
        let mut raw = Vec::new();
        for &v in &values {
            raw.extend_from_slice(&v.to_be_bytes());
        }
        let mut cd = nbit_atomic_cd(values.len() as u32, 2, 10, 0);
        cd[5] = NBIT_ORDER_BE;
        let packed = apply_nbit(&raw, &cd, true).unwrap();
        let unpacked = apply_nbit(&packed, &cd, false).unwrap();
        assert_eq!(unpacked, raw);
    }
}
