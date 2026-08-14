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
pub(crate) const NBIT_ATOMIC: u32 = 1;
const NBIT_ARRAY: u32 = 2;
const NBIT_COMPOUND: u32 = 3;
const NBIT_NOOPTYPE: u32 = 4;
pub(crate) const NBIT_ORDER_LE: u32 = 0;
pub(crate) const NBIT_ORDER_BE: u32 = 1;

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
        let begin_i = if !(p.precision + p.offset).is_multiple_of(8) {
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
        let end_i = if !p.offset.is_multiple_of(8) {
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
        let begin_i = if !(p.precision + p.offset).is_multiple_of(8) {
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
        let end_i = if !p.offset.is_multiple_of(8) {
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
    // Validate every atomic (top-level, array member, compound member) so
    // the bit math below cannot overflow or panic on a crafted file.
    let bits = p.size.checked_mul(8);
    let span = p.precision.checked_add(p.offset);
    match (bits, span) {
        (Some(bits), Some(span))
            if p.size > 0 && p.precision > 0 && p.precision <= bits && span <= bits => {}
        _ => {
            return Err(FormatError::InvalidData(format!(
                "nbit: invalid atomic datatype (size={}, precision={}, offset={})",
                p.size, p.precision, p.offset
            )));
        }
    }
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
/// First cd_values index holding the (optional) packed fill value.
const SO_PARM_FILVAL: usize = 8;

pub(crate) const SO_CLS_INTEGER: u32 = 0;
pub(crate) const SO_CLS_FLOAT: u32 = 1;
pub(crate) const SO_ORDER_LE: u32 = 0;
const SO_FILL_DEFINED: u32 = 1;
// Float scale type: 0 = variable-minimum-bits (D-scale); 1 = E-scale (unsupported).
pub(crate) const SO_FLOAT_DSCALE: u32 = 0;
/// `H5Z_SO_INT`: the scale type an integer dataset must carry.
pub(crate) const SO_INT: u32 = 2;
/// `H5Z_SCALEOFFSET_SGN_NONE`: an unsigned integer.
pub(crate) const SO_SGN_NONE: u32 = 0;
/// `H5Z_SCALEOFFSET_SGN_2`: a two's-complement signed integer.
pub(crate) const SO_SGN_2: u32 = 1;
/// `H5Z_SCALEOFFSET_ORDER_BE`.
pub(crate) const SO_ORDER_BE: u32 = 1;
/// `H5Z_SCALEOFFSET_TOTAL_NPARMS`: the length of the stored `cd_values`.
pub(crate) const SO_TOTAL_NPARMS: usize = 20;

/// 21-byte parameter header stored in front of every scale-offset chunk.
const SO_BUF_OFFSET: usize = 21;

/// `H5Z__scaleoffset_log2`: the ceiling of log2, with `log2(0) == 1`.
fn so_log2(num: u64) -> u32 {
    let mut v = 0u32;
    let mut lower_bound: u64 = 1;
    let mut val = num;
    while {
        val >>= 1;
        val != 0
    } {
        v += 1;
        lower_bound <<= 1;
    }
    if num == lower_bound {
        v
    } else {
        v + 1
    }
}

/// The filter parameters `cd_values` carries, parsed once for both
/// directions the way `H5Z__filter_scaleoffset` reads them before it splits
/// on `H5Z_FLAG_REVERSE`.
#[derive(Clone, Copy)]
struct SoParams {
    scale_factor: i32,
    d_nelmts: usize,
    dtype_class: u32,
    /// Element size in bytes; 1, 2, 4 or 8 (`H5Z__scaleoffset_get_type`).
    size: usize,
    dtype_sign: u32,
    order: u32,
    fill_defined: bool,
    /// The fill value's bit image, masked to `size` bytes.
    filval: u64,
}

impl SoParams {
    fn parse(cd_values: &[u32]) -> FormatResult<Self> {
        if cd_values.len() < 8 {
            return Err(FormatError::InvalidData(
                "scaleoffset: cd_values too short".into(),
            ));
        }
        let scale_type = cd_values[SO_PARM_SCALETYPE];
        let dtype_class = cd_values[SO_PARM_CLASS];
        let size = cd_values[SO_PARM_SIZE] as usize;
        let fill_defined = cd_values[SO_PARM_FILAVAIL] == SO_FILL_DEFINED;

        if size == 0 || size > 8 {
            return Err(FormatError::InvalidData(format!(
                "scaleoffset: unsupported datatype size {}",
                size
            )));
        }
        if dtype_class == SO_CLS_FLOAT && scale_type != SO_FLOAT_DSCALE {
            return Err(FormatError::UnsupportedFeature(
                "scaleoffset E-scaling method is not supported".into(),
            ));
        }

        // Reconstruct the packed fill value from cd_values[8..]. libhdf5
        // stores it 4 bytes per cd_value, least-significant cd_value first;
        // each cd_value holds the bytes in the dataset datatype's byte order.
        // We read it as a raw `size`-byte little-endian-composed value
        // (correct for the common little-endian-dataset case h5py emits on
        // x86/ARM).
        let filval: u64 = if fill_defined {
            let mut v: u64 = 0;
            let n_cd = size.div_ceil(4);
            if cd_values.len() < SO_PARM_FILVAL + n_cd {
                return Err(FormatError::InvalidData(
                    "scaleoffset: cd_values missing fill value".into(),
                ));
            }
            for (w, cd) in cd_values[SO_PARM_FILVAL..SO_PARM_FILVAL + n_cd]
                .iter()
                .enumerate()
            {
                v |= (*cd as u64) << (w * 32);
            }
            v & mask_u64(size * 8)
        } else {
            0
        };

        // For integer types, a negative scale factor is reset to 0 by the
        // library, which makes it compute the minimum bit count itself.
        let mut scale_factor = cd_values[SO_PARM_SCALEFACTOR] as i32;
        if dtype_class == SO_CLS_INTEGER && scale_factor < 0 {
            scale_factor = 0;
        }

        Ok(Self {
            scale_factor,
            d_nelmts: cd_values[SO_PARM_NELMTS] as usize,
            dtype_class,
            size,
            dtype_sign: cd_values[SO_PARM_SIGN],
            order: cd_values[SO_PARM_ORDER],
            fill_defined,
            filval,
        })
    }

    /// True when the filter does nothing at all in either direction: an
    /// integer dataset whose user-set minimum bit count already spans the
    /// datatype (`HGOTO_DONE(nbytes)` in `H5Z__filter_scaleoffset`, reached
    /// before the forward/reverse split, so no header is written either).
    fn is_noop(&self) -> bool {
        self.dtype_class == SO_CLS_INTEGER && self.scale_factor as usize == self.size * 8
    }

    /// The element's precision in bits.
    fn dtype_len(&self) -> u32 {
        (self.size * 8) as u32
    }

    /// The low `size * 8` bits set.
    fn width_mask(&self) -> u64 {
        mask_u64(self.size * 8)
    }
}

/// `~((u64)0 << n)`: the low `n` bits set.
fn mask_u64(n: usize) -> u64 {
    if n >= 64 {
        u64::MAX
    } else {
        !(u64::MAX << n)
    }
}

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

/// Compress one scale-offset byte, mirroring
/// `H5Z__scaleoffset_compress_one_byte`.
///
/// `cur.buf_len` is the C's `bits_to_fill`: how much room is left in the
/// buffer byte the cursor sits on.
#[allow(clippy::too_many_arguments)]
fn so_compress_one_byte(
    data: &[u8],
    data_offset: usize,
    k: u32,
    begin_i: u32,
    buffer: &mut [u8],
    cur: &mut NbitCursor,
    minbits: u32,
    dtype_len: u32,
) {
    let val = data[data_offset + k as usize];
    let mut bits_to_copy: usize = if k == begin_i {
        8 - ((dtype_len - minbits) % 8) as usize
    } else {
        8
    };

    if cur.buf_len > bits_to_copy {
        buffer[cur.j] |=
            ((val as u32 & mask_u32(bits_to_copy)) << (cur.buf_len - bits_to_copy)) as u8;
        cur.buf_len -= bits_to_copy;
    } else {
        buffer[cur.j] |=
            ((val >> (bits_to_copy - cur.buf_len)) as u32 & mask_u32(cur.buf_len)) as u8;
        bits_to_copy -= cur.buf_len;
        cur.next_byte();
        if bits_to_copy == 0 {
            return;
        }
        buffer[cur.j] =
            ((val as u32 & mask_u32(bits_to_copy)) << (cur.buf_len - bits_to_copy)) as u8;
        cur.buf_len -= bits_to_copy;
    }
}

/// Compress one scale-offset atomic element, mirroring
/// `H5Z__scaleoffset_compress_one_atomic`.
fn so_compress_one_atomic(
    data: &[u8],
    data_offset: usize,
    buffer: &mut [u8],
    cur: &mut NbitCursor,
    size: u32,
    minbits: u32,
    order: u32,
) {
    let dtype_len = size * 8;
    if order == SO_ORDER_LE {
        let begin_i = size - 1 - (dtype_len - minbits) / 8;
        let mut k = begin_i as i64;
        while k >= 0 {
            so_compress_one_byte(
                data,
                data_offset,
                k as u32,
                begin_i,
                buffer,
                cur,
                minbits,
                dtype_len,
            );
            k -= 1;
        }
    } else {
        let begin_i = (dtype_len - minbits) / 8;
        for k in begin_i..=(size - 1) {
            so_compress_one_byte(
                data,
                data_offset,
                k,
                begin_i,
                buffer,
                cur,
                minbits,
                dtype_len,
            );
        }
    }
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
    let p = SoParams::parse(cd_values)?;
    let (d_nelmts, size, order) = (p.d_nelmts, p.size, p.order);
    let size_out = d_nelmts * size;

    if p.is_noop() {
        // A user-set minimum-bit count equal to full precision makes the
        // filter a no-op in both directions (`HGOTO_DONE(nbytes)`, before the
        // forward/reverse split in `H5Z__filter_scaleoffset`): the chunk is
        // the raw element buffer, with no parameter header in front of it.
        if data.len() < size_out {
            return Err(FormatError::InvalidData(
                "scaleoffset: buffer too short".into(),
            ));
        }
        return Ok(data[..size_out].to_vec());
    }

    // Read minbits + minval from the 21-byte header (always little-endian).
    if data.len() < SO_BUF_OFFSET {
        return Err(FormatError::InvalidData(
            "scaleoffset: buffer too short for header".into(),
        ));
    }
    let mut minbits: u32 = 0;
    for (i, &b) in data[..4].iter().enumerate() {
        minbits |= (b as u32) << (i * 8);
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
        p.dtype_class,
        p.dtype_sign,
        minbits,
        minval,
        p.scale_factor,
        p.fill_defined,
        p.filval,
    );

    Ok(out)
}

/// Apply the HDF5 scale-offset filter (compress).
///
/// `data` is the raw element buffer in the dataset datatype's byte order and
/// the result is the stored chunk: a 21-byte parameter header holding the
/// chosen minimum bit count and the chunk minimum, followed by the packed
/// values. Both the header and the compressed length mirror
/// `H5Z__filter_scaleoffset`'s forward branch, including its allocation of
/// one byte more than the packed bits need.
pub fn forward_scaleoffset(data: &[u8], cd_values: &[u32]) -> FormatResult<Vec<u8>> {
    let p = SoParams::parse(cd_values)?;
    let nbytes = p.d_nelmts * p.size;
    if data.len() != nbytes {
        return Err(FormatError::InvalidData(format!(
            "scaleoffset: chunk is {} bytes, but the filter parameters describe {} elements of \
             {} bytes",
            data.len(),
            p.d_nelmts,
            p.size
        )));
    }
    if p.is_noop() {
        return Ok(data.to_vec());
    }
    if p.dtype_class == SO_CLS_INTEGER && p.scale_factor as usize > p.size * 8 {
        return Err(FormatError::InvalidData(
            "scaleoffset: minimum number of bits exceeds the datatype".into(),
        ));
    }

    // Preprocess: rewrite every element as its offset from the chunk minimum
    // (fill values become the all-ones sentinel), and settle the bit count
    // that offset needs.
    let mut buf = data.to_vec();
    let (minbits, minval) = if p.dtype_class == SO_CLS_INTEGER {
        precompress_int(&mut buf, &p)
    } else {
        precompress_float(&mut buf, &p)?
    };
    debug_assert!(minbits <= p.dtype_len());

    // `size_out` is what libhdf5 allocates and reports: one byte more than
    // the packed bits occupy, which is what a reader finds stored.
    let size_out = SO_BUF_OFFSET + nbytes * minbits as usize / (p.size * 8) + 1;
    let mut out = vec![0u8; size_out];
    out[..4].copy_from_slice(&minbits.to_le_bytes());
    // libhdf5 stores `sizeof(unsigned long long)` here and the reader takes
    // the smaller of that and its own, so the count of minval bytes that
    // follow is fixed at 8.
    out[4] = 8;
    out[5..13].copy_from_slice(&minval.to_le_bytes());

    if minbits as usize == p.size * 8 {
        // Full precision: the offsets need every bit, so they are stored
        // unpacked, and the trailing allocated byte is not part of the chunk.
        out.truncate(SO_BUF_OFFSET + nbytes);
        out[SO_BUF_OFFSET..].copy_from_slice(&buf);
        return Ok(out);
    }
    if minbits != 0 {
        let (header, payload) = out.split_at_mut(SO_BUF_OFFSET);
        let _ = header;
        let mut cur = NbitCursor { j: 0, buf_len: 8 };
        for i in 0..p.d_nelmts {
            so_compress_one_atomic(
                &buf,
                i * p.size,
                payload,
                &mut cur,
                p.size as u32,
                minbits,
                p.order,
            );
        }
    }
    // minbits == 0: every element is the chunk minimum, so the payload is
    // the single zero byte the size formula leaves.
    Ok(out)
}

/// Preprocess an integer chunk, mirroring `H5Z__scaleoffset_precompress_i`.
///
/// Returns `(minbits, minval)` and leaves `buf` holding each element's offset
/// from the chunk minimum — or the all-ones sentinel where the element was
/// the fill value.
fn precompress_int(buf: &mut [u8], p: &SoParams) -> (u32, u64) {
    let signed = p.dtype_sign == SO_SGN_2;
    let width_mask = p.width_mask();
    // The comparison key: a signed element orders by its sign-extended
    // value, an unsigned one by its raw bits. `i128` holds both.
    let key = |raw: u64| -> i128 {
        if signed {
            sign_extend(raw, p.size) as i128
        } else {
            raw as i128
        }
    };
    let elem = |buf: &[u8], i: usize| read_uint(buf, i * p.size, p.size, p.order);

    let mut minbits = p.scale_factor as u32;
    let mut min: i128 = 0;
    let mut max: i128 = 0;

    if p.fill_defined {
        // Fill elements take no part in the range.
        let first = (0..p.d_nelmts).find(|&i| elem(buf, i) != p.filval);
        if let Some(f) = first {
            min = key(elem(buf, f));
            max = min;
            for i in f..p.d_nelmts {
                let raw = elem(buf, i);
                if raw == p.filval {
                    continue;
                }
                let v = key(raw);
                max = max.max(v);
                min = min.min(v);
            }
        }
        if minbits == 0 {
            let span_minus_1 = (max - min) as u64;
            if span_minus_1 > width_mask - 2 {
                // No sentinel fits above the range; store at full precision.
                // libhdf5 leaves minval at 0 here, which the reader ignores.
                return (p.dtype_len(), 0);
            }
            minbits = so_log2(span_minus_1 + 2);
        }
        if minbits != p.dtype_len() {
            let sentinel = mask_u64(minbits as usize);
            for i in 0..p.d_nelmts {
                let raw = elem(buf, i);
                let v = if raw == p.filval {
                    sentinel
                } else {
                    (key(raw) - min) as u64 & width_mask
                };
                write_uint(buf, i * p.size, p.size, p.order, v);
            }
        }
    } else {
        min = key(elem(buf, 0));
        max = min;
        for i in 0..p.d_nelmts {
            let v = key(elem(buf, i));
            max = max.max(v);
            min = min.min(v);
        }
        if minbits == 0 {
            let span_minus_1 = (max - min) as u64;
            if span_minus_1 > width_mask - 2 {
                return (p.dtype_len(), 0);
            }
            minbits = so_log2(span_minus_1 + 1);
        }
        if minbits != p.dtype_len() {
            for i in 0..p.d_nelmts {
                let v = (key(elem(buf, i)) - min) as u64 & width_mask;
                write_uint(buf, i * p.size, p.size, p.order, v);
            }
        }
    }

    (
        minbits,
        min as i64 as u64 & if signed { u64::MAX } else { width_mask },
    )
}

/// The float operations `H5Z_scaleoffset_precompress_3` performs, each in the
/// element's own precision: `powf`/`roundf`/`lroundf` for a 4-byte element,
/// `pow`/`round`/`lround` for an 8-byte one. Doing them all in `f64` would
/// pick a different `minbits` at the boundary for `float` data.
trait SoFloat: Copy + PartialOrd + std::ops::Mul<Output = Self> + std::ops::Sub<Output = Self> {
    const ZERO: Self;
    fn from_stored(v: u64) -> Self;
    fn to_stored(self) -> u64;
    fn widen(self) -> f64;
    /// `pow_fun((type)base, (type)exp)`.
    fn pow(base: f64, exp: f64) -> Self;
    fn abs(self) -> Self;
    /// `round_fun`: to nearest, halfway away from zero.
    fn round(self) -> Self;
    /// `lround_fun` / `llround_fun`.
    fn lround(self) -> i64;
}

impl SoFloat for f32 {
    const ZERO: Self = 0.0;
    fn from_stored(v: u64) -> Self {
        f32::from_bits(v as u32)
    }
    fn to_stored(self) -> u64 {
        self.to_bits() as u64
    }
    fn widen(self) -> f64 {
        self as f64
    }
    fn pow(base: f64, exp: f64) -> Self {
        (base as f32).powf(exp as f32)
    }
    fn abs(self) -> Self {
        f32::abs(self)
    }
    fn round(self) -> Self {
        f32::round(self)
    }
    fn lround(self) -> i64 {
        f32::round(self) as i64
    }
}

impl SoFloat for f64 {
    const ZERO: Self = 0.0;
    fn from_stored(v: u64) -> Self {
        f64::from_bits(v)
    }
    fn to_stored(self) -> u64 {
        self.to_bits()
    }
    fn widen(self) -> f64 {
        self
    }
    fn pow(base: f64, exp: f64) -> Self {
        base.powf(exp)
    }
    fn abs(self) -> Self {
        f64::abs(self)
    }
    fn round(self) -> Self {
        f64::round(self)
    }
    fn lround(self) -> i64 {
        f64::round(self) as i64
    }
}

/// Preprocess a floating-point chunk by the variable-minimum-bits (D-scale)
/// method, mirroring `H5Z__scaleoffset_precompress_fd`.
///
/// Each value becomes the integer `round(v * 10^D - min * 10^D)`, so the
/// scale factor is the number of decimal digits kept. Returns
/// `(minbits, minval)`, where `minval` is the bit image of the chunk minimum
/// as a float — the form [`postdecompress`] reads it back in.
fn precompress_float(buf: &mut [u8], p: &SoParams) -> FormatResult<(u32, u64)> {
    match p.size {
        4 => Ok(precompress_float_typed::<f32>(buf, p)),
        8 => Ok(precompress_float_typed::<f64>(buf, p)),
        n => Err(FormatError::InvalidData(format!(
            "scaleoffset: no floating-point type of {n} bytes"
        ))),
    }
}

fn precompress_float_typed<T: SoFloat>(buf: &mut [u8], p: &SoParams) -> (u32, u64) {
    let d_val = p.scale_factor as f64;
    let pow10 = T::pow(10.0, d_val);
    let filval = T::from_stored(p.filval);
    let get = |buf: &[u8], i: usize| T::from_stored(read_uint(buf, i * p.size, p.size, p.order));
    // `H5Z_scaleoffset_max_min_3` widens the difference to `double` and
    // compares against a `double` threshold whatever the element type is,
    // while `H5Z_scaleoffset_modify_1` stays in the element type. For a
    // 4-byte element the two thresholds are not the same number, so the scan
    // and the rewrite each use their own.
    let scan_epsilon = 10f64.powf(-d_val);
    let is_fill_scan = |v: T| (v - filval).widen().abs() < scan_epsilon;
    let modify_epsilon = T::pow(10.0, -d_val);
    let is_fill_modify = |v: T| (v - filval).abs() < modify_epsilon;

    let mut min = T::ZERO;
    let mut max = T::ZERO;
    if p.fill_defined {
        if let Some(f) = (0..p.d_nelmts).find(|&i| !is_fill_scan(get(buf, i))) {
            min = get(buf, f);
            max = min;
            for i in f..p.d_nelmts {
                let v = get(buf, i);
                if is_fill_scan(v) {
                    continue;
                }
                if v > max {
                    max = v;
                }
                if v < min {
                    min = v;
                }
            }
        }
    } else if p.d_nelmts > 0 {
        min = get(buf, 0);
        max = min;
        for i in 0..p.d_nelmts {
            let v = get(buf, i);
            if v > max {
                max = v;
            }
            if v < min {
                min = v;
            }
        }
    }

    // `H5Z_scaleoffset_check_3`: the scaled span has to stay inside the signed
    // integer the values are stored as. When it does not, the library gives up
    // on scaling and stores at full precision, leaving `minval` at 0 (its
    // `goto done` skips `H5Z_scaleoffset_save_min`).
    let dtype_len = p.dtype_len();
    let scaled = max * pow10 - min * pow10;
    if scaled.round() > T::pow(2.0, (dtype_len - 1) as f64) {
        return (dtype_len, 0);
    }
    let span = scaled.lround() as u64 + 1;
    let minbits = if p.fill_defined {
        so_log2(span + 1)
    } else {
        so_log2(span)
    };

    if minbits != dtype_len {
        let sentinel = mask_u64(minbits as usize);
        for i in 0..p.d_nelmts {
            let v = get(buf, i);
            let stored = if p.fill_defined && is_fill_modify(v) {
                sentinel
            } else {
                (v * pow10 - min * pow10).lround() as u64 & p.width_mask()
            };
            write_uint(buf, i * p.size, p.size, p.order, stored);
        }
    }

    (minbits, min.to_stored())
}

/// Sign-extend the low `size*8` bits of `v` to a full `i64`.
fn sign_extend(v: u64, size: usize) -> i64 {
    if size >= 8 {
        return v as i64;
    }
    let bits = size * 8;
    let shift = 64 - bits;
    ((v << shift) as i64) >> shift
}

/// Postprocess decompressed scale-offset data.
#[allow(clippy::too_many_arguments)]
fn postdecompress(
    out: &mut [u8],
    d_nelmts: usize,
    size: usize,
    order: u32,
    dtype_class: u32,
    dtype_sign: u32,
    minbits: u32,
    minval: u64,
    scale_factor: i32,
    fill_defined: bool,
    filval: u64,
) {
    // Sentinel: a fully decompressed value equal to (1 << minbits) - 1 is
    // restored to the fill value rather than offset-added.
    let sentinel: u64 = if (minbits as usize) >= 64 {
        u64::MAX
    } else {
        (1u64 << minbits) - 1
    };
    let width_mask: u64 = if size >= 8 {
        u64::MAX
    } else {
        (1u64 << (size * 8)) - 1
    };

    if dtype_class == SO_CLS_INTEGER {
        // buf[i] = (buf[i] == sentinel) ? filval : buf[i] + minval.
        for i in 0..d_nelmts {
            let off = i * size;
            let v = read_uint(out, off, size, order);
            let result = if fill_defined && v == sentinel {
                filval
            } else {
                v.wrapping_add(minval) & width_mask
            };
            write_uint(out, off, size, order, result);
        }
        let _ = dtype_sign;
    } else {
        // Float D-scale: value = (signed decompressed int) / 10^D + min,
        // where `min` reinterprets `minval`'s low bits as the float type.
        let d_val = scale_factor as f64;
        let divisor = 10f64.powf(d_val);
        if size == 4 {
            let min = f32::from_bits(minval as u32);
            let filval_f = f32::from_bits(filval as u32);
            for i in 0..d_nelmts {
                let off = i * size;
                let raw = read_uint(out, off, size, order);
                let val = if fill_defined && raw == sentinel {
                    filval_f
                } else {
                    (sign_extend(raw, size) as f32) / (divisor as f32) + min
                };
                write_uint(out, off, size, order, val.to_bits() as u64);
            }
        } else if size == 8 {
            let min = f64::from_bits(minval);
            let filval_f = f64::from_bits(filval);
            for i in 0..d_nelmts {
                let off = i * size;
                let raw = read_uint(out, off, size, order);
                if fill_defined && raw == sentinel {
                    write_uint(out, off, size, order, filval_f.to_bits());
                    continue;
                }
                let val = (sign_extend(raw, size) as f64) / divisor + min;
                write_uint(out, off, size, order, val.to_bits());
            }
        }
    }
}

// ===========================================================================
//  Post-filter datatype conversion (H5T_convert equivalent)
// ===========================================================================

use crate::format::messages::datatype::{ByteOrder, DatatypeMessage};

/// True if `dt` is a floating-point type whose bit layout is an IEEE 754
/// interchange format — the only float layouts that can be reinterpreted in
/// place. [`DatatypeMessage::ieee_format`] is the single owner of that rule.
fn is_standard_ieee_float(dt: &DatatypeMessage) -> bool {
    dt.ieee_format().is_some()
}

/// True if the filter-pipeline / on-disk output for `dt` needs a post-filter
/// datatype conversion before the element values are usable.
///
/// For a `FixedPoint` datatype the filter pipeline output (or contiguous
/// on-disk bytes) carries the significant value in `bit_precision` bits
/// starting at `bit_offset`, with the rest zero-filled and the sign bit NOT
/// extended. libhdf5 fixes this up with a datatype conversion
/// (`H5T_convert`) after the filter pipeline; this returns true for any
/// such non-trivial layout.
///
/// It also returns true for a non-standard `FloatingPoint` layout, so the
/// caller routes it through [`apply_datatype_conversion`], which then
/// returns a clear error rather than silently yielding wrong data.
pub fn datatype_needs_bit_conversion(dt: &DatatypeMessage) -> bool {
    match dt {
        DatatypeMessage::FixedPoint {
            size,
            bit_offset,
            bit_precision,
            ..
        } => *bit_offset != 0 || (*bit_precision as u32) < *size * 8,
        DatatypeMessage::FloatingPoint { .. } => !is_standard_ieee_float(dt),
        _ => false,
    }
}

/// Apply the post-filter datatype conversion in place to a fully-decoded
/// output buffer.
///
/// This mirrors libhdf5's `H5T_convert` step that runs AFTER the filter
/// pipeline. For a `FixedPoint` datatype with `bit_offset != 0` or
/// `bit_precision < size*8`, each `size`-byte element is rewritten so the
/// significant value occupies the whole element with bit offset 0:
///
///   1. interpret the element as an unsigned integer (respecting byte order),
///   2. shift right by `bit_offset`,
///   3. mask to `bit_precision` low bits,
///   4. sign-extend from bit `bit_precision-1` if the type is signed,
///   5. write the result back in the same byte order.
///
/// It is a strict no-op for ordinary full-width datatypes (and for any
/// non-`FixedPoint` class).
///
/// For `FloatingPoint` types with a non-standard bit layout that cannot be
/// faithfully reinterpreted, an error is returned rather than wrong data.
pub fn apply_datatype_conversion(buffer: &mut [u8], dt: &DatatypeMessage) -> FormatResult<()> {
    match dt {
        DatatypeMessage::FixedPoint {
            size,
            byte_order,
            signed,
            bit_offset,
            bit_precision,
        } => {
            let size = *size as usize;
            let precision = *bit_precision as usize;
            let offset = *bit_offset as usize;

            // Full-width plain integer: nothing to do.
            if offset == 0 && precision == size * 8 {
                return Ok(());
            }
            if size == 0 || size > 8 {
                return Err(FormatError::InvalidData(format!(
                    "datatype conversion: unsupported FixedPoint size {size}"
                )));
            }
            if precision == 0 || offset + precision > size * 8 {
                return Err(FormatError::InvalidData(format!(
                    "datatype conversion: invalid bit layout (offset {offset}, \
                     precision {precision}, size {size})"
                )));
            }
            if !buffer.len().is_multiple_of(size) {
                return Err(FormatError::InvalidData(format!(
                    "datatype conversion: buffer length {} not a multiple of \
                     element size {size}",
                    buffer.len()
                )));
            }

            let big_endian = matches!(byte_order, ByteOrder::BigEndian);
            let precision_mask: u64 = if precision == 64 {
                u64::MAX
            } else {
                (1u64 << precision) - 1
            };
            let sign_bit: u64 = 1u64 << (precision - 1);

            for elem in buffer.chunks_exact_mut(size) {
                // Load element as a u64 in native value space.
                let mut raw: u64 = 0;
                if big_endian {
                    for &b in elem.iter() {
                        raw = (raw << 8) | b as u64;
                    }
                } else {
                    for (i, &b) in elem.iter().enumerate() {
                        raw |= (b as u64) << (8 * i);
                    }
                }

                // Extract the significant bits.
                let mut value = (raw >> offset) & precision_mask;

                // Sign-extend from bit `precision-1` when signed.
                if *signed && (value & sign_bit) != 0 {
                    value |= !precision_mask;
                }

                // Store back in the same byte order, full element width.
                if big_endian {
                    for i in 0..size {
                        elem[size - 1 - i] = (value >> (8 * i)) as u8;
                    }
                } else {
                    for (i, b) in elem.iter_mut().enumerate() {
                        *b = (value >> (8 * i)) as u8;
                    }
                }
            }
            Ok(())
        }
        DatatypeMessage::FloatingPoint { .. } => {
            // Standard IEEE-754 layouts need no conversion. Anything else
            // cannot be faithfully reinterpreted here.
            if is_standard_ieee_float(dt) {
                Ok(())
            } else {
                Err(FormatError::InvalidData(
                    "datatype conversion: non-standard floating-point bit \
                     layout cannot be converted"
                        .into(),
                ))
            }
        }
        _ => Ok(()),
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

    // ---------------------------------------------------------------
    //  Post-filter datatype conversion
    // ---------------------------------------------------------------

    fn fixed(size: u32, signed: bool, offset: u16, precision: u16) -> DatatypeMessage {
        DatatypeMessage::FixedPoint {
            size,
            byte_order: ByteOrder::LittleEndian,
            signed,
            bit_offset: offset,
            bit_precision: precision,
        }
    }

    #[test]
    fn conversion_noop_for_full_width_types() {
        // 32-bit unsigned, offset 0, precision 32 -> plain integer, no-op.
        let dt = fixed(4, false, 0, 32);
        assert!(!datatype_needs_bit_conversion(&dt));
        let mut buf = vec![0x78, 0x56, 0x34, 0x12, 0xFF, 0xFF, 0xFF, 0xFF];
        let before = buf.clone();
        apply_datatype_conversion(&mut buf, &dt).unwrap();
        assert_eq!(buf, before);
    }

    #[test]
    fn conversion_noop_for_non_numeric_types() {
        let dt = DatatypeMessage::fixed_string(8);
        assert!(!datatype_needs_bit_conversion(&dt));
        let mut buf = b"hello!!\0".to_vec();
        let before = buf.clone();
        apply_datatype_conversion(&mut buf, &dt).unwrap();
        assert_eq!(buf, before);
    }

    #[test]
    fn conversion_unsigned_offset_shifts_right() {
        // u16, bit_offset 3, precision 10. The value lives in bits [3,13).
        // Raw element layout (LE u16): value 0x2A5 placed at offset 3 ->
        // 0x2A5 << 3 = 0x1528.
        let dt = fixed(2, false, 3, 10);
        assert!(datatype_needs_bit_conversion(&dt));
        let mut buf = (0x1528u16).to_le_bytes().to_vec();
        apply_datatype_conversion(&mut buf, &dt).unwrap();
        assert_eq!(u16::from_le_bytes([buf[0], buf[1]]), 0x2A5);
    }

    #[test]
    fn conversion_signed_negative_sign_extends() {
        // i16, bit_offset 4, precision 8. Store -3 (8-bit two's complement
        // = 0xFD) at offset 4: 0xFD << 4 = 0xFD0.
        let dt = fixed(2, true, 4, 8);
        let mut buf = (0x0FD0u16).to_le_bytes().to_vec();
        apply_datatype_conversion(&mut buf, &dt).unwrap();
        assert_eq!(i16::from_le_bytes([buf[0], buf[1]]), -3);
    }

    #[test]
    fn conversion_signed_positive_stays_positive() {
        // i16, bit_offset 4, precision 8. Store +5 at offset 4 -> 0x050.
        let dt = fixed(2, true, 4, 8);
        let mut buf = (0x0050u16).to_le_bytes().to_vec();
        apply_datatype_conversion(&mut buf, &dt).unwrap();
        assert_eq!(i16::from_le_bytes([buf[0], buf[1]]), 5);
    }

    #[test]
    fn conversion_reduced_precision_offset_zero() {
        // i32, bit_offset 0, precision 20 -> still non-trivial (precision <
        // size*8). Store -1 in 20 bits = 0xFFFFF.
        let dt = fixed(4, true, 0, 20);
        assert!(datatype_needs_bit_conversion(&dt));
        let mut buf = (0x000FFFFFu32).to_le_bytes().to_vec();
        apply_datatype_conversion(&mut buf, &dt).unwrap();
        assert_eq!(i32::from_le_bytes(buf.clone().try_into().unwrap()), -1);
    }

    #[test]
    fn conversion_big_endian_signed() {
        // i16, BE, bit_offset 4, precision 8, value -3.
        let dt = DatatypeMessage::FixedPoint {
            size: 2,
            byte_order: ByteOrder::BigEndian,
            signed: true,
            bit_offset: 4,
            bit_precision: 8,
        };
        let mut buf = (0x0FD0u16).to_be_bytes().to_vec();
        apply_datatype_conversion(&mut buf, &dt).unwrap();
        assert_eq!(i16::from_be_bytes([buf[0], buf[1]]), -3);
    }

    #[test]
    fn conversion_multiple_elements() {
        // u32, bit_offset 5, precision 16. Three elements.
        let dt = fixed(4, false, 5, 16);
        let vals: [u32; 3] = [0x1234, 0xABCD, 0x0001];
        let mut buf = Vec::new();
        for v in vals {
            buf.extend_from_slice(&(v << 5).to_le_bytes());
        }
        apply_datatype_conversion(&mut buf, &dt).unwrap();
        for (i, v) in vals.iter().enumerate() {
            let e = u32::from_le_bytes(buf[i * 4..i * 4 + 4].try_into().unwrap());
            assert_eq!(e, *v);
        }
    }

    #[test]
    fn conversion_rejects_non_standard_float() {
        // A float with a non-IEEE bit layout must error, not corrupt data.
        let dt = DatatypeMessage::FloatingPoint {
            size: 4,
            byte_order: ByteOrder::LittleEndian,
            sign_location: 30,
            bit_offset: 1,
            bit_precision: 31,
            exponent_location: 22,
            exponent_size: 8,
            mantissa_location: 0,
            mantissa_size: 22,
            exponent_bias: 127,
        };
        assert!(datatype_needs_bit_conversion(&dt));
        let mut buf = vec![0u8; 4];
        assert!(apply_datatype_conversion(&mut buf, &dt).is_err());
    }

    #[test]
    fn conversion_standard_float_is_noop() {
        let dt = DatatypeMessage::f64_type();
        assert!(!datatype_needs_bit_conversion(&dt));
        let mut buf = 12.5f64.to_le_bytes().to_vec();
        let before = buf.clone();
        apply_datatype_conversion(&mut buf, &dt).unwrap();
        assert_eq!(buf, before);
    }

    #[test]
    fn conversion_rejects_bad_buffer_length() {
        let dt = fixed(4, false, 3, 16);
        let mut buf = vec![0u8; 5]; // not a multiple of 4
        assert!(apply_datatype_conversion(&mut buf, &dt).is_err());
    }
}
