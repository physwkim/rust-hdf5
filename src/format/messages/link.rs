//! Link message (type 0x06) — encodes a single link within a group.
//!
//! Binary layout (version 1):
//!   Byte 0: version = 1
//!   Byte 1: flags
//!     bits 0-1: size of name-length field (0=1B, 1=2B, 2=4B, 3=8B)
//!     bit 2:    creation order present
//!     bit 3:    link type present
//!     bit 4:    charset field present
//!   [if bit 3]: link_type u8 (0=hard, 1=soft, 64+=external)
//!   [if bit 2]: creation_order i64 LE
//!   [if bit 4]: charset u8 (0=ASCII, 1=UTF-8)
//!   name_length: 1/2/4/8 bytes per bits 0-1
//!   name:        name_length bytes (UTF-8)
//!   [hard link]:  address (sizeof_addr bytes)
//!   [soft link]:  target_length u16 LE + target string
//!   [ud link]:    udata_length u16 LE + udata bytes
//!
//! The external link is the one user-defined link class libhdf5 ships
//! (`H5L_TYPE_EXTERNAL` = 64). Its udata is a version/flags byte followed by
//! the NUL-terminated target file name and the NUL-terminated object path
//! within that file (`H5Lexternal.c`).

use crate::format::bytes::read_le_uint as read_uint;
use crate::format::{FormatContext, FormatError, FormatResult};

const VERSION: u8 = 1;

const FLAG_NAME_LEN_MASK: u8 = 0x03;
const FLAG_CREATION_ORDER: u8 = 0x04;
const FLAG_LINK_TYPE: u8 = 0x08;
const FLAG_CHARSET: u8 = 0x10;

const LINK_TYPE_HARD: u8 = 0;
const LINK_TYPE_SOFT: u8 = 1;
/// `H5L_TYPE_EXTERNAL` — also `H5L_TYPE_UD_MIN`, the bottom of the
/// user-defined link range (64..=255).
const LINK_TYPE_EXTERNAL: u8 = 64;

/// Version nibble of the external-link udata (`H5L_EXT_VERSION`).
const EXT_VERSION: u8 = 0;

/// Link target discriminant.
#[derive(Debug, Clone, PartialEq)]
pub enum LinkTarget {
    /// Hard link — points to an object header at `address`.
    Hard { address: u64 },
    /// Soft link — points to a path string, resolved at traversal time.
    Soft { target: String },
    /// External link — points to `path` inside the file named `file`.
    External { file: String, path: String },
    /// Any other user-defined link class (65..=255). libhdf5 needs a
    /// registered link class to interpret `udata`, so it is kept verbatim:
    /// the link still has a name and still belongs in a listing.
    UserDefined { link_type: u8, udata: Vec<u8> },
}

/// Link message payload.
#[derive(Debug, Clone, PartialEq)]
pub struct LinkMessage {
    pub name: String,
    pub target: LinkTarget,
    /// Creation order within the parent group, present only when the group
    /// tracks it (`H5O_LINK_STORE_CORDER`). `H5G_obj_insert` stamps it from
    /// the Link Info message's running maximum.
    pub creation_order: Option<i64>,
}

impl LinkMessage {
    /// Create a hard link.
    pub fn hard(name: &str, address: u64) -> Self {
        Self {
            name: name.to_string(),
            target: LinkTarget::Hard { address },
            creation_order: None,
        }
    }

    /// Create a soft link.
    pub fn soft(name: &str, target: &str) -> Self {
        Self {
            name: name.to_string(),
            target: LinkTarget::Soft {
                target: target.to_string(),
            },
            creation_order: None,
        }
    }

    /// Create an external link to `path` inside `file`.
    pub fn external(name: &str, file: &str, path: &str) -> Self {
        Self {
            name: name.to_string(),
            target: LinkTarget::External {
                file: file.to_string(),
                path: path.to_string(),
            },
            creation_order: None,
        }
    }

    /// The same link, stamped with its creation order.
    pub fn with_creation_order(mut self, corder: i64) -> Self {
        self.creation_order = Some(corder);
        self
    }

    // ------------------------------------------------------------------ encode

    pub fn encode(&self, ctx: &FormatContext) -> Vec<u8> {
        let name_bytes = self.name.as_bytes();
        let name_len = name_bytes.len();
        let name_len_size = min_bytes_for_value(name_len as u64);
        let name_len_code = match name_len_size {
            1 => 0u8,
            2 => 1,
            4 => 2,
            _ => 3, // 8
        };

        let link_type = match &self.target {
            LinkTarget::Hard { .. } => LINK_TYPE_HARD,
            LinkTarget::Soft { .. } => LINK_TYPE_SOFT,
            LinkTarget::External { .. } => LINK_TYPE_EXTERNAL,
            LinkTarget::UserDefined { link_type, .. } => *link_type,
        };

        // Always store link type so that soft links are correctly identified.
        let mut flags: u8 = name_len_code & FLAG_NAME_LEN_MASK;
        flags |= FLAG_LINK_TYPE; // always include link type for clarity
        flags |= FLAG_CHARSET; // always include charset (UTF-8)
        if self.creation_order.is_some() {
            flags |= FLAG_CREATION_ORDER;
        }

        let mut buf = Vec::with_capacity(32);
        buf.push(VERSION);
        buf.push(flags);

        // link type
        buf.push(link_type);

        // creation order, before the charset byte (`H5O__link_encode`)
        if let Some(corder) = self.creation_order {
            buf.extend_from_slice(&corder.to_le_bytes());
        }

        // charset: 1 = UTF-8
        buf.push(1u8);

        // name length
        match name_len_size {
            1 => buf.push(name_len as u8),
            2 => buf.extend_from_slice(&(name_len as u16).to_le_bytes()),
            4 => buf.extend_from_slice(&(name_len as u32).to_le_bytes()),
            _ => buf.extend_from_slice(&(name_len as u64).to_le_bytes()),
        }

        // name
        buf.extend_from_slice(name_bytes);

        // link info
        match &self.target {
            LinkTarget::Hard { address } => {
                let sa = ctx.sizeof_addr as usize;
                buf.extend_from_slice(&address.to_le_bytes()[..sa]);
            }
            LinkTarget::Soft { target } => {
                let tbytes = target.as_bytes();
                buf.extend_from_slice(&(tbytes.len() as u16).to_le_bytes());
                buf.extend_from_slice(tbytes);
            }
            LinkTarget::External { file, path } => {
                let udata = encode_external_udata(file, path);
                buf.extend_from_slice(&(udata.len() as u16).to_le_bytes());
                buf.extend_from_slice(&udata);
            }
            LinkTarget::UserDefined { udata, .. } => {
                buf.extend_from_slice(&(udata.len() as u16).to_le_bytes());
                buf.extend_from_slice(udata);
            }
        }

        buf
    }

    // ------------------------------------------------------------------ decode

    pub fn decode(buf: &[u8], ctx: &FormatContext) -> FormatResult<(Self, usize)> {
        if buf.len() < 2 {
            return Err(FormatError::BufferTooShort {
                needed: 2,
                available: buf.len(),
            });
        }

        let version = buf[0];
        if version != VERSION {
            return Err(FormatError::InvalidVersion(version));
        }

        let flags = buf[1];
        let name_len_code = flags & FLAG_NAME_LEN_MASK;
        let has_creation_order = (flags & FLAG_CREATION_ORDER) != 0;
        let has_link_type = (flags & FLAG_LINK_TYPE) != 0;
        let has_charset = (flags & FLAG_CHARSET) != 0;

        let mut pos = 2;

        // link type
        let link_type = if has_link_type {
            check_len(buf, pos, 1)?;
            let lt = buf[pos];
            pos += 1;
            lt
        } else {
            LINK_TYPE_HARD // default
        };

        // creation order — an 8-byte signed integer (H5Olink.c INT64DECODE),
        // not 4.
        let creation_order = if has_creation_order {
            check_len(buf, pos, 8)?;
            let v = i64::from_le_bytes(buf[pos..pos + 8].try_into().unwrap());
            pos += 8;
            Some(v)
        } else {
            None
        };

        // charset
        if has_charset {
            check_len(buf, pos, 1)?;
            // skip charset byte
            pos += 1;
        }

        // name length
        let name_len_size: usize = match name_len_code {
            0 => 1,
            1 => 2,
            2 => 4,
            _ => 8,
        };
        check_len(buf, pos, name_len_size)?;
        let name_len = read_uint(&buf[pos..], name_len_size) as usize;
        pos += name_len_size;

        // name
        check_len(buf, pos, name_len)?;
        let name = std::str::from_utf8(&buf[pos..pos + name_len])
            .map_err(|e| FormatError::InvalidData(format!("invalid UTF-8 link name: {}", e)))?
            .to_string();
        pos += name_len;

        // target
        let target = match link_type {
            LINK_TYPE_HARD => {
                let sa = ctx.sizeof_addr as usize;
                check_len(buf, pos, sa)?;
                let address = read_uint(&buf[pos..], sa);
                pos += sa;
                LinkTarget::Hard { address }
            }
            LINK_TYPE_SOFT => {
                check_len(buf, pos, 2)?;
                let tlen = u16::from_le_bytes([buf[pos], buf[pos + 1]]) as usize;
                pos += 2;
                check_len(buf, pos, tlen)?;
                let target = std::str::from_utf8(&buf[pos..pos + tlen])
                    .map_err(|e| {
                        FormatError::InvalidData(format!("invalid UTF-8 soft link target: {}", e))
                    })?
                    .to_string();
                pos += tlen;
                LinkTarget::Soft { target }
            }
            // User-defined links (64..=255) all carry a u16-prefixed opaque
            // value; only the external-link class is interpreted here. A type
            // below the user-defined range is not a link libhdf5 would have
            // written (`H5O__link_decode` rejects it too).
            ud if ud >= LINK_TYPE_EXTERNAL => {
                check_len(buf, pos, 2)?;
                let ulen = u16::from_le_bytes([buf[pos], buf[pos + 1]]) as usize;
                pos += 2;
                check_len(buf, pos, ulen)?;
                let udata = &buf[pos..pos + ulen];
                pos += ulen;
                if ud == LINK_TYPE_EXTERNAL {
                    let (file, path) = decode_external_udata(udata)?;
                    LinkTarget::External { file, path }
                } else {
                    LinkTarget::UserDefined {
                        link_type: ud,
                        udata: udata.to_vec(),
                    }
                }
            }
            other => {
                return Err(FormatError::InvalidData(format!(
                    "unknown link type {}",
                    other
                )));
            }
        };

        Ok((
            Self {
                name,
                target,
                creation_order,
            },
            pos,
        ))
    }
}

// ========================================================================= helpers

/// Strip duplicate and trailing slashes from an object path, the way
/// `H5G_normalize` does before `H5Lcreate_external` stores it.
///
/// The stored value is what a reader gets back from `H5Lget_val`, so a link
/// this crate writes and one libhdf5 writes from the same arguments have to
/// agree here or the two files differ in a field a comparison reports.
pub(crate) fn normalize_object_path(path: &str) -> String {
    let mut out = String::with_capacity(path.len());
    let mut last_slash = false;
    for c in path.chars() {
        if c == '/' && last_slash {
            continue;
        }
        last_slash = c == '/';
        out.push(c);
    }
    // The root path is the one trailing slash that stays.
    if out.len() > 1 && out.ends_with('/') {
        out.pop();
    }
    out
}

/// Encode the external-link value: `(version << 4) | flags`, then the
/// NUL-terminated file name and the NUL-terminated object path (`H5L.c`,
/// `H5L__create_ud` for `H5L_TYPE_EXTERNAL`).
fn encode_external_udata(file: &str, path: &str) -> Vec<u8> {
    let mut udata = Vec::with_capacity(1 + file.len() + path.len() + 2);
    udata.push(EXT_VERSION << 4);
    udata.extend_from_slice(file.as_bytes());
    udata.push(0);
    udata.extend_from_slice(path.as_bytes());
    udata.push(0);
    udata
}

/// Decode the external-link value written by `encode_external_udata`.
/// libhdf5 rejects a value shorter than 3 bytes, a version above
/// `H5L_EXT_VERSION`, and any flag bit set (`H5L_EXT_FLAGS_ALL` is 0).
fn decode_external_udata(udata: &[u8]) -> FormatResult<(String, String)> {
    if udata.len() < 3 {
        return Err(FormatError::InvalidData(format!(
            "external link value is {} bytes, below the 3-byte minimum",
            udata.len()
        )));
    }
    let version = udata[0] >> 4;
    let flags = udata[0] & 0x0f;
    if version != EXT_VERSION {
        return Err(FormatError::InvalidVersion(version));
    }
    if flags != 0 {
        return Err(FormatError::InvalidData(format!(
            "external link flags {flags:#x} are not recognized"
        )));
    }
    let body = &udata[1..];
    let split = body.iter().position(|&b| b == 0).ok_or_else(|| {
        FormatError::InvalidData("external link file name is not NUL-terminated".into())
    })?;
    let file = str_from_utf8(&body[..split], "external link file name")?;
    let rest = &body[split + 1..];
    // The object path's own NUL terminator is present in every file libhdf5
    // writes; tolerate its absence by taking the remainder, as the C traverse
    // path does once it has the file name.
    let end = rest.iter().position(|&b| b == 0).unwrap_or(rest.len());
    let path = str_from_utf8(&rest[..end], "external link object path")?;
    Ok((file, path))
}

fn str_from_utf8(bytes: &[u8], what: &str) -> FormatResult<String> {
    std::str::from_utf8(bytes)
        .map(|s| s.to_string())
        .map_err(|e| FormatError::InvalidData(format!("invalid UTF-8 {what}: {e}")))
}

fn check_len(buf: &[u8], pos: usize, need: usize) -> FormatResult<()> {
    // `need` can be a file-derived length up to 8 bytes wide; a checked add
    // ensures `pos + need` cannot wrap to a small value that spuriously
    // passes the bound check (and then panics a slice in the caller).
    match pos.checked_add(need) {
        Some(end) if end <= buf.len() => Ok(()),
        _ => Err(FormatError::BufferTooShort {
            needed: pos.saturating_add(need),
            available: buf.len(),
        }),
    }
}

/// Minimum number of bytes (1, 2, 4, or 8) to represent `v`.
fn min_bytes_for_value(v: u64) -> usize {
    if v <= u8::MAX as u64 {
        1
    } else if v <= u16::MAX as u64 {
        2
    } else if v <= u32::MAX as u64 {
        4
    } else {
        8
    }
}

// ======================================================================= tests

#[cfg(test)]
mod tests {
    use super::*;

    fn ctx8() -> FormatContext {
        FormatContext {
            sizeof_addr: 8,
            sizeof_size: 8,
        }
    }

    fn ctx4() -> FormatContext {
        FormatContext {
            sizeof_addr: 4,
            sizeof_size: 4,
        }
    }

    #[test]
    fn roundtrip_hard_link() {
        let msg = LinkMessage::hard("dataset1", 0x1000);
        let encoded = msg.encode(&ctx8());
        let (decoded, consumed) = LinkMessage::decode(&encoded, &ctx8()).unwrap();
        assert_eq!(consumed, encoded.len());
        assert_eq!(decoded, msg);
    }

    #[test]
    fn roundtrip_hard_link_ctx4() {
        let msg = LinkMessage::hard("grp", 0x2000);
        let encoded = msg.encode(&ctx4());
        let (decoded, consumed) = LinkMessage::decode(&encoded, &ctx4()).unwrap();
        assert_eq!(consumed, encoded.len());
        assert_eq!(decoded, msg);
    }

    #[test]
    fn roundtrip_soft_link() {
        let msg = LinkMessage::soft("alias", "/group/dataset");
        let encoded = msg.encode(&ctx8());
        let (decoded, consumed) = LinkMessage::decode(&encoded, &ctx8()).unwrap();
        assert_eq!(consumed, encoded.len());
        assert_eq!(decoded, msg);
    }

    #[test]
    fn roundtrip_empty_name() {
        // edge case: empty name
        let msg = LinkMessage::hard("", 0x100);
        let encoded = msg.encode(&ctx8());
        let (decoded, _) = LinkMessage::decode(&encoded, &ctx8()).unwrap();
        assert_eq!(decoded, msg);
    }

    #[test]
    fn roundtrip_long_name() {
        // name longer than 255 bytes triggers 2-byte name length
        let long_name: String = "a".repeat(300);
        let msg = LinkMessage::hard(&long_name, 0xABCD);
        let encoded = msg.encode(&ctx8());
        let (decoded, consumed) = LinkMessage::decode(&encoded, &ctx8()).unwrap();
        assert_eq!(consumed, encoded.len());
        assert_eq!(decoded, msg);
    }

    #[test]
    fn roundtrip_unicode_name() {
        let msg = LinkMessage::hard("日本語データ", 0x4000);
        let encoded = msg.encode(&ctx8());
        let (decoded, _) = LinkMessage::decode(&encoded, &ctx8()).unwrap();
        assert_eq!(decoded, msg);
    }

    #[test]
    fn roundtrip_external_link() {
        let msg = LinkMessage::external("ext", "sibling.h5", "/payload");
        let encoded = msg.encode(&ctx8());
        let (decoded, consumed) = LinkMessage::decode(&encoded, &ctx8()).unwrap();
        assert_eq!(consumed, encoded.len());
        assert_eq!(decoded, msg);
    }

    /// The exact bytes h5py wrote for `f['ext'] = h5py.ExternalLink(...)`:
    /// version 1, flags 0x08 (link type present, 1-byte name length), type 64,
    /// then the udata (version/flags byte, NUL-terminated file, NUL-terminated
    /// path). Rejecting this message dropped the link from the listing.
    #[test]
    fn decode_h5py_external_link_bytes() {
        let mut buf = vec![1u8, 0x08, 64, 3];
        buf.extend_from_slice(b"ext");
        let udata = {
            let mut u = vec![0u8];
            u.extend_from_slice(b"link_external_ext.h5\0");
            u.extend_from_slice(b"/payload\0");
            u
        };
        assert_eq!(udata.len(), 31);
        buf.extend_from_slice(&(udata.len() as u16).to_le_bytes());
        buf.extend_from_slice(&udata);

        let (decoded, consumed) = LinkMessage::decode(&buf, &ctx8()).unwrap();
        assert_eq!(consumed, buf.len());
        assert_eq!(decoded.name, "ext");
        assert_eq!(
            decoded.target,
            LinkTarget::External {
                file: "link_external_ext.h5".into(),
                path: "/payload".into(),
            }
        );
    }

    /// The other half of [`decode_h5py_external_link_bytes`]: the value this
    /// encoder produces for the same arguments is the byte string
    /// `H5Lcreate_external` builds. Only the value is compared — the message
    /// envelope around it differs from libhdf5's by the charset byte this
    /// encoder always writes, which libhdf5 reads either way.
    ///
    /// [`decode_h5py_external_link_bytes`]: self::tests::decode_h5py_external_link_bytes
    #[test]
    fn encoded_external_value_is_the_byte_string_h5lcreate_external_builds() {
        let encoded =
            LinkMessage::external("ext", "link_external_ext.h5", "/payload").encode(&ctx8());
        let mut want = vec![0u8];
        want.extend_from_slice(b"link_external_ext.h5\0");
        want.extend_from_slice(b"/payload\0");
        let len_at = encoded.len() - want.len() - 2;
        assert_eq!(
            u16::from_le_bytes([encoded[len_at], encoded[len_at + 1]]) as usize,
            want.len()
        );
        assert_eq!(&encoded[len_at + 2..], &want[..]);
    }

    /// A user-defined class other than the external link keeps its value
    /// verbatim so the link still has a name and still appears in a listing.
    #[test]
    fn roundtrip_unregistered_user_defined_link() {
        let msg = LinkMessage {
            name: "ud".into(),
            target: LinkTarget::UserDefined {
                link_type: 200,
                udata: vec![9, 8, 7],
            },
            creation_order: None,
        };
        let encoded = msg.encode(&ctx8());
        let (decoded, consumed) = LinkMessage::decode(&encoded, &ctx8()).unwrap();
        assert_eq!(consumed, encoded.len());
        assert_eq!(decoded, msg);
    }

    #[test]
    fn decode_unknown_link_type_below_user_defined_range() {
        let buf = [1u8, 0x08, 7, 1, b'x'];
        match LinkMessage::decode(&buf, &ctx8()).unwrap_err() {
            FormatError::InvalidData(ref s) => assert!(s.contains("link type 7"), "{s}"),
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn decode_external_value_too_short() {
        let mut buf = vec![1u8, 0x08, 64, 1, b'e'];
        buf.extend_from_slice(&2u16.to_le_bytes());
        buf.extend_from_slice(&[0u8, 0]);
        match LinkMessage::decode(&buf, &ctx8()).unwrap_err() {
            FormatError::InvalidData(ref s) => assert!(s.contains("3-byte minimum"), "{s}"),
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn decode_external_value_bad_version() {
        let mut buf = vec![1u8, 0x08, 64, 1, b'e'];
        let udata = [0x10u8, b'f', 0, b'/', 0];
        buf.extend_from_slice(&(udata.len() as u16).to_le_bytes());
        buf.extend_from_slice(&udata);
        match LinkMessage::decode(&buf, &ctx8()).unwrap_err() {
            FormatError::InvalidVersion(1) => {}
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn decode_bad_version() {
        let buf = [2u8, 0]; // version 2 unsupported
        let err = LinkMessage::decode(&buf, &ctx8()).unwrap_err();
        match err {
            FormatError::InvalidVersion(2) => {}
            other => panic!("unexpected error: {:?}", other),
        }
    }

    #[test]
    fn decode_buffer_too_short() {
        let buf = [1u8];
        let err = LinkMessage::decode(&buf, &ctx8()).unwrap_err();
        match err {
            FormatError::BufferTooShort { .. } => {}
            other => panic!("unexpected error: {:?}", other),
        }
    }

    #[test]
    fn version_byte() {
        let encoded = LinkMessage::hard("x", 0).encode(&ctx8());
        assert_eq!(encoded[0], 1);
    }

    #[test]
    fn roundtrip_with_creation_order() {
        let msg = LinkMessage::hard("d00", 0x1000).with_creation_order(7);
        let encoded = msg.encode(&ctx8());
        // The flag byte announces it, and the value costs eight bytes.
        assert_eq!(encoded[1] & FLAG_CREATION_ORDER, FLAG_CREATION_ORDER);
        assert_eq!(
            encoded.len(),
            LinkMessage::hard("d00", 0x1000).encode(&ctx8()).len() + 8
        );
        let (decoded, consumed) = LinkMessage::decode(&encoded, &ctx8()).unwrap();
        assert_eq!(consumed, encoded.len());
        assert_eq!(decoded, msg);
        assert_eq!(decoded.creation_order, Some(7));
    }

    /// The creation order sits between the link type and the charset byte, so
    /// a decoder that skipped the wrong span would misread the name.
    #[test]
    fn creation_order_precedes_the_charset_byte() {
        let msg = LinkMessage::soft("alias", "/orig").with_creation_order(-3);
        let encoded = msg.encode(&ctx8());
        assert_eq!(&encoded[3..11], &(-3i64).to_le_bytes());
        assert_eq!(encoded[11], 1, "charset follows the creation order");
        assert_eq!(LinkMessage::decode(&encoded, &ctx8()).unwrap().0, msg);
    }

    /// `H5G_normalize`: duplicate slashes collapse, one trailing slash goes,
    /// and the root path keeps its only character.
    #[test]
    fn object_paths_normalize_like_h5g_normalize() {
        assert_eq!(normalize_object_path("/payload"), "/payload");
        assert_eq!(normalize_object_path("//a///b"), "/a/b");
        assert_eq!(normalize_object_path("/a/b/"), "/a/b");
        assert_eq!(normalize_object_path("/a/b//"), "/a/b");
        assert_eq!(normalize_object_path("/"), "/");
        assert_eq!(normalize_object_path("//"), "/");
        assert_eq!(normalize_object_path("a/b"), "a/b");
        assert_eq!(normalize_object_path(""), "");
    }

    #[test]
    fn min_bytes_for_value_checks() {
        assert_eq!(min_bytes_for_value(0), 1);
        assert_eq!(min_bytes_for_value(255), 1);
        assert_eq!(min_bytes_for_value(256), 2);
        assert_eq!(min_bytes_for_value(65535), 2);
        assert_eq!(min_bytes_for_value(65536), 4);
        assert_eq!(min_bytes_for_value(u32::MAX as u64), 4);
        assert_eq!(min_bytes_for_value(u32::MAX as u64 + 1), 8);
    }
}
