pub mod attr_info;
pub mod attribute;
pub mod continuation;
pub mod data_layout;
pub mod dataspace;
pub mod datatype;
pub mod external_file_list;
pub mod fill_value;
pub mod filter;
pub mod group_info;
pub mod link;
pub mod link_info;
pub mod mod_time;
pub mod shared;
pub mod superblock_ext;
pub mod virtual_mapping;

// Message type IDs
/// `H5O_NULL_ID` — space in an object header that holds no message
/// (H5Oprivate.h:206).
pub const MSG_NULL: u8 = 0x00;
pub const MSG_DATASPACE: u8 = 0x01;
pub const MSG_LINK_INFO: u8 = 0x02;
pub const MSG_DATATYPE: u8 = 0x03;
pub const MSG_FILL_VALUE_OLD: u8 = 0x04;
pub const MSG_FILL_VALUE: u8 = 0x05;
pub const MSG_LINK: u8 = 0x06;
pub const MSG_EXTERNAL_FILE_LIST: u8 = 0x07;
pub const MSG_DATA_LAYOUT: u8 = 0x08;
pub const MSG_GROUP_INFO: u8 = 0x0A;
pub const MSG_FILTER_PIPELINE: u8 = 0x0B;
pub const MSG_ATTRIBUTE: u8 = 0x0C;
pub const MSG_MOD_TIME_OLD: u8 = 0x0E;
pub const MSG_SHARED_MESSAGE_TABLE: u8 = 0x0F;
pub const MSG_OBJ_HEADER_CONTINUATION: u8 = 0x10;
pub const MSG_SYMBOL_TABLE: u8 = 0x11;
pub const MSG_MOD_TIME: u8 = 0x12;
pub const MSG_BTREE_K: u8 = 0x13;
pub const MSG_DRIVER_INFO: u8 = 0x14;
pub const MSG_ATTR_INFO: u8 = 0x15;
pub const MSG_OBJ_REF_COUNT: u8 = 0x16;
pub const MSG_FILE_SPACE_INFO: u8 = 0x17;

// Object header message flags (the `flags` byte of the message envelope)
/// `H5O_MSG_FLAG_CONSTANT`: the message will not change once written, so a
/// reader may cache it. libhdf5 sets it on a datatype message.
pub const MSG_FLAG_CONSTANT: u8 = 0x01;
/// `H5O_MSG_FLAG_SHARED`: the message body is not the message, it is a
/// pointer to where the body is really stored (SOHM heap or another object
/// header). Decoding the body literally silently yields nonsense.
pub const MSG_FLAG_SHARED: u8 = 0x02;
/// `H5O_MSG_FLAG_DONTSHARE`: the message must never be moved to the shared
/// message heap.
pub const MSG_FLAG_DONTSHARE: u8 = 0x04;
/// `H5O_MSG_FLAG_SHAREABLE`: the message body *is* the message, and a
/// shared-message index also holds a record naming it here
/// (`H5SM_IN_OH`). `H5SM__write_mesg` leaves the first copy of a
/// share-in-object-header class where it was written and only moves the body
/// to the heap when a second object wants it (H5SM.c:1400-1417), so a reader
/// must take the body literally — unlike [`MSG_FLAG_SHARED`], this flag says
/// nothing about the bytes that follow.
pub const MSG_FLAG_SHAREABLE: u8 = 0x40;
/// `H5O_MSG_FLAG_MARK_IF_UNKNOWN`: a writer that does not understand this
/// message must mark it as one it may have invalidated, so a later reader
/// knows the object was edited by a library that could not maintain it.
pub const MSG_FLAG_MARK_IF_UNKNOWN: u8 = 0x10;
