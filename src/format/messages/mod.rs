pub mod attribute;
pub mod continuation;
pub mod data_layout;
pub mod dataspace;
pub mod datatype;
pub mod fill_value;
pub mod filter;
pub mod group_info;
pub mod link;
pub mod link_info;
pub mod mod_time;

// Message type IDs
pub const MSG_DATASPACE: u8 = 0x01;
pub const MSG_LINK_INFO: u8 = 0x02;
pub const MSG_DATATYPE: u8 = 0x03;
pub const MSG_FILL_VALUE_OLD: u8 = 0x04;
pub const MSG_FILL_VALUE: u8 = 0x05;
pub const MSG_LINK: u8 = 0x06;
pub const MSG_DATA_LAYOUT: u8 = 0x08;
pub const MSG_GROUP_INFO: u8 = 0x0A;
pub const MSG_FILTER_PIPELINE: u8 = 0x0B;
pub const MSG_ATTRIBUTE: u8 = 0x0C;
pub const MSG_OBJ_HEADER_CONTINUATION: u8 = 0x10;
pub const MSG_SYMBOL_TABLE: u8 = 0x11;
pub const MSG_MOD_TIME: u8 = 0x12;
pub const MSG_BTREE_K: u8 = 0x13;
pub const MSG_ATTR_INFO: u8 = 0x15;
pub const MSG_OBJ_REF_COUNT: u8 = 0x16;
