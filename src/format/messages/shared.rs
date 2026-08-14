//! Shared object header messages (`H5O_shared_t`).
//!
//! A message whose header flags carry [`MSG_FLAG_SHARED`] does not hold its
//! own body. What follows the message header is a *reference* to where the
//! body really lives — another object header, or the file's shared-message
//! heap — and decoding those bytes as if they were the body is not a decode
//! failure but a wrong answer: the first byte of a v2 shared message is the
//! version `2`, which a datatype decoder reads as version 0, class 2.
//!
//! This is how a dataset built on a committed (named) datatype stores its
//! type: one `H5O_SHARE_TYPE_COMMITTED` reference to the object header of the
//! datatype it shares.
//!
//! The reference itself is decoded by
//! [`SharedMessagePointer`](crate::format::sohm::SharedMessagePointer), which
//! the object-header reader resolves through for both storage types; this
//! module is only the flag that says a body is one.

/// `H5O_MSG_FLAG_SHARED` — the message body is a reference, not the message.
pub const MSG_FLAG_SHARED: u8 = 0x02;
