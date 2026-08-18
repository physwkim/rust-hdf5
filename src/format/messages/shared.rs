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
/// Defined with the rest of the message-envelope flags and re-exported here,
/// where the readers that consult it look for it.
pub use super::MSG_FLAG_SHARED;

/// How an object header stores one message, as its flags byte says.
///
/// The three cases are what `H5O__debug_real` prints as the message's flags
/// and, for a shared one, what `H5O__shared_debug` prints beneath it
/// (H5Odbg.c:409-455, H5Oshared.c:682-706).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MessageStorage {
    /// The body is the message and no index names it.
    Private,
    /// [`MSG_FLAG_SHAREABLE`](super::MSG_FLAG_SHAREABLE): the body is still
    /// the message, and a shared-message index holds an `H5SM_IN_OH` record
    /// naming it here — the first copy of a share-in-object-header class
    /// (H5SM.c:1400-1417).
    Shareable,
    /// [`MSG_FLAG_SHARED`]: the body is a pointer to where the message
    /// really lives.
    Shared(crate::format::sohm::SharedLocation),
}
