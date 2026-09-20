//! Bounded lifecycle metadata frames carried by the authenticated bootstrap UDS.

use std::io;

pub const LIFECYCLE_HEADER_BYTES: usize = 20;
pub const MAX_LIFECYCLE_PAYLOAD: usize = 64 * 1024 * 1024;
const MAGIC: u32 = 0x4f52_424c;
const VERSION: u16 = 3;

#[derive(Clone, Copy, Debug)]
#[repr(u16)]
pub enum LifecycleCommand {
    Health = 1,
    Register = 2,
    Unregister = 3,
    Session = 4,
}

impl TryFrom<u16> for LifecycleCommand {
    type Error = io::Error;

    fn try_from(value: u16) -> io::Result<Self> {
        match value {
            1 => Ok(Self::Health),
            2 => Ok(Self::Register),
            3 => Ok(Self::Unregister),
            4 => Ok(Self::Session),
            _ => Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "unknown lifecycle command",
            )),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct LifecycleHeader {
    pub code: u16,
    pub epoch: u64,
    pub payload_len: usize,
}

impl LifecycleHeader {
    pub fn encode(self) -> io::Result<[u8; LIFECYCLE_HEADER_BYTES]> {
        if self.payload_len > MAX_LIFECYCLE_PAYLOAD {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "lifecycle metadata exceeds limit",
            ));
        }
        let mut bytes = [0; LIFECYCLE_HEADER_BYTES];
        bytes[..4].copy_from_slice(&MAGIC.to_le_bytes());
        bytes[4..6].copy_from_slice(&VERSION.to_le_bytes());
        bytes[6..8].copy_from_slice(&self.code.to_le_bytes());
        bytes[8..16].copy_from_slice(&self.epoch.to_le_bytes());
        bytes[16..20].copy_from_slice(&(self.payload_len as u32).to_le_bytes());
        Ok(bytes)
    }

    pub fn decode(bytes: [u8; LIFECYCLE_HEADER_BYTES]) -> io::Result<Self> {
        if bytes[..4] != MAGIC.to_le_bytes() || bytes[4..6] != VERSION.to_le_bytes() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "incompatible lifecycle protocol",
            ));
        }
        let header = Self {
            code: u16::from_le_bytes(bytes[6..8].try_into().expect("fixed header")),
            epoch: u64::from_le_bytes(bytes[8..16].try_into().expect("fixed header")),
            payload_len: u32::from_le_bytes(bytes[16..20].try_into().expect("fixed header"))
                as usize,
        };
        if header.payload_len > MAX_LIFECYCLE_PAYLOAD {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "lifecycle metadata exceeds limit",
            ));
        }
        Ok(header)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_incompatible_and_unbounded_frames_before_allocating() {
        let header = LifecycleHeader {
            code: 2,
            epoch: 42,
            payload_len: 512,
        };
        let mut bytes = header.encode().unwrap();
        assert_eq!(LifecycleHeader::decode(bytes).unwrap().epoch, 42);
        bytes[16..20].copy_from_slice(&u32::MAX.to_le_bytes());
        assert!(LifecycleHeader::decode(bytes).is_err());
        bytes = header.encode().unwrap();
        bytes[4] = 99;
        assert!(LifecycleHeader::decode(bytes).is_err());
    }
}
