//! Peers: what this rank exports and what it maps from the others.
//!
//! A manifest with a `topology` is SPMD: every rank loads it with its own
//! [`Topology`](crate::Topology), its index in each group. `export`
//! buffers and every state are virtual-memory allocations every rank can
//! map (a fabric handle where the device offers one, the allocation
//! handle itself otherwise: [`PeerHandle`]); [`Runtime::export_handles`]
//! hands this rank's out, the caller carries them to the group (within
//! this process for a tray; over whatever channel it has for fabric
//! handles), and [`Runtime::import_peers`] maps the other ranks'
//! allocations into this address space and writes the group's addresses
//! into each `peer` buffer, the array a kernel indexes by rank. A remote allocation is
//! mapped once however many peer buffers name it, and the mapping lives
//! as long as the addresses do.
//!
//! Nothing launches with a peer array still holding zeros:
//! [`Runtime::require_peers`] guards every program issue.

use std::collections::BTreeMap;

use crate::device::{self, Mapped, PeerHandle};
use crate::error::bail;
use crate::{Error, Result, Runtime};

/// A `peer` buffer waiting for (or holding) its group's addresses.
pub(crate) struct PeerSlot {
    pub(crate) of: String,
    pub(crate) group: String,
    pub(crate) filled: bool,
}

impl Runtime {
    /// This rank's index in a topology group.
    pub fn rank(&self, group: &str) -> Option<u64> {
        self.ranks.get(group).copied()
    }

    /// Map another process's allocation into this context: a weight
    /// cache's bucket, whose fabric handle the daemon published. The
    /// mapping lives as long as the [`Mapped`] does.
    pub fn map(&self, handle: &PeerHandle, what: &str) -> Result<Mapped> {
        self.ctx.bind_to_thread()?;
        Ok(Mapped::new(device::import(&self.stream, self.gpu as i32, handle, what)?))
    }

    /// A handle for every `export` buffer and every state, by name: what
    /// the other ranks pass to [`Runtime::import_peers`].
    pub fn export_handles(&self) -> Result<BTreeMap<String, PeerHandle>> {
        self.ctx.bind_to_thread()?;
        let mut out = BTreeMap::new();
        for (name, b) in &self.buffers {
            if self.manifest.buffers[name].export {
                let h = b.export()?.ok_or_else(|| {
                    Error::Cuda(format!("buffer `{name}`: exported but not a virtual-memory allocation"))
                })?;
                out.insert(name.clone(), h);
            }
        }
        for (name, s) in &self.states {
            if let Some(h) = s.export()? {
                out.insert(name.clone(), h);
            }
        }
        Ok(out)
    }

    /// The `peer` buffers not yet filled, by name.
    pub fn pending_peers(&self) -> Vec<&str> {
        self.peers.iter().filter(|(_, p)| !p.filled).map(|(n, _)| n.as_str()).collect()
    }

    /// Map every group member's exported allocations and fill the group's
    /// `peer` buffers with their addresses. `members[i]` is what rank `i`'s
    /// [`Runtime::export_handles`] returned (this rank's own entry may be
    /// anything: its local addresses are used). Synchronous.
    pub fn import_peers(&mut self, group: &str, members: &[BTreeMap<String, PeerHandle>]) -> Result<()> {
        self.ctx.bind_to_thread()?;
        let Some(&me) = self.ranks.get(group) else {
            bail!(Api, "no topology group `{group}`");
        };
        let size = self.manifest.group_size(group).unwrap_or(0);
        if members.len() as u64 != size {
            bail!(Api, "group `{group}` has {size} members, got handles for {}", members.len());
        }
        let stream = self.stream.clone();
        let mut mapped: BTreeMap<(usize, String), u64> = BTreeMap::new();
        let names: Vec<String> = self.peers.iter().filter(|(_, p)| p.group == group).map(|(n, _)| n.clone()).collect();
        for name in names {
            let of = self.peers[&name].of.clone();
            let own = self.buffers.get(&of).or_else(|| self.states.get(&of)).ok_or_else(|| {
                Error::Manifest(format!("peer buffer `{name}`: `of` `{of}` is neither a buffer nor a state"))
            })?;
            let own_bytes = own.export()?.map(|h| h.bytes()).unwrap_or(own.bytes);
            let own_ptr = own.ptr;
            let mut addrs = Vec::with_capacity(size as usize);
            for (i, m) in members.iter().enumerate() {
                if i as u64 == me {
                    addrs.push(own_ptr);
                    continue;
                }
                let key = (i, of.clone());
                let ptr = match mapped.get(&key) {
                    Some(&p) => p,
                    None => {
                        let Some(h) = m.get(&of) else {
                            bail!(Api, "group `{group}` rank {i}: no handle for `{of}`");
                        };
                        if h.bytes() != own_bytes {
                            bail!(
                                Api,
                                "group `{group}` rank {i}: `{of}` is {} bytes there, {own_bytes} here",
                                h.bytes()
                            );
                        }
                        let buf =
                            device::import(&stream, self.gpu as i32, h, &format!("group `{group}` rank {i} `{of}`"))?;
                        let p = buf.ptr;
                        self.imports.push(buf);
                        mapped.insert(key, p);
                        p
                    }
                };
                addrs.push(ptr);
            }
            let bytes: Vec<u8> = addrs.iter().flat_map(|a| a.to_le_bytes()).collect();
            let dst = self.buffers.get_mut(&name).unwrap();
            if bytes.len() as u64 != dst.bytes {
                bail!(
                    Manifest,
                    "peer buffer `{name}`: {} bytes for {size} addresses, allocated {}",
                    bytes.len(),
                    dst.bytes
                );
            }
            stream.memcpy_htod(&bytes, dst)?;
            self.peers.get_mut(&name).unwrap().filled = true;
            tracing::info!("peer buffer `{name}`: {size} addresses of `{of}` over group `{group}`");
        }
        stream.synchronize()?;
        Ok(())
    }

    /// Nothing launches with a peer array still holding zeros.
    pub(crate) fn require_peers(&self) -> Result<()> {
        let pending = self.pending_peers();
        if pending.is_empty() {
            return Ok(());
        }
        bail!(Api, "peer buffers not imported yet: {}", pending.join(", "))
    }
}
