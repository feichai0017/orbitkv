use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Duration;

use etcd_client::{Client, EventType, GetOptions, KeyValue, WatchOptions};
use orbitkv_catalog::{GlobalIndex, IndexUpdate, MembershipView};
use orbitkv_state::InventoryRecord;
use prost::Message;

use super::publish::{MAX_RECORD_BYTES, Progress, record_key};
use super::{BootstrapError, MAX_MEMBERS, MEMBER_BYTES, Member, cluster_id, parse_label, rpc};

const FORMAT: &[u8] = b"orbitkv/global-index/v2";

pub(super) enum FollowError {
    Disconnected(String),
    Rebuild(String),
}
impl From<etcd_client::Error> for FollowError {
    fn from(error: etcd_client::Error) -> Self {
        match &error {
            etcd_client::Error::IoError(_) | etcd_client::Error::TransportError(_) => {
                Self::Disconnected(error.to_string())
            }
            etcd_client::Error::GRpcStatus(status) if disconnected_status(status) => {
                Self::Disconnected(error.to_string())
            }
            _ => Self::Rebuild(error.to_string()),
        }
    }
}

fn disconnected_status(status: &tonic::Status) -> bool {
    if matches!(
        status.code(),
        tonic::Code::Unavailable | tonic::Code::Cancelled | tonic::Code::DeadlineExceeded
    ) {
        return true;
    }
    // tonic maps an HTTP/2 connection reset while reading a Watch body to
    // Unknown. Resume from the last applied revision; etcd requests a rebuild
    // if that history was compacted while the connection was unavailable.
    status.code() == tonic::Code::Unknown
        && (status.message().contains("h2 protocol error")
            || status.message().contains("transport error"))
}

impl From<String> for FollowError {
    fn from(error: String) -> Self {
        Self::Disconnected(error)
    }
}

pub(super) async fn run(
    mut client: Client,
    prefix: String,
    expected_cluster: u64,
    registration: Member,
    view: Arc<MembershipView>,
    index: Arc<GlobalIndex>,
) {
    let mut snapshot = None;
    while view.registration_valid() {
        if snapshot.is_none() {
            index.reset();
            view.invalidate_snapshot();
            match bootstrap(
                &mut client,
                &prefix,
                expected_cluster,
                &registration,
                &view,
                &index,
            )
            .await
            {
                Ok(state) => snapshot = Some(state),
                Err(error) => {
                    index.reset();
                    log::warn!("Global index snapshot needs repair: {error}");
                    tokio::time::sleep(Duration::from_secs(1)).await;
                    continue;
                }
            }
        }
        let (members, revision) = snapshot.as_mut().expect("completed metadata snapshot");
        match follow(
            &mut client,
            &prefix,
            expected_cluster,
            &registration,
            &view,
            &index,
            members,
            revision,
        )
        .await
        {
            Ok(()) => break,
            Err(FollowError::Disconnected(error)) => {
                log::warn!("Metadata Watch will resume at revision {revision}: {error}");
            }
            Err(FollowError::Rebuild(error)) => {
                log::warn!("Metadata Watch requires a fresh snapshot: {error}");
                index.reset();
                view.invalidate_snapshot();
                snapshot = None;
            }
        }
        tokio::time::sleep(Duration::from_secs(1)).await;
    }
    index.reset();
    view.invalidate_snapshot();
}

pub(super) async fn install_format(
    client: &mut Client,
    prefix: &str,
    expected_cluster: u64,
) -> Result<(), BootstrapError> {
    use etcd_client::{Compare, CompareOp, Txn, TxnOp};
    let key = format!("{prefix}format");
    let response = rpc(client.txn(
        Txn::new()
            .when([Compare::version(key.clone(), CompareOp::Equal, 0)])
            .and_then([TxnOp::put(key.clone(), FORMAT, None)]),
    ))
    .await?;
    if cluster_id(response.header()).map_err(BootstrapError::Rejected)? != expected_cluster {
        return Err(BootstrapError::Rejected(
            "coordinator changed during format registration".into(),
        ));
    }
    let response = rpc(client.get(key, None)).await?;
    if cluster_id(response.header()).map_err(BootstrapError::Rejected)? != expected_cluster
        || response.kvs().len() != 1
        || response.kvs()[0].value() != FORMAT
        || response.kvs()[0].lease() != 0
    {
        return Err(BootstrapError::Rejected(
            "cluster metadata format differs".into(),
        ));
    }
    Ok(())
}

pub(super) async fn bootstrap(
    client: &mut Client,
    prefix: &str,
    expected_cluster: u64,
    registration: &Member,
    view: &MembershipView,
    index: &GlobalIndex,
) -> Result<(BTreeMap<String, Member>, i64), String> {
    // A page can contain 128 maximum-sized records plus keys/protobuf overhead.
    // Keep snapshot decoding bounded while allowing every valid record shape.
    let mut pages = client
        .kv_client()
        .max_decoding_message_size(128 * (MAX_RECORD_BYTES + 1024));
    let mut members = BTreeMap::new();
    let mut start = prefix.as_bytes().to_vec();
    let mut end = start.clone();
    *end.last_mut().ok_or("empty metadata prefix")? += 1;
    let mut revision = 0;
    let mut format_seen = false;
    loop {
        let response = rpc(pages.get(
            start.clone(),
            Some(
                GetOptions::new()
                    .with_range(end.clone())
                    .with_revision(revision)
                    .with_limit(128),
            ),
        ))
        .await?;
        if cluster_id(response.header())? != expected_cluster {
            view.fence();
            return Err("coordinator changed during snapshot".into());
        }
        if revision == 0 {
            revision = response
                .header()
                .ok_or("missing snapshot revision")?
                .revision();
        }
        let mut updates = Vec::new();
        for kv in response.kvs() {
            decode(
                prefix,
                kv,
                false,
                &mut members,
                &mut updates,
                &mut format_seen,
            )?;
        }
        index.apply(revision, updates)?;
        if !response.more() {
            break;
        }
        start = response
            .kvs()
            .last()
            .ok_or("empty continuation page")?
            .key()
            .to_vec();
        start.push(0);
        tokio::task::yield_now().await;
    }
    if !format_seen || members.get(&registration.node_id) != Some(registration) {
        view.fence();
        return Err("cluster format or own registration changed".into());
    }
    view.replace_members(
        members
            .values()
            .map(|member| (member.node_id.clone(), member.owner.clone())),
    );
    index.finish_snapshot(revision)?;
    Ok((members, revision))
}

#[allow(
    clippy::too_many_arguments,
    reason = "explicit revision, membership and index owners share one Watch boundary"
)]
pub(super) async fn follow(
    client: &mut Client,
    prefix: &str,
    expected_cluster: u64,
    registration: &Member,
    view: &MembershipView,
    index: &GlobalIndex,
    members: &mut BTreeMap<String, Member>,
    applied: &mut i64,
) -> Result<(), FollowError> {
    // No fragmentation: etcd preserves transaction atomicity. Oversized responses
    // fail the bounded gRPC decoder and are repaired through paginated bootstrap.
    let mut stream = rpc(client.watch(
        prefix,
        Some(
            WatchOptions::new()
                .with_prefix()
                .with_start_revision(*applied + 1)
                .with_prev_key(),
        ),
    ))
    .await?;
    loop {
        if !view.registration_valid() {
            return Ok(());
        }
        let response = match tokio::time::timeout(Duration::from_secs(2), stream.message()).await {
            Ok(response) => response.map_err(FollowError::from)?,
            Err(_) => {
                rpc(stream.request_progress()).await?;
                continue;
            }
        }
        .ok_or_else(|| FollowError::Disconnected("metadata Watch closed".into()))?;
        if cluster_id(response.header())? != expected_cluster {
            view.fence();
            return Ok(());
        }
        if response.canceled() || response.compact_revision() > 0 {
            return Err(FollowError::Rebuild(format!(
                "Watch canceled or compacted: {}",
                response.cancel_reason()
            )));
        }
        if response.created() {
            continue;
        }
        let mut through = *applied;
        let mut updates = Vec::new();
        let mut format_valid = true;
        for event in response.events() {
            let kv = event
                .kv()
                .ok_or_else(|| FollowError::Rebuild("missing event key".into()))?;
            if kv.mod_revision() <= *applied {
                continue;
            }
            through = through.max(kv.mod_revision());
            let deleted = event.event_type() == EventType::Delete;
            let value = if deleted {
                event
                    .prev_kv()
                    .ok_or_else(|| FollowError::Rebuild("delete lacks previous metadata".into()))?
            } else {
                kv
            };
            decode(
                prefix,
                value,
                deleted,
                members,
                &mut updates,
                &mut format_valid,
            )
            .map_err(FollowError::Rebuild)?;
            if !format_valid || members.get(&registration.node_id) != Some(registration) {
                view.fence();
                return Ok(());
            }
        }
        if response.events().is_empty() {
            through = response
                .header()
                .ok_or_else(|| FollowError::Rebuild("missing Watch revision".into()))?
                .revision()
                .max(through);
        }
        if through > *applied {
            view.replace_members(
                members
                    .values()
                    .map(|member| (member.node_id.clone(), member.owner.clone())),
            );
            index
                .apply(through, updates)
                .map_err(FollowError::Rebuild)?;
            *applied = through;
        }
    }
}

fn decode(
    prefix: &str,
    kv: &KeyValue,
    deleted: bool,
    members: &mut BTreeMap<String, Member>,
    updates: &mut Vec<IndexUpdate>,
    format_valid: &mut bool,
) -> Result<(), String> {
    let suffix = kv
        .key_str()
        .map_err(|error| error.to_string())?
        .strip_prefix(prefix)
        .ok_or("metadata key outside cluster")?;
    if kv.value().len() > MAX_RECORD_BYTES {
        return Err("metadata record exceeds byte limit".into());
    }
    if suffix == "format" {
        *format_valid = !deleted && kv.value() == FORMAT && kv.lease() == 0;
        if !*format_valid {
            return Err("cluster metadata format changed".into());
        }
    } else if let Some(node) = suffix.strip_prefix("members/") {
        let member = decode_member(node, kv)?;
        if deleted {
            members.remove(node);
            updates.push(IndexUpdate::RemoveOwner(member.owner.incarnation));
        } else {
            if let Some(old) = members.insert(node.into(), member.clone())
                && old.owner != member.owner
            {
                updates.push(IndexUpdate::RemoveOwner(old.owner.incarnation));
            }
        }
        if members.len() > MAX_MEMBERS {
            return Err("metadata member limit exceeded".into());
        }
    } else if let Some(owner) = suffix.strip_prefix("publishers/") {
        let owner = owner.parse().map_err(|_| "invalid publisher incarnation")?;
        if kv.lease() == 0 {
            return Err("publisher is not leased".into());
        }
        let progress: Progress =
            serde_json::from_slice(kv.value()).map_err(|error| error.to_string())?;
        if deleted {
            updates.push(IndexUpdate::RemoveOwner(owner));
        } else {
            updates.push(IndexUpdate::Publisher {
                owner,
                ready: progress.ready,
            });
        }
    } else if let Some(block) = suffix.strip_prefix("blocks/") {
        let owner = block
            .split('/')
            .next()
            .ok_or("missing block owner")?
            .parse()
            .map_err(|_| "invalid block owner")?;
        if kv.lease() == 0 {
            return Err("block metadata is not leased".into());
        }
        let wire = orbitkv_proto::proto::engine::InventoryRecord::decode(kv.value())
            .map_err(|error| error.to_string())?;
        let mut record: InventoryRecord = wire.into();
        if !record.present || record_key(prefix, owner, &record)?.as_bytes() != kv.key() {
            return Err("block metadata key/value mismatch".into());
        }
        record.present = !deleted;
        updates.push(IndexUpdate::Residency { owner, record });
    } else if let Some(node) = suffix.strip_prefix("epochs/") {
        parse_label(node)?;
        if kv.lease() != 0
            || kv
                .value_str()
                .map_err(|error| error.to_string())?
                .parse::<u64>()
                .is_err()
        {
            return Err("invalid persistent node epoch".into());
        }
    } else {
        return Err("unknown cluster metadata record".into());
    }
    Ok(())
}

fn decode_member(node: &str, kv: &KeyValue) -> Result<Member, String> {
    if kv.value().len() > MEMBER_BYTES || kv.lease() == 0 {
        return Err("member record oversized or unleased".into());
    }
    let mut member: Member =
        serde_json::from_slice(kv.value()).map_err(|error| error.to_string())?;
    member.lease = kv.lease();
    parse_label(node)?;
    let address: std::net::SocketAddr = member
        .owner
        .endpoint
        .parse()
        .map_err(|_| "invalid member endpoint")?;
    if node != member.node_id
        || member.epoch == 0
        || member.owner.incarnation.is_nil()
        || address.ip().is_unspecified()
        || address.port() == 0
    {
        return Err("invalid member identity".into());
    }
    Ok(member)
}

#[cfg(test)]
#[path = "../../tests/unit/cluster/watch.rs"]
mod tests;
