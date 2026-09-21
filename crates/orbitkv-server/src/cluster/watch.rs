use std::collections::BTreeMap;
use std::time::Duration;

use etcd_client::{Client, EventType, GetOptions, KeyValue, WatchOptions};
use orbitkv_catalog::MembershipView;

use super::{MAX_MEMBERS, MEMBER_BYTES, Member, cluster_id, parse_label, rpc};

pub(super) async fn run(
    mut client: Client,
    prefix: String,
    expected_cluster: u64,
    registration: Member,
    view: &MembershipView,
) {
    loop {
        if !view.registration_valid() {
            return;
        }
        let result = async {
            let (members, revision) = snapshot(&mut client, &prefix, expected_cluster).await?;
            let placement_key = format!("{}placement", prefix.trim_end_matches("members/"));
            let response = rpc(client.get(
                placement_key,
                Some(GetOptions::new().with_revision(revision)),
            ))
            .await?;
            if cluster_id(response.header())? != expected_cluster
                || response.kvs().len() != 1
                || !valid_placement(&response.kvs()[0], view)
            {
                view.fence();
                return Err("committed catalog placement changed".into());
            }
            if members.get(&registration.node_id) != Some(&registration) {
                view.fence();
                return Err("own membership registration changed".into());
            }
            view.replace_members(
                members
                    .values()
                    .map(|member| (member.node_id.clone(), member.owner.clone())),
            );
            follow(
                &mut client,
                &prefix,
                expected_cluster,
                &registration,
                view,
                members,
                revision,
            )
            .await
        }
        .await;
        view.invalidate_snapshot();
        if let Err(error) = result {
            log::warn!("Membership watch needs a fresh snapshot: {error}");
        }
        tokio::time::sleep(Duration::from_secs(1)).await;
    }
}

pub(super) async fn snapshot(
    client: &mut Client,
    prefix: &str,
    expected_cluster: u64,
) -> Result<(BTreeMap<String, Member>, i64), String> {
    let mut members = BTreeMap::new();
    let mut start = prefix.as_bytes().to_vec();
    let mut end = start.clone();
    *end.last_mut().ok_or("empty membership prefix")? += 1;
    let mut revision = 0;
    loop {
        let response = rpc(client.get(
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
            return Err("coordinator cluster changed during snapshot".into());
        }
        if revision == 0 {
            revision = response
                .header()
                .ok_or("missing snapshot revision")?
                .revision();
        }
        for kv in response.kvs() {
            let member = decode(prefix, kv)?;
            members.insert(member.node_id.clone(), member);
            if members.len() > MAX_MEMBERS {
                return Err("membership snapshot exceeds member budget".into());
            }
        }
        if !response.more() {
            return Ok((members, revision));
        }
        start = response
            .kvs()
            .last()
            .ok_or("empty continuation page")?
            .key()
            .to_vec();
        start.push(0);
    }
}

pub(super) async fn follow(
    client: &mut Client,
    prefix: &str,
    expected_cluster: u64,
    registration: &Member,
    view: &MembershipView,
    mut members: BTreeMap<String, Member>,
    revision: i64,
) -> Result<(), String> {
    let mut stream = rpc(client.watch(
        prefix.trim_end_matches("members/"),
        Some(
            WatchOptions::new()
                .with_prefix()
                .with_start_revision(revision + 1),
        ),
    ))
    .await?;
    let mut applied = revision;
    loop {
        if !view.registration_valid() {
            return Err("membership registration is no longer valid".into());
        }
        let response =
            match tokio::time::timeout(Duration::from_secs(10), stream.message()).await {
                Ok(response) => response.map_err(|error| error.to_string())?,
                Err(_) => {
                    rpc(stream.request_progress()).await?;
                    rpc(stream.message()).await?
                }
            }
            .ok_or("membership watch closed")?;
        if cluster_id(response.header())? != expected_cluster {
            view.fence();
            return Err("coordinator cluster changed during watch".into());
        }
        if response.canceled() || response.compact_revision() > 0 {
            return Err(format!(
                "watch cancelled or compacted: {}",
                response.cancel_reason()
            ));
        }
        let mut through = applied;
        for event in response.events() {
            let kv = event.kv().ok_or("watch event missing key")?;
            if kv.mod_revision() <= applied {
                continue;
            }
            through = through.max(kv.mod_revision());
            if kv.key() == format!("{}placement", prefix.trim_end_matches("members/")).as_bytes() {
                if event.event_type() == EventType::Delete || !valid_placement(kv, view) {
                    view.fence();
                    return Err("committed catalog placement changed".into());
                }
                continue;
            }
            if !kv.key().starts_with(prefix.as_bytes()) {
                continue;
            }
            match event.event_type() {
                EventType::Put => {
                    let member = decode(prefix, kv)?;
                    members.insert(member.node_id.clone(), member);
                }
                EventType::Delete => {
                    let node = kv
                        .key_str()
                        .map_err(|error| error.to_string())?
                        .strip_prefix(prefix)
                        .ok_or("watch key outside membership prefix")?;
                    members.remove(node);
                }
            }
            if members.len() > MAX_MEMBERS {
                return Err("membership watch exceeds member budget".into());
            }
            // Fence the observed transition even if a later event in this
            // response recreates the same endpoint/incarnation.
            if members.get(&registration.node_id) != Some(registration) {
                view.fence();
                return Err("own membership registration changed".into());
            }
        }
        if through > applied {
            applied = through;
            view.replace_members(
                members
                    .values()
                    .map(|member| (member.node_id.clone(), member.owner.clone())),
            );
        }
    }
}

fn valid_placement(kv: &KeyValue, view: &MembershipView) -> bool {
    kv.lease() == 0
        && kv.value().len() <= 4096
        && serde_json::from_slice::<orbitkv_catalog::Placement>(kv.value())
            .ok()
            .as_ref()
            == Some(view.placement())
}

fn decode(prefix: &str, kv: &KeyValue) -> Result<Member, String> {
    if kv.value().len() > MEMBER_BYTES || kv.lease() == 0 {
        return Err("membership record is oversized or has no lease".into());
    }
    let mut member: Member =
        serde_json::from_slice(kv.value()).map_err(|error| error.to_string())?;
    member.lease = kv.lease();
    parse_label(&member.node_id)?;
    let address: std::net::SocketAddr = member
        .owner
        .endpoint
        .parse()
        .map_err(|_| "invalid member endpoint")?;
    if member.epoch == 0
        || member.owner.incarnation.is_nil()
        || address.port() == 0
        || address.ip().is_unspecified()
        || kv.key() != format!("{prefix}{}", member.node_id).as_bytes()
    {
        return Err("invalid membership record identity".into());
    }
    Ok(member)
}
