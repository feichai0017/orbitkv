//! Device-written semantic status packets and host-side decoding.

use loom_infer::ContractError;
use thiserror::Error;

pub(crate) const STATUS_PACKET_WORDS: usize = 5;

pub(crate) const STATUS_SUCCESS: i32 = 0;
pub(crate) const STATUS_INVALID_PAGE_INDPTR_START: i32 = 1;
pub(crate) const STATUS_NON_MONOTONIC_PAGE_INDPTR: i32 = 2;
pub(crate) const STATUS_EMPTY_PAGED_REQUEST: i32 = 3;
pub(crate) const STATUS_INVALID_LAST_PAGE_LENGTH: i32 = 4;
pub(crate) const STATUS_PAGE_INDICES_LENGTH_MISMATCH: i32 = 5;
pub(crate) const STATUS_PAGE_INDEX_OUT_OF_RANGE: i32 = 6;
pub(crate) const STATUS_APPEND_BATCH_INDEX_OUT_OF_RANGE: i32 = 7;
pub(crate) const STATUS_APPEND_POSITION_OUT_OF_RANGE: i32 = 8;
pub(crate) const STATUS_DUPLICATE_APPEND_SLOT: i32 = 9;
pub(crate) const STATUS_PAGE_REFERENCE_COUNT_TOO_SMALL: i32 = 10;
pub(crate) const STATUS_NON_EXCLUSIVE_APPEND_TARGET: i32 = 11;
pub(crate) const STATUS_ELEMENT_COUNT_OVERFLOW: i32 = 12;
pub(crate) const STATUS_INVALID_QO_INDPTR_START: i32 = 13;
pub(crate) const STATUS_NON_MONOTONIC_QO_INDPTR: i32 = 14;
pub(crate) const STATUS_EMPTY_QO_REQUEST: i32 = 15;
pub(crate) const STATUS_QO_INDPTR_LENGTH_MISMATCH: i32 = 16;
pub(crate) const STATUS_RAGGED_QUERY_LONGER_THAN_KV: i32 = 17;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum AppendMapKind {
    Requests,
    ExplicitTokens,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DeviceStatusKind {
    PagedAppend(AppendMapKind),
    PagedBatchDecode,
    PagedPrefill,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct DeviceStatusDecoder {
    kind: DeviceStatusKind,
    items: usize,
    batch_size: usize,
    max_num_pages: usize,
    page_indices_len: usize,
    page_size: usize,
}

impl DeviceStatusDecoder {
    pub(crate) const fn paged_append(
        kind: AppendMapKind,
        items: usize,
        batch_size: usize,
        max_num_pages: usize,
        page_indices_len: usize,
        page_size: usize,
    ) -> Self {
        Self {
            kind: DeviceStatusKind::PagedAppend(kind),
            items,
            batch_size,
            max_num_pages,
            page_indices_len,
            page_size,
        }
    }

    pub(crate) const fn paged_batch_decode(
        batch_size: usize,
        max_num_pages: usize,
        page_indices_len: usize,
        page_size: usize,
    ) -> Self {
        Self {
            kind: DeviceStatusKind::PagedBatchDecode,
            items: batch_size,
            batch_size,
            max_num_pages,
            page_indices_len,
            page_size,
        }
    }

    pub(crate) const fn paged_prefill(
        batch_size: usize,
        nnz_qo: usize,
        max_num_pages: usize,
        page_indices_len: usize,
        page_size: usize,
    ) -> Self {
        Self {
            kind: DeviceStatusKind::PagedPrefill,
            items: nnz_qo,
            batch_size,
            max_num_pages,
            page_indices_len,
            page_size,
        }
    }

    pub(crate) const fn operation(self) -> &'static str {
        match self.kind {
            DeviceStatusKind::PagedAppend(AppendMapKind::Requests) => "paged KV append map",
            DeviceStatusKind::PagedAppend(AppendMapKind::ExplicitTokens) => {
                "explicit-token paged KV append map"
            }
            DeviceStatusKind::PagedBatchDecode => "paged batch decode metadata",
            DeviceStatusKind::PagedPrefill => "paged prefill metadata",
        }
    }

    pub(crate) fn decode(
        self,
        packet: &[i32],
    ) -> Result<Option<ContractError>, DeviceStatusProtocolError> {
        if packet.len() != STATUS_PACKET_WORDS {
            return Err(DeviceStatusProtocolError::new(
                self.operation(),
                -1,
                "status packet has the wrong length",
            ));
        }
        let code = packet[0];
        let detail = &packet[1..];
        let error = match code {
            STATUS_SUCCESS => {
                self.require_zero_tail(code, detail, 0)?;
                return Ok(None);
            }
            STATUS_INVALID_PAGE_INDPTR_START => {
                self.require(code, detail[0] != 0, "invalid page indptr start is zero")?;
                self.require_zero_tail(code, detail, 1)?;
                ContractError::InvalidPageIndptrStart { actual: detail[0] }
            }
            STATUS_NON_MONOTONIC_PAGE_INDPTR => {
                let request = self.index_below(
                    code,
                    detail[0],
                    self.batch_size,
                    "request index is outside the batch",
                )?;
                self.require(code, detail[2] < detail[1], "page indptr pair is monotonic")?;
                self.require_zero_tail(code, detail, 3)?;
                ContractError::NonMonotonicPageIndptr {
                    request,
                    start: detail[1],
                    end: detail[2],
                }
            }
            STATUS_EMPTY_PAGED_REQUEST => {
                let request = self.index_below(
                    code,
                    detail[0],
                    self.batch_size,
                    "request index is outside the batch",
                )?;
                self.require_zero_tail(code, detail, 1)?;
                ContractError::EmptyPagedRequest { request }
            }
            STATUS_INVALID_LAST_PAGE_LENGTH => {
                let request = self.index_below(
                    code,
                    detail[0],
                    self.batch_size,
                    "request index is outside the batch",
                )?;
                self.require(
                    code,
                    detail[1] < 1 || usize::try_from(detail[1]).is_ok_and(|v| v > self.page_size),
                    "last page length is inside the valid range",
                )?;
                self.require_zero_tail(code, detail, 2)?;
                ContractError::InvalidLastPageLength {
                    request,
                    length: detail[1],
                    page_size: self.page_size,
                }
            }
            STATUS_PAGE_INDICES_LENGTH_MISMATCH => {
                let expected = self.index(code, detail[0])?;
                self.require(
                    code,
                    expected != self.page_indices_len,
                    "page_indices lengths are equal",
                )?;
                self.require_zero_tail(code, detail, 1)?;
                ContractError::LengthMismatch {
                    buffer: "page_indices",
                    expected,
                    actual: self.page_indices_len,
                }
            }
            STATUS_PAGE_INDEX_OUT_OF_RANGE => {
                let position = self.index_below(
                    code,
                    detail[0],
                    self.page_indices_len,
                    "page index position is outside page_indices",
                )?;
                self.require(
                    code,
                    detail[1] < 0
                        || usize::try_from(detail[1])
                            .is_ok_and(|index| index >= self.max_num_pages),
                    "physical page index is inside the valid range",
                )?;
                self.require_zero_tail(code, detail, 2)?;
                ContractError::PageIndexOutOfRange {
                    position,
                    index: detail[1],
                    max_num_pages: self.max_num_pages,
                }
            }
            STATUS_APPEND_BATCH_INDEX_OUT_OF_RANGE => {
                if self.append_kind() != Some(AppendMapKind::ExplicitTokens) {
                    return Err(self.unexpected(code));
                }
                let token = self.index_below(
                    code,
                    detail[0],
                    self.items,
                    "token index is outside the append span",
                )?;
                self.require(
                    code,
                    detail[1] < 0
                        || usize::try_from(detail[1])
                            .is_ok_and(|request| request >= self.batch_size),
                    "append batch index is inside the valid range",
                )?;
                self.require_zero_tail(code, detail, 2)?;
                ContractError::AppendBatchIndexOutOfRange {
                    token,
                    index: detail[1],
                    batch_size: self.batch_size,
                }
            }
            STATUS_APPEND_POSITION_OUT_OF_RANGE => {
                if self.append_kind() != Some(AppendMapKind::ExplicitTokens) {
                    return Err(self.unexpected(code));
                }
                let token = self.index_below(
                    code,
                    detail[0],
                    self.items,
                    "token index is outside the append span",
                )?;
                let request = self.index_below(
                    code,
                    detail[1],
                    self.batch_size,
                    "request index is outside the batch",
                )?;
                let kv_len = self.index(code, detail[3])?;
                let max_kv_len = self
                    .page_indices_len
                    .checked_mul(self.page_size)
                    .ok_or_else(|| {
                        DeviceStatusProtocolError::new(
                            self.operation(),
                            code,
                            "decoder KV length domain overflows",
                        )
                    })?;
                self.require(
                    code,
                    kv_len > 0 && kv_len <= max_kv_len,
                    "KV length is outside the operation domain",
                )?;
                self.require(
                    code,
                    detail[2] < 0
                        || usize::try_from(detail[2]).is_ok_and(|position| position >= kv_len),
                    "append position is inside the valid range",
                )?;
                ContractError::AppendPositionOutOfRange {
                    token,
                    request,
                    position: detail[2],
                    kv_len,
                }
            }
            STATUS_DUPLICATE_APPEND_SLOT => {
                let Some(kind) = self.append_kind() else {
                    return Err(self.unexpected(code));
                };
                let first = self.index_below(
                    code,
                    detail[0],
                    self.items,
                    "first item index is outside the append span",
                )?;
                let second = self.index_below(
                    code,
                    detail[1],
                    self.items,
                    "second item index is outside the append span",
                )?;
                self.require(
                    code,
                    first < second,
                    "duplicate append item order is invalid",
                )?;
                let physical_page = self.index_below(
                    code,
                    detail[2],
                    self.max_num_pages,
                    "physical page index is outside the page pool",
                )?;
                let offset = self.index_below(
                    code,
                    detail[3],
                    self.page_size,
                    "page offset is outside the page",
                )?;
                match kind {
                    AppendMapKind::Requests => ContractError::DuplicatePageAppendSlot {
                        first_request: first,
                        second_request: second,
                        physical_page,
                        offset,
                    },
                    AppendMapKind::ExplicitTokens => ContractError::DuplicatePageAppendTokenSlot {
                        first_token: first,
                        second_token: second,
                        physical_page,
                        offset,
                    },
                }
            }
            STATUS_PAGE_REFERENCE_COUNT_TOO_SMALL => {
                if self.append_kind().is_none() {
                    return Err(self.unexpected(code));
                }
                let physical_page = self.index_below(
                    code,
                    detail[0],
                    self.max_num_pages,
                    "physical page index is outside the page pool",
                )?;
                let minimum = self.index(code, detail[1])?;
                self.require(
                    code,
                    minimum <= self.page_indices_len,
                    "minimum page reference count exceeds page_indices length",
                )?;
                self.require(
                    code,
                    detail[2] < 0
                        || usize::try_from(detail[2]).is_ok_and(|actual| actual < minimum),
                    "page reference count satisfies the minimum",
                )?;
                self.require_zero_tail(code, detail, 3)?;
                ContractError::PageReferenceCountTooSmall {
                    physical_page,
                    minimum,
                    actual: detail[2],
                }
            }
            STATUS_NON_EXCLUSIVE_APPEND_TARGET => {
                if self.append_kind().is_none() {
                    return Err(self.unexpected(code));
                }
                let physical_page = self.index_below(
                    code,
                    detail[0],
                    self.max_num_pages,
                    "physical page index is outside the page pool",
                )?;
                self.require(
                    code,
                    detail[1] != 1,
                    "append target has an exclusive reference count",
                )?;
                self.require_zero_tail(code, detail, 2)?;
                ContractError::NonExclusivePageAppendTarget {
                    physical_page,
                    reference_count: detail[1],
                }
            }
            STATUS_ELEMENT_COUNT_OVERFLOW => {
                self.require_zero_tail(code, detail, 0)?;
                ContractError::ElementCountOverflow
            }
            STATUS_INVALID_QO_INDPTR_START => {
                self.require_paged_prefill(code)?;
                self.require(code, detail[0] != 0, "invalid qo indptr start is zero")?;
                self.require_zero_tail(code, detail, 1)?;
                ContractError::InvalidIndptrStart {
                    buffer: "qo_indptr",
                    actual: detail[0],
                }
            }
            STATUS_NON_MONOTONIC_QO_INDPTR => {
                self.require_paged_prefill(code)?;
                let request = self.index_below(
                    code,
                    detail[0],
                    self.batch_size,
                    "request index is outside the batch",
                )?;
                self.require(code, detail[2] < detail[1], "qo indptr pair is monotonic")?;
                self.require_zero_tail(code, detail, 3)?;
                ContractError::NonMonotonicIndptr {
                    buffer: "qo_indptr",
                    request,
                    start: detail[1],
                    end: detail[2],
                }
            }
            STATUS_EMPTY_QO_REQUEST => {
                self.require_paged_prefill(code)?;
                let request = self.index_below(
                    code,
                    detail[0],
                    self.batch_size,
                    "request index is outside the batch",
                )?;
                self.require_zero_tail(code, detail, 1)?;
                ContractError::EmptyRaggedRequest {
                    buffer: "qo_indptr",
                    request,
                }
            }
            STATUS_QO_INDPTR_LENGTH_MISMATCH => {
                self.require_paged_prefill(code)?;
                let actual = self.index(code, detail[0])?;
                self.require(code, actual != self.items, "qo indptr totals are equal")?;
                self.require_zero_tail(code, detail, 1)?;
                ContractError::LengthMismatch {
                    buffer: "qo_indptr",
                    expected: self.items,
                    actual,
                }
            }
            STATUS_RAGGED_QUERY_LONGER_THAN_KV => {
                self.require_paged_prefill(code)?;
                let request = self.index_below(
                    code,
                    detail[0],
                    self.batch_size,
                    "request index is outside the batch",
                )?;
                let query_len = self.index(code, detail[1])?;
                let kv_len = self.index(code, detail[2])?;
                let max_kv_len = self
                    .page_indices_len
                    .checked_mul(self.page_size)
                    .ok_or_else(|| {
                        DeviceStatusProtocolError::new(
                            self.operation(),
                            code,
                            "decoder KV length domain overflows",
                        )
                    })?;
                self.require(
                    code,
                    query_len > 0 && query_len <= self.items,
                    "query length is outside the operation domain",
                )?;
                self.require(
                    code,
                    kv_len > 0 && kv_len <= max_kv_len && query_len > kv_len,
                    "query length does not exceed KV length",
                )?;
                self.require_zero_tail(code, detail, 3)?;
                ContractError::RaggedQueryLongerThanKv {
                    request,
                    query_len,
                    kv_len,
                }
            }
            _ => return Err(self.unexpected(code)),
        };
        Ok(Some(error))
    }

    const fn append_kind(self) -> Option<AppendMapKind> {
        match self.kind {
            DeviceStatusKind::PagedAppend(kind) => Some(kind),
            DeviceStatusKind::PagedBatchDecode | DeviceStatusKind::PagedPrefill => None,
        }
    }

    fn require_paged_prefill(self, code: i32) -> Result<(), DeviceStatusProtocolError> {
        if self.kind == DeviceStatusKind::PagedPrefill {
            Ok(())
        } else {
            Err(self.unexpected(code))
        }
    }

    fn index(self, code: i32, value: i32) -> Result<usize, DeviceStatusProtocolError> {
        usize::try_from(value).map_err(|_| {
            DeviceStatusProtocolError::new(
                self.operation(),
                code,
                "status packet contains a negative index",
            )
        })
    }

    fn index_below(
        self,
        code: i32,
        value: i32,
        upper: usize,
        reason: &'static str,
    ) -> Result<usize, DeviceStatusProtocolError> {
        let value = self.index(code, value)?;
        if value < upper {
            Ok(value)
        } else {
            Err(DeviceStatusProtocolError::new(
                self.operation(),
                code,
                reason,
            ))
        }
    }

    fn require(
        self,
        code: i32,
        condition: bool,
        reason: &'static str,
    ) -> Result<(), DeviceStatusProtocolError> {
        if condition {
            Ok(())
        } else {
            Err(DeviceStatusProtocolError::new(
                self.operation(),
                code,
                reason,
            ))
        }
    }

    fn require_zero_tail(
        self,
        code: i32,
        detail: &[i32],
        used: usize,
    ) -> Result<(), DeviceStatusProtocolError> {
        self.require(
            code,
            detail[used..].iter().all(|&value| value == 0),
            "unused status detail is nonzero",
        )
    }

    const fn unexpected(self, code: i32) -> DeviceStatusProtocolError {
        DeviceStatusProtocolError::new(
            self.operation(),
            code,
            "status packet contains an unknown code",
        )
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Error)]
#[error("{operation} status protocol failed with code {code}: {reason}")]
pub struct DeviceStatusProtocolError {
    operation: &'static str,
    code: i32,
    reason: &'static str,
}

impl DeviceStatusProtocolError {
    const fn new(operation: &'static str, code: i32, reason: &'static str) -> Self {
        Self {
            operation,
            code,
            reason,
        }
    }

    pub const fn operation(self) -> &'static str {
        self.operation
    }

    pub const fn code(self) -> i32 {
        self.code
    }

    pub const fn reason(self) -> &'static str {
        self.reason
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn explicit_duplicate_status_decodes_to_contract_error() {
        let decoder =
            DeviceStatusDecoder::paged_append(AppendMapKind::ExplicitTokens, 5, 3, 8, 5, 16);
        let packet = [STATUS_DUPLICATE_APPEND_SLOT, 1, 4, 7, 3];
        assert_eq!(
            decoder.decode(&packet),
            Ok(Some(ContractError::DuplicatePageAppendTokenSlot {
                first_token: 1,
                second_token: 4,
                physical_page: 7,
                offset: 3,
            }))
        );
    }

    #[test]
    fn unknown_status_code_is_a_protocol_error() {
        let decoder = DeviceStatusDecoder::paged_append(AppendMapKind::Requests, 2, 2, 4, 2, 16);
        let error = decoder
            .decode(&[99, 0, 0, 0, 0])
            .expect_err("unknown status must fail");
        assert_eq!(error.code(), 99);
    }

    #[test]
    fn valid_metadata_rejections_keep_their_contract_errors() {
        let requests = DeviceStatusDecoder::paged_append(AppendMapKind::Requests, 2, 2, 4, 3, 16);
        for packet in [
            [STATUS_INVALID_PAGE_INDPTR_START, 2, 0, 0, 0],
            [STATUS_EMPTY_PAGED_REQUEST, 1, 0, 0, 0],
            [STATUS_INVALID_LAST_PAGE_LENGTH, 0, 17, 0, 0],
            [STATUS_PAGE_INDICES_LENGTH_MISMATCH, 2, 0, 0, 0],
            [STATUS_DUPLICATE_APPEND_SLOT, 0, 1, 3, 15],
            [STATUS_PAGE_REFERENCE_COUNT_TOO_SMALL, 3, 1, 0, 0],
            [STATUS_NON_EXCLUSIVE_APPEND_TARGET, 3, 2, 0, 0],
            [STATUS_ELEMENT_COUNT_OVERFLOW, 0, 0, 0, 0],
        ] {
            assert!(
                matches!(requests.decode(&packet), Ok(Some(_))),
                "packet should decode as a metadata rejection: {packet:?}"
            );
        }
        assert_eq!(
            requests.decode(&[STATUS_NON_MONOTONIC_PAGE_INDPTR, 1, 4, 3, 0]),
            Ok(Some(ContractError::NonMonotonicPageIndptr {
                request: 1,
                start: 4,
                end: 3,
            }))
        );
        assert_eq!(
            requests.decode(&[STATUS_PAGE_INDEX_OUT_OF_RANGE, 2, 4, 0, 0]),
            Ok(Some(ContractError::PageIndexOutOfRange {
                position: 2,
                index: 4,
                max_num_pages: 4,
            }))
        );

        let tokens =
            DeviceStatusDecoder::paged_append(AppendMapKind::ExplicitTokens, 3, 2, 4, 3, 16);
        assert!(matches!(
            tokens.decode(&[STATUS_APPEND_BATCH_INDEX_OUT_OF_RANGE, 2, -1, 0, 0]),
            Ok(Some(ContractError::AppendBatchIndexOutOfRange { .. }))
        ));
        assert_eq!(
            tokens.decode(&[STATUS_APPEND_POSITION_OUT_OF_RANGE, 2, 1, 17, 17]),
            Ok(Some(ContractError::AppendPositionOutOfRange {
                token: 2,
                request: 1,
                position: 17,
                kv_len: 17,
            }))
        );
    }

    #[test]
    fn out_of_domain_details_are_protocol_errors() {
        let decoder = DeviceStatusDecoder::paged_append(AppendMapKind::Requests, 2, 2, 4, 3, 16);
        for packet in [
            [STATUS_SUCCESS, 1, 0, 0, 0],
            [STATUS_INVALID_PAGE_INDPTR_START, 0, 0, 0, 0],
            [STATUS_NON_MONOTONIC_PAGE_INDPTR, 0, 3, 3, 0],
            [STATUS_EMPTY_PAGED_REQUEST, 2, 0, 0, 0],
            [STATUS_INVALID_LAST_PAGE_LENGTH, 0, 16, 0, 0],
            [STATUS_PAGE_INDICES_LENGTH_MISMATCH, 3, 0, 0, 0],
            [STATUS_PAGE_INDEX_OUT_OF_RANGE, 3, 4, 0, 0],
            [STATUS_PAGE_INDEX_OUT_OF_RANGE, 0, 3, 0, 0],
            [STATUS_DUPLICATE_APPEND_SLOT, 1, 0, 0, 0],
            [STATUS_DUPLICATE_APPEND_SLOT, 0, 1, 4, 0],
            [STATUS_DUPLICATE_APPEND_SLOT, 0, 1, 3, 16],
            [STATUS_PAGE_REFERENCE_COUNT_TOO_SMALL, 4, 1, 0, 0],
            [STATUS_PAGE_REFERENCE_COUNT_TOO_SMALL, 0, 4, 0, 0],
            [STATUS_PAGE_REFERENCE_COUNT_TOO_SMALL, 0, 1, 1, 0],
            [STATUS_NON_EXCLUSIVE_APPEND_TARGET, 0, 1, 0, 0],
            [STATUS_ELEMENT_COUNT_OVERFLOW, 0, 0, 0, 1],
        ] {
            assert!(
                decoder.decode(&packet).is_err(),
                "packet should fail protocol validation: {packet:?}"
            );
        }
    }

    #[test]
    fn explicit_token_details_are_bounded() {
        let decoder =
            DeviceStatusDecoder::paged_append(AppendMapKind::ExplicitTokens, 3, 2, 4, 3, 16);
        for packet in [
            [STATUS_APPEND_BATCH_INDEX_OUT_OF_RANGE, 3, -1, 0, 0],
            [STATUS_APPEND_BATCH_INDEX_OUT_OF_RANGE, 0, 1, 0, 0],
            [STATUS_APPEND_POSITION_OUT_OF_RANGE, 3, 0, -1, 1],
            [STATUS_APPEND_POSITION_OUT_OF_RANGE, 0, 2, -1, 1],
            [STATUS_APPEND_POSITION_OUT_OF_RANGE, 0, 0, 0, 0],
            [STATUS_APPEND_POSITION_OUT_OF_RANGE, 0, 0, 0, 1],
            [STATUS_APPEND_POSITION_OUT_OF_RANGE, 0, 0, 48, 49],
        ] {
            assert!(
                decoder.decode(&packet).is_err(),
                "packet should fail protocol validation: {packet:?}"
            );
        }
    }

    #[test]
    fn paged_decode_status_decodes_contract_rejections() {
        let decoder = DeviceStatusDecoder::paged_batch_decode(2, 4, 3, 16);
        assert_eq!(
            decoder.decode(&[STATUS_NON_MONOTONIC_PAGE_INDPTR, 1, 3, 2, 0]),
            Ok(Some(ContractError::NonMonotonicPageIndptr {
                request: 1,
                start: 3,
                end: 2,
            }))
        );
        assert_eq!(
            decoder.decode(&[STATUS_PAGE_INDEX_OUT_OF_RANGE, 2, 4, 0, 0]),
            Ok(Some(ContractError::PageIndexOutOfRange {
                position: 2,
                index: 4,
                max_num_pages: 4,
            }))
        );
    }

    #[test]
    fn paged_decode_rejects_malformed_or_append_only_status() {
        let decoder = DeviceStatusDecoder::paged_batch_decode(2, 4, 3, 16);
        for packet in [
            [STATUS_SUCCESS, 1, 0, 0, 0],
            [STATUS_EMPTY_PAGED_REQUEST, 2, 0, 0, 0],
            [STATUS_INVALID_LAST_PAGE_LENGTH, 0, 16, 0, 0],
            [STATUS_DUPLICATE_APPEND_SLOT, 0, 1, 3, 0],
            [STATUS_NON_EXCLUSIVE_APPEND_TARGET, 3, 2, 0, 0],
            [99, 0, 0, 0, 0],
        ] {
            assert!(
                decoder.decode(&packet).is_err(),
                "packet should fail paged-decode protocol validation: {packet:?}"
            );
        }
    }

    #[test]
    fn paged_prefill_status_decodes_query_and_page_rejections() {
        let decoder = DeviceStatusDecoder::paged_prefill(2, 3, 4, 3, 16);
        for (packet, expected) in [
            (
                [STATUS_INVALID_QO_INDPTR_START, 1, 0, 0, 0],
                ContractError::InvalidIndptrStart {
                    buffer: "qo_indptr",
                    actual: 1,
                },
            ),
            (
                [STATUS_NON_MONOTONIC_QO_INDPTR, 1, 2, 1, 0],
                ContractError::NonMonotonicIndptr {
                    buffer: "qo_indptr",
                    request: 1,
                    start: 2,
                    end: 1,
                },
            ),
            (
                [STATUS_EMPTY_QO_REQUEST, 0, 0, 0, 0],
                ContractError::EmptyRaggedRequest {
                    buffer: "qo_indptr",
                    request: 0,
                },
            ),
            (
                [STATUS_QO_INDPTR_LENGTH_MISMATCH, 2, 0, 0, 0],
                ContractError::LengthMismatch {
                    buffer: "qo_indptr",
                    expected: 3,
                    actual: 2,
                },
            ),
            (
                [STATUS_RAGGED_QUERY_LONGER_THAN_KV, 0, 2, 1, 0],
                ContractError::RaggedQueryLongerThanKv {
                    request: 0,
                    query_len: 2,
                    kv_len: 1,
                },
            ),
            (
                [STATUS_PAGE_INDEX_OUT_OF_RANGE, 2, 4, 0, 0],
                ContractError::PageIndexOutOfRange {
                    position: 2,
                    index: 4,
                    max_num_pages: 4,
                },
            ),
        ] {
            assert_eq!(decoder.decode(&packet), Ok(Some(expected)));
        }
    }

    #[test]
    fn paged_prefill_rejects_malformed_or_foreign_status() {
        let decoder = DeviceStatusDecoder::paged_prefill(2, 3, 4, 3, 16);
        for packet in [
            [STATUS_SUCCESS, 1, 0, 0, 0],
            [STATUS_INVALID_QO_INDPTR_START, 0, 0, 0, 0],
            [STATUS_NON_MONOTONIC_QO_INDPTR, 0, 1, 1, 0],
            [STATUS_EMPTY_QO_REQUEST, 2, 0, 0, 0],
            [STATUS_QO_INDPTR_LENGTH_MISMATCH, 3, 0, 0, 0],
            [STATUS_RAGGED_QUERY_LONGER_THAN_KV, 0, 0, 0, 0],
            [STATUS_RAGGED_QUERY_LONGER_THAN_KV, 0, 2, 2, 0],
            [STATUS_APPEND_BATCH_INDEX_OUT_OF_RANGE, 0, -1, 0, 0],
            [99, 0, 0, 0, 0],
        ] {
            assert!(
                decoder.decode(&packet).is_err(),
                "packet should fail paged-prefill protocol validation: {packet:?}"
            );
        }

        let decode = DeviceStatusDecoder::paged_batch_decode(2, 4, 3, 16);
        let append = DeviceStatusDecoder::paged_append(AppendMapKind::Requests, 2, 2, 4, 3, 16);
        for packet in [
            [STATUS_INVALID_QO_INDPTR_START, 1, 0, 0, 0],
            [STATUS_NON_MONOTONIC_QO_INDPTR, 1, 2, 1, 0],
            [STATUS_EMPTY_QO_REQUEST, 0, 0, 0, 0],
            [STATUS_QO_INDPTR_LENGTH_MISMATCH, 2, 0, 0, 0],
            [STATUS_RAGGED_QUERY_LONGER_THAN_KV, 0, 2, 1, 0],
        ] {
            assert!(decode.decode(&packet).is_err());
            assert!(append.decode(&packet).is_err());
        }
    }
}
