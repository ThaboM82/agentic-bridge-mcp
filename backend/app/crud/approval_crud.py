# ---------------------------------------------------------------------------
# Create
# ---------------------------------------------------------------------------


async def create_approval_request(
    db: AsyncSession,
    *,
    workflow_id: str | UUID,
    workflow_step: str,
    requested_by: str | None = None,
    tenant_id: str | None = None,
    approver_role: str | None = None,
    approver_user_id: str | None = None,
    payload: dict[str, Any] | None = None,
    expires_at: datetime | None = None,
) -> ApprovalRequestModel:
    """
    Create and persist a new approval request.

    Args:
        db: Active async database session.
        workflow_id: Workflow execution requiring approval.
        workflow_step: Workflow step that created the request.
        requested_by: User or service requesting approval.
        tenant_id: Owning tenant identifier.
        approver_role: Role permitted to approve the request.
        approver_user_id: Specific user permitted to approve the request.
        payload: Non-sensitive approval context.
        expires_at: Optional expiration timestamp.

    Returns:
        The persisted approval request.

    Raises:
        ValidationError: If required values are invalid.
        DatabaseIntegrityError: If a constraint is violated.
        PersistenceError: If another database failure occurs.
    """
    safe_workflow_id = _normalize_workflow_id(workflow_id)

    if safe_workflow_id is None:
        raise ValidationError("workflow_id is required.")

    safe_workflow_step = _normalize_reason(workflow_step)

    if safe_workflow_step is None:
        raise ValidationError("workflow_step is required.")

    if expires_at is not None and expires_at <= _utc_now():
        raise ValidationError(
            "Approval expiration must be in the future."
        )

    approval = ApprovalRequestModel(
        workflow_id=safe_workflow_id,
        workflow_step=safe_workflow_step,
        requested_by=requested_by,
        tenant_id=tenant_id,
        approver_role=approver_role,
        approver_user_id=approver_user_id,
        payload=payload or {},
        status=ApprovalStatus.PENDING.value,
        expires_at=expires_at,
    )

    approval = await create_record(db, approval)

    bind_workflow_context(str(safe_workflow_id))

    logger.info(
        "approval_request_created",
        approval_id=str(approval.id),
        workflow_id=str(safe_workflow_id),
        workflow_step=approval.workflow_step,
        tenant_id=tenant_id,
        approver_role=approver_role,
        expires_at=expires_at.isoformat() if expires_at else None,
    )

    return approval


# ---------------------------------------------------------------------------
# Read
# ---------------------------------------------------------------------------


async def get_approval_request(
    db: AsyncSession,
    approval_id: str | UUID,
    *,
    tenant_id: str | None = None,
) -> ApprovalRequestModel | None:
    """
    Retrieve an approval request by ID.

    Args:
        db: Active async database session.
        approval_id: Approval-request UUID.
        tenant_id: Optional tenant isolation filter.

    Returns:
        The approval request, or `None` when it does not exist or belongs to
        another tenant.
    """
    approval_uuid = _normalize_approval_id(approval_id)

    query = select(ApprovalRequestModel).where(
        ApprovalRequestModel.id == approval_uuid
    )

    query = _apply_tenant_filter(query, tenant_id)

    result = await db.execute(query)

    return result.scalar_one_or_none()


async def get_approval_request_or_404(
    db: AsyncSession,
    approval_id: str | UUID,
    *,
    tenant_id: str | None = None,
) -> ApprovalRequestModel:
    """
    Retrieve an approval request by ID or raise `NotFoundError`.

    Args:
        db: Active async database session.
        approval_id: Approval-request UUID.
        tenant_id: Optional tenant isolation filter.

    Returns:
        The matching approval request.

    Raises:
        NotFoundError: If the request does not exist or is outside the tenant.
    """
    approval = await get_approval_request(
        db,
        approval_id,
        tenant_id=tenant_id,
    )

    if approval is None:
        raise NotFoundError(
            "Approval request was not found.",
            details={"approval_id": str(approval_id)},
        )

    return approval


async def get_pending_approval_for_workflow(
    db: AsyncSession,
    workflow_id: str | UUID,
    *,
    workflow_step: str | None = None,
    tenant_id: str | None = None,
) -> ApprovalRequestModel | None:
    """
    Retrieve the active pending approval for a workflow.

    Args:
        db: Active async database session.
        workflow_id: Workflow execution UUID.
        workflow_step: Optional workflow-step filter.
        tenant_id: Optional tenant isolation filter.

    Returns:
        The pending approval request, or `None` when none exists.
    """
    safe_workflow_id = _normalize_workflow_id(workflow_id)

    if safe_workflow_id is None:
        raise ValidationError("workflow_id is required.")

    query = select(ApprovalRequestModel).where(
        and_(
            ApprovalRequestModel.workflow_id == safe_workflow_id,
            ApprovalRequestModel.status == ApprovalStatus.PENDING.value,
        )
    )

    safe_workflow_step = _normalize_reason(workflow_step)

    if safe_workflow_step:
        query = query.where(
            ApprovalRequestModel.workflow_step == safe_workflow_step
        )

    query = _apply_tenant_filter(query, tenant_id)
    query = query.order_by(
        ApprovalRequestModel.created_at.desc()
    ).limit(1)

    result = await db.execute(query)

    return result.scalar_one_or_none()


async def list_approval_requests(
    db: AsyncSession,
    *,
    pagination: PaginationParams,
    tenant_id: str | None = None,
    workflow_id: str | UUID | None = None,
    status: str | ApprovalStatus | None = None,
    workflow_step: str | None = None,
    approver_user_id: str | None = None,
    approver_role: str | None = None,
    created_after: datetime | None = None,
    created_before: datetime | None = None,
) -> Page[ApprovalRequestModel]:
    """
    Retrieve a paginated, filtered list of approval requests.

    Args:
        db: Active async database session.
        pagination: Validated pagination parameters.
        tenant_id: Optional tenant isolation filter.
        workflow_id: Optional workflow filter.
        status: Optional approval-status filter.
        workflow_step: Optional workflow-step filter.
        approver_user_id: Optional assigned approver filter.
        approver_role: Optional approver-role filter.
        created_after: Optional inclusive lower created-at bound.
        created_before: Optional inclusive upper created-at bound.

    Returns:
        A `Page` containing approval requests and pagination metadata.
    """
    safe_workflow_id = _normalize_workflow_id(workflow_id)
    safe_status = _normalize_status(status)
    safe_workflow_step = _normalize_reason(workflow_step)

    filters = []

    if tenant_id:
        filters.append(ApprovalRequestModel.tenant_id == tenant_id)

    if safe_workflow_id is not None:
        filters.append(ApprovalRequestModel.workflow_id == safe_workflow_id)

    if safe_status is not None:
        filters.append(ApprovalRequestModel.status == safe_status.value)

    if safe_workflow_step:
        filters.append(
            ApprovalRequestModel.workflow_step == safe_workflow_step
        )

    if approver_user_id:
        filters.append(
            ApprovalRequestModel.approver_user_id == approver_user_id
        )

    if approver_role:
        filters.append(ApprovalRequestModel.approver_role == approver_role)

    if created_after is not None:
        filters.append(ApprovalRequestModel.created_at >= created_after)

    if created_before is not None:
        filters.append(ApprovalRequestModel.created_at <= created_before)

    count_query = select(func.count()).select_from(ApprovalRequestModel)

    if filters:
        count_query = count_query.where(and_(*filters))

    count_result = await db.execute(count_query)
    total = int(count_result.scalar_one())

    query = select(ApprovalRequestModel).order_by(
        ApprovalRequestModel.created_at.desc(),
        ApprovalRequestModel.id.desc(),
    )

    if filters:
        query = query.where(and_(*filters))

    result = await db.execute(
        query.offset(pagination.skip).limit(pagination.limit)
    )

    approvals = list(result.scalars().all())

    return Page[ApprovalRequestModel].create(
        items=approvals,
        total=total,
        skip=pagination.skip,
        limit=pagination.limit,
    )


async def count_approval_requests(
    db: AsyncSession,
    *,
    tenant_id: str | None = None,
    status: str | ApprovalStatus | None = None,
) -> int:
    """
    Count approval requests matching optional filters.

    Args:
        db: Active async database session.
        tenant_id: Optional tenant isolation filter.
        status: Optional approval-status filter.

    Returns:
        Total matching approval-request count.
    """
    safe_status = _normalize_status(status)

    filters = []

    if tenant_id:
        filters.append(ApprovalRequestModel.tenant_id == tenant_id)

    if safe_status is not None:
        filters.append(ApprovalRequestModel.status == safe_status.value)

    query = select(func.count()).select_from(ApprovalRequestModel)

    if filters:
        query = query.where(and_(*filters))

    result = await db.execute(query)

    return int(result.scalar_one())


# ---------------------------------------------------------------------------
# Decision helpers
# ---------------------------------------------------------------------------


async def _ensure_decision_allowed(
    db: AsyncSession,
    approval: ApprovalRequestModel,
) -> ApprovalStatus:
    """
    Ensure an approval request can still receive a decision.

    Args:
        db: Active async database session.
        approval: Approval request to validate.

    Returns:
        The current approval status.

    Raises:
        ConflictError: If the request has already been decided.
        WorkflowStateError: If the request has expired.
    """
    current_status = _normalize_status(approval.status)

    if current_status is None:
        raise ValidationError("Approval request status is invalid.")

    if current_status is not ApprovalStatus.PENDING:
        raise ConflictError(
            "Approval request has already been decided.",
            details={"current_status": current_status.value},
        )

    if approval.expires_at is not None and approval.expires_at <= _utc_now():
        approval.status = ApprovalStatus.EXPIRED.value
        approval.decision = ApprovalDecision.REQUEST_CHANGES.value
        approval.decided_at = _utc_now()
        approval.updated_at = _utc_now()

        approval = await update_record(db, approval)

        logger.warning(
            "approval_request_expired_during_decision",
            approval_id=str(approval.id),
            workflow_id=str(approval.workflow_id),
        )

        raise WorkflowStateError(
            "Approval request has expired.",
            details={"approval_id": str(approval.id)},
        )

    return current_status


# ---------------------------------------------------------------------------
# Decisions
# ---------------------------------------------------------------------------


async def approve_approval_request(
    db: AsyncSession,
    approval_id: str | UUID,
    *,
    decided_by: str,
    comment: str | None = None,
    tenant_id: str | None = None,
) -> ApprovalRequestModel:
    """
    Approve a pending approval request.

    Args:
        db: Active async database session.
        approval_id: Approval-request UUID.
        decided_by: Authenticated user ID making the decision.
        comment: Optional approver comment.
        tenant_id: Optional tenant isolation filter.

    Returns:
        The approved approval request.

    Raises:
        ValidationError: If decision metadata is invalid.
        NotFoundError: If the request does not exist.
        ConflictError: If the request was already decided.
        WorkflowStateError: If the request has expired.
    """
    safe_decided_by = _normalize_reason(decided_by)

    if safe_decided_by is None:
        raise ValidationError("decided_by is required.")

    approval = await get_approval_request_or_404(
        db,
        approval_id,
        tenant_id=tenant_id,
    )

    await _ensure_decision_allowed(db, approval)

    approval.status = ApprovalStatus.APPROVED.value
    approval.decision = ApprovalDecision.APPROVED.value
    approval.decided_by = safe_decided_by
    approval.decision_comment = _normalize_comment(comment)
    approval.decided_at = _utc_now()
    approval.updated_at = _utc_now()

    approval = await update_record(db, approval)

    bind_workflow_context(str(approval.workflow_id))

    logger.info(
        "approval_request_approved",
        approval_id=str(approval.id),
        workflow_id=str(approval.workflow_id),
        decided_by=safe_decided_by,
    )

    return approval


async def reject_approval_request(
    db: AsyncSession,
    approval_id: str | UUID,
    *,
    decided_by: str,
    comment: str | None = None,
    tenant_id: str | None = None,
) -> ApprovalRequestModel:
    """
    Reject a pending approval request.

    Args:
        db: Active async database session.
        approval_id: Approval-request UUID.
        decided_by: Authenticated user ID making the decision.
        comment: Optional reason for rejection.
        tenant_id: Optional tenant isolation filter.

    Returns:
        The rejected approval request.

    Raises:
        ValidationError: If decision metadata is invalid.
        NotFoundError: If the request does not exist.
        ConflictError: If the request was already decided.
        WorkflowStateError: If the request has expired.
    """
    safe_decided_by = _normalize_reason(decided_by)

    if safe_decided_by is None:
        raise ValidationError("decided_by is required.")

    approval = await get_approval_request_or_404(
        db,
        approval_id,
        tenant_id=tenant_id,
    )

    await _ensure_decision_allowed(db, approval)

    approval.status = ApprovalStatus.REJECTED.value
    approval.decision = ApprovalDecision.REJECTED.value
    approval.decided_by = safe_decided_by
    approval.decision_comment = _normalize_comment(comment)
    approval.decided_at = _utc_now()
    approval.updated_at = _utc_now()

    approval = await update_record(db, approval)

    bind_workflow_context(str(approval.workflow_id))

    logger.warning(
        "approval_request_rejected",
        approval_id=str(approval.id),
        workflow_id=str(approval.workflow_id),
        decided_by=safe_decided_by,
    )

    return approval


async def request_changes(
    db: AsyncSession,
    approval_id: str | UUID,
    *,
    decided_by: str,
    comment: str | None = None,
    tenant_id: str | None = None,
) -> ApprovalRequestModel:
    """
    Request changes on a pending approval request.

    This records a reviewer decision without approving the workflow step.

    Args:
        db: Active async database session.
        approval_id: Approval-request UUID.
        decided_by: Authenticated user ID making the decision.
        comment: Required or optional reviewer feedback.
        tenant_id: Optional tenant isolation filter.

    Returns:
        The updated approval request.

    Raises:
        ValidationError: If decision metadata is invalid.
        NotFoundError: If the request does not exist.
        ConflictError: If the request was already decided.
    """
    safe_decided_by = _normalize_reason(decided_by)

    if safe_decided_by is None:
        raise ValidationError("decided_by is required.")

    approval = await get_approval_request_or_404(
        db,
        approval_id,
        tenant_id=tenant_id,
    )

    await _ensure_decision_allowed(db, approval)

    approval.status = ApprovalStatus.REJECTED.value
    approval.decision = ApprovalDecision.REQUEST_CHANGES.value
    approval.decided_by = safe_decided_by
    approval.decision_comment = _normalize_comment(comment)
    approval.decided_at = _utc_now()
    approval.updated_at = _utc_now()

    approval = await update_record(db, approval)

    bind_workflow_context(str(approval.workflow_id))

    logger.info(
        "approval_changes_requested",
        approval_id=str(approval.id),
        workflow_id=str(approval.workflow_id),
        decided_by=safe_decided_by,
    )

    return approval


async def cancel_approval_request(
    db: AsyncSession,
    approval_id: str | UUID,
    *,
    cancelled_by: str | None = None,
    reason: str | None = None,
    tenant_id: str | None = None,
) -> ApprovalRequestModel:
    """
    Cancel a pending approval request.

    Args:
        db: Active async database session.
        approval_id: Approval-request UUID.
        cancelled_by: Optional user or service cancelling the request.
        reason: Optional cancellation reason.
        tenant_id: Optional tenant isolation filter.

    Returns:
        The cancelled approval request.

    Raises:
        NotFoundError: If the request does not exist.
        ConflictError: If the request was already decided.
    """
    approval = await get_approval_request_or_404(
        db,
        approval_id,
        tenant_id=tenant_id,
    )

    current_status = _normalize_status(approval.status)

    if current_status is None:
        raise ValidationError("Approval request status is invalid.")

    if current_status is not ApprovalStatus.PENDING:
        raise ConflictError(
            "Only pending approval requests can be cancelled.",
            details={"current_status": current_status.value},
        )

    approval.status = ApprovalStatus.CANCELLED.value
    approval.decision = ApprovalDecision.REQUEST_CHANGES.value
    approval.decided_by = _normalize_reason(cancelled_by)
    approval.decision_comment = _normalize_reason(reason)
    approval.decided_at = _utc_now()
    approval.updated_at = _utc_now()

    approval = await update_record(db, approval)

    bind_workflow_context(str(approval.workflow_id))

    logger.info(
        "approval_request_cancelled",
        approval_id=str(approval.id),
        workflow_id=str(approval.workflow_id),
        cancelled_by=cancelled_by,
    )

    return approval


async def expire_stale_approvals(
    db: AsyncSession,
    *,
    now: datetime | None = None,
    tenant_id: str | None = None,
    limit: int = 100,
) -> Sequence[ApprovalRequestModel]:
    """
    Expire pending approval requests whose expiration time has passed.

    Args:
        db: Active async database session.
        now: Optional current-time override for tests and maintenance jobs.
        tenant_id: Optional tenant isolation filter.
        limit: Maximum number of requests to expire.

    Returns:
        The expired approval requests.
    """
    current_time = _utc_now() if now is None else now

    query = select(ApprovalRequestModel).where(
        and_(
            ApprovalRequestModel.status == ApprovalStatus.PENDING.value,
            ApprovalRequestModel.expires_at.is_not(None),
            ApprovalRequestModel.expires_at <= current_time,
        )
    )

    query = _apply_tenant_filter(query, tenant_id)
    query = query.order_by(
        ApprovalRequestModel.expires_at.asc()
    ).limit(max(1, limit))

    result = await db.execute(query)
    approvals = list(result.scalars().all())

    expired_approvals: list[ApprovalRequestModel] = []

    for approval in approvals:
        approval.status = ApprovalStatus.EXPIRED.value
        approval.decision = ApprovalDecision.REQUEST_CHANGES.value
        approval.decided_at = current_time
        approval.updated_at = current_time

        expired_approvals.append(await update_record(db, approval))

    if expired_approvals:
        logger.warning(
            "approval_requests_expired",
            expired_count=len(expired_approvals),
            tenant_id=tenant_id,
        )

    return expired_approvals