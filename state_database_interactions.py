import json
import hashlib
import json
import secrets
import datetime

import sqlalchemy as sqa

import database_definition as dbd
import database_interactions as dbi


def _normalize_json_string(json_string: str) -> str:
    """
    Parse and re-serialize JSON into a canonical form so logically identical
    states deduplicate even if whitespace/key order differ.
    """
    try:
        obj = json.loads(json_string)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON: {e}") from e

    return json.dumps(
        obj,
        separators=(",", ":"),
        sort_keys=True,
        ensure_ascii=False,
    )


def _make_dedupe_hash(
    normalized_json: str,
    *,
    state_version: int,
) -> str:
    """
    Include compatibility metadata in the dedupe key.

    That means:
    - same JSON + same version + same commit => same bookmark row
    - same JSON but different deploy/version => different bookmark row
    """
    material = f"{state_version}\0{normalized_json}".encode("utf-8")
    return hashlib.sha256(material).hexdigest()


def _new_opaque_id() -> str:
    return secrets.token_urlsafe(16)


# ---------- function 1: store ----------

def get_exisiting_state(session: sqa.orm.Session, dedupe_hash: str) -> dbd.BookmarkState:
    existing = session.scalars(
        sqa.select(dbd.BookmarkState).where(dbd.BookmarkState.dedupe_hash == dedupe_hash)
    ).first()
    return existing


def store_bookmark_state(
    json_string: str,
    *,
    state_version: int,
    ttl: datetime.timedelta = datetime.timedelta(days=365),
) -> str:
    """
    Store a JSON state and return the opaque bookmark id.

    Deduplication:
    - if an identical compatible state already exists, reuse its opaque_id
    - otherwise create a new row

    This function commits the session.
    """
    now = dbd.utcnow()
    expires_at = now + ttl

    normalized_json = _normalize_json_string(json_string)
    dedupe_hash = _make_dedupe_hash(
        normalized_json,
        state_version=state_version,
    )

    with sqa.orm.Session(dbi.get_db_engine()) as session:

        # check whether the state is already existing
        existing = get_exisiting_state(session, dedupe_hash)

        if existing is not None:
            existing.last_used_at = now
            if existing.expires_at < expires_at:
                existing.expires_at = expires_at
            session.commit()
            return existing.opaque_id

        # Handle rare races / id collisions.
        for _ in range(5):
            row = dbd.BookmarkState(
                opaque_id=_new_opaque_id(),
                dedupe_hash=dedupe_hash,
                json_state=normalized_json,
                state_version=state_version,
                created_at=now,
                last_used_at=now,
                expires_at=expires_at,
            )
            session.add(row)

            try:
                session.commit()
                return row.opaque_id
            except sqa.exc.IntegrityError:
                session.rollback()

                # Most likely another request inserted the same dedupe_hash first.
                existing = get_exisiting_state(session, dedupe_hash)
                if existing is not None:
                    existing.last_used_at = now
                    if existing.expires_at < expires_at:
                        existing.expires_at = expires_at
                    session.commit()
                    return existing.opaque_id

    raise RuntimeError("Could not create a unique bookmark id after several attempts")


# ---------- function 2: load ----------

def load_bookmark_state(
    opaque_id: str,
    *,
    current_state_version: int | None = None,
    extend_ttl: datetime.timedelta | None = datetime.timedelta(days=365),
) -> str:
    """
    Load a JSON state by opaque bookmark id.

    Validation:
    - rejects unknown ids
    - rejects expired rows
    - optionally rejects rows from another state_version and/or git_commit

    This function commits the session if it updates last_used_at / expires_at.
    """
    now = dbd.utcnow()

    with sqa.orm.Session(dbi.get_db_engine()) as session:
        row = session.scalars(
            sqa.select(dbd.BookmarkState).where(dbd.BookmarkState.opaque_id == opaque_id)
        ).first()

        if row is None:
            raise LookupError("Unknown bookmark id")

        if row.expires_at <= now:
            raise LookupError("Bookmark has expired")

        if current_state_version is not None and row.state_version != current_state_version:
            raise LookupError(
                f"Bookmark state version {row.state_version} is not supported "
                f"by current version {current_state_version}"
            )

        row.last_used_at = now

        if extend_ttl is not None:
            new_expires_at = now + extend_ttl
            if row.expires_at < new_expires_at:
                row.expires_at = new_expires_at
        json_state = str(row.json_state)
        session.commit()
    return json_state


# function to delete old entries

def delete_expired_bookmarks() -> int:
    with sqa.orm.Session(dbi.get_db_engine()) as session:
        result = session.execute(
            sqa.delete(dbd.BookmarkState).where(dbd.BookmarkState.expires_at <= dbd.utcnow())
        )
        session.commit()
        return result.rowcount or 0
