"""
Admin infrastructure endpoints: Cloud Run revisions, Cloud Monitoring metrics,
Cloud Build history, Cloud Logging entries.

All rely on Application Default Credentials (service account attached to the
Cloud Run service). IAM required: run.viewer (implied by run.admin),
monitoring.viewer, cloudbuild.builds.viewer, logging.viewer.
"""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from routes.admin import require_admin

logger = logging.getLogger(__name__)

router = APIRouter()

PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get("GCP_PROJECT_ID") or "prosodyssm"
REGION = os.environ.get("CLOUD_RUN_REGION", "us-central1")
SERVICE_NAME = os.environ.get("CLOUD_RUN_SERVICE", "prosody-api")


# ---------------------------------------------------------------------------
# Cloud Run
# ---------------------------------------------------------------------------

@router.get("/infra/cloudrun", dependencies=[Depends(require_admin)])
async def get_cloudrun_state() -> dict[str, Any]:
    """Current revision, resource config, and recent revisions with traffic split."""
    try:
        from google.cloud import run_v2
    except ImportError as e:
        raise HTTPException(status_code=501, detail=f"google-cloud-run not installed: {e}")

    try:
        services = run_v2.ServicesClient()
        service_path = f"projects/{PROJECT_ID}/locations/{REGION}/services/{SERVICE_NAME}"
        svc = services.get_service(name=service_path)

        template = svc.template
        container = template.containers[0] if template.containers else None

        cpu = container.resources.limits.get("cpu") if container else None
        memory = container.resources.limits.get("memory") if container else None
        image = container.image if container else None

        # Traffic: map revision -> percent + type
        traffic_by_rev: dict[str, dict[str, Any]] = {}
        for t in svc.traffic_statuses or []:
            rev = t.revision or "LATEST"
            traffic_by_rev[rev] = {
                "percent": t.percent,
                "type": run_v2.TrafficTargetAllocationType(t.type_).name if t.type_ else None,
                "tag": t.tag or None,
                "uri": t.uri or None,
            }

        # List recent revisions
        revisions_client = run_v2.RevisionsClient()
        rev_parent = f"projects/{PROJECT_ID}/locations/{REGION}/services/{SERVICE_NAME}"
        revisions = []
        for rev in revisions_client.list_revisions(parent=rev_parent):
            rev_name = rev.name.rsplit("/", 1)[-1]
            traffic = traffic_by_rev.get(rev_name, {})
            labels = dict(rev.labels) if rev.labels else {}
            rev_container = rev.containers[0] if rev.containers else None
            revisions.append({
                "name": rev_name,
                "image": rev_container.image if rev_container else None,
                "created": rev.create_time.isoformat() if rev.create_time else None,
                "commit_sha": labels.get("commit-sha"),
                "build_id": labels.get("gcb-build-id"),
                "cpu": rev_container.resources.limits.get("cpu") if rev_container else None,
                "memory": rev_container.resources.limits.get("memory") if rev_container else None,
                "traffic_percent": traffic.get("percent", 0),
                "is_latest": rev_name == svc.latest_ready_revision.rsplit("/", 1)[-1] if svc.latest_ready_revision else False,
            })

        revisions.sort(key=lambda r: r.get("created") or "", reverse=True)
        revisions = revisions[:10]

        current_rev_name = svc.latest_ready_revision.rsplit("/", 1)[-1] if svc.latest_ready_revision else None
        current = next((r for r in revisions if r["name"] == current_rev_name), None)

        # Env var names only (no values)
        env_names = [e.name for e in (container.env if container else [])]

        return {
            "service": SERVICE_NAME,
            "region": REGION,
            "project": PROJECT_ID,
            "uri": svc.uri,
            "current_revision": current,
            "config": {
                "cpu": cpu,
                "memory": memory,
                "image": image,
                "concurrency": template.max_instance_request_concurrency or None,
                "timeout_seconds": int(template.timeout.total_seconds()) if template.timeout else None,
                "min_instances": template.scaling.min_instance_count if template.scaling else None,
                "max_instances": template.scaling.max_instance_count if template.scaling else None,
                "service_account": template.service_account or None,
                "env_names": env_names,
            },
            "traffic": [
                {"revision": rev, **info}
                for rev, info in traffic_by_rev.items()
            ],
            "revisions": revisions,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("cloudrun_state failed")
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Cloud Monitoring
# ---------------------------------------------------------------------------

def _make_interval(hours: int):
    from google.cloud.monitoring_v3 import TimeInterval
    from google.protobuf import timestamp_pb2

    end = int(time.time())
    start = end - hours * 3600
    start_ts = timestamp_pb2.Timestamp()
    start_ts.FromSeconds(start)
    end_ts = timestamp_pb2.Timestamp()
    end_ts.FromSeconds(end)
    return TimeInterval(start_time=start_ts, end_time=end_ts)


def _aggregation(alignment_period_s: int, aligner, reducer=None):
    from google.cloud.monitoring_v3 import Aggregation
    from google.protobuf.duration_pb2 import Duration

    period = Duration(seconds=alignment_period_s)
    kwargs: dict[str, Any] = {"alignment_period": period, "per_series_aligner": aligner}
    if reducer is not None:
        kwargs["cross_series_reducer"] = reducer
    return Aggregation(**kwargs)


@router.get("/infra/metrics", dependencies=[Depends(require_admin)])
async def get_metrics(hours: int = Query(24, ge=1, le=168)) -> dict[str, Any]:
    """Cloud Monitoring TimeSeries for prosody-api: RPS, errors, latency, CPU, memory."""
    try:
        from google.cloud import monitoring_v3
    except ImportError as e:
        raise HTTPException(status_code=501, detail=f"google-cloud-monitoring not installed: {e}")

    Aligner = monitoring_v3.Aggregation.Aligner
    Reducer = monitoring_v3.Aggregation.Reducer

    client = monitoring_v3.MetricServiceClient()
    project_name = f"projects/{PROJECT_ID}"
    interval = _make_interval(hours)

    # 5-minute buckets; for 1h use 1-minute
    bucket_s = 60 if hours <= 1 else 300

    base_filter = (
        f'resource.type = "cloud_run_revision" '
        f'AND resource.labels.service_name = "{SERVICE_NAME}" '
        f'AND resource.labels.location = "{REGION}"'
    )

    def _ts_seconds(t) -> int:
        """Handle both proto Timestamp and DatetimeWithNanoseconds."""
        if hasattr(t, "seconds") and isinstance(getattr(t, "seconds"), int):
            return t.seconds
        if hasattr(t, "timestamp"):
            return int(t.timestamp())
        return 0

    def _point_value(p):
        v = p.value
        # Prefer distribution mean, then double, then int
        if v.HasField("distribution_value") if hasattr(v, "HasField") else False:
            dist = v.distribution_value
            return dist.mean if dist.count > 0 else 0.0
        if v.double_value:
            return v.double_value
        if v.int64_value:
            return v.int64_value
        return 0.0

    def _series(points):
        return [
            {
                "t": _ts_seconds(p.interval.end_time),
                "v": _point_value(p),
            }
            for p in reversed(list(points))
        ]

    out: dict[str, Any] = {
        "hours": hours,
        "bucket_seconds": bucket_s,
        "request_count": [],
        "error_count": [],
        "latency_p50": [],
        "latency_p95": [],
        "latency_p99": [],
        "cpu_utilization": [],
        "memory_utilization": [],
    }

    # Request count (sum across response codes)
    try:
        results = client.list_time_series(
            request={
                "name": project_name,
                "filter": f'{base_filter} AND metric.type = "run.googleapis.com/request_count"',
                "interval": interval,
                "view": monitoring_v3.ListTimeSeriesRequest.TimeSeriesView.FULL,
                "aggregation": _aggregation(bucket_s, Aligner.ALIGN_DELTA, Reducer.REDUCE_SUM),
            }
        )
        all_points: list[Any] = []
        for ts in results:
            all_points.extend(ts.points)
        out["request_count"] = _series(all_points)
    except Exception as e:
        logger.warning(f"request_count fetch failed: {e}")

    # 5xx errors
    try:
        results = client.list_time_series(
            request={
                "name": project_name,
                "filter": f'{base_filter} AND metric.type = "run.googleapis.com/request_count" AND metric.labels.response_code_class = "5xx"',
                "interval": interval,
                "view": monitoring_v3.ListTimeSeriesRequest.TimeSeriesView.FULL,
                "aggregation": _aggregation(bucket_s, Aligner.ALIGN_DELTA, Reducer.REDUCE_SUM),
            }
        )
        all_points = []
        for ts in results:
            all_points.extend(ts.points)
        out["error_count"] = _series(all_points)
    except Exception as e:
        logger.warning(f"error_count fetch failed: {e}")

    # Latency percentiles
    for pct, aligner, key in [
        (50, Aligner.ALIGN_PERCENTILE_50, "latency_p50"),
        (95, Aligner.ALIGN_PERCENTILE_95, "latency_p95"),
        (99, Aligner.ALIGN_PERCENTILE_99, "latency_p99"),
    ]:
        try:
            results = client.list_time_series(
                request={
                    "name": project_name,
                    "filter": f'{base_filter} AND metric.type = "run.googleapis.com/request_latencies"',
                    "interval": interval,
                    "view": monitoring_v3.ListTimeSeriesRequest.TimeSeriesView.FULL,
                    "aggregation": _aggregation(bucket_s, aligner, Reducer.REDUCE_MEAN),
                }
            )
            all_points = []
            for ts in results:
                all_points.extend(ts.points)
            out[key] = _series(all_points)
        except Exception as e:
            logger.warning(f"{key} (p{pct}) fetch failed: {e}")

    # CPU utilization
    try:
        results = client.list_time_series(
            request={
                "name": project_name,
                "filter": f'{base_filter} AND metric.type = "run.googleapis.com/container/cpu/utilizations"',
                "interval": interval,
                "view": monitoring_v3.ListTimeSeriesRequest.TimeSeriesView.FULL,
                "aggregation": _aggregation(bucket_s, Aligner.ALIGN_PERCENTILE_95, Reducer.REDUCE_MEAN),
            }
        )
        all_points = []
        for ts in results:
            all_points.extend(ts.points)
        out["cpu_utilization"] = _series(all_points)
    except Exception as e:
        logger.warning(f"cpu_utilization fetch failed: {e}")

    # Memory utilization
    try:
        results = client.list_time_series(
            request={
                "name": project_name,
                "filter": f'{base_filter} AND metric.type = "run.googleapis.com/container/memory/utilizations"',
                "interval": interval,
                "view": monitoring_v3.ListTimeSeriesRequest.TimeSeriesView.FULL,
                "aggregation": _aggregation(bucket_s, Aligner.ALIGN_PERCENTILE_95, Reducer.REDUCE_MEAN),
            }
        )
        all_points = []
        for ts in results:
            all_points.extend(ts.points)
        out["memory_utilization"] = _series(all_points)
    except Exception as e:
        logger.warning(f"memory_utilization fetch failed: {e}")

    return out


# ---------------------------------------------------------------------------
# Cloud Build
# ---------------------------------------------------------------------------

@router.get("/infra/builds", dependencies=[Depends(require_admin)])
async def list_builds(limit: int = Query(10, ge=1, le=50)) -> dict[str, Any]:
    """List recent Cloud Build runs for the API service."""
    try:
        from google.cloud.devtools import cloudbuild_v1
    except ImportError as e:
        raise HTTPException(status_code=501, detail=f"google-cloud-build not installed: {e}")

    client = cloudbuild_v1.CloudBuildClient()

    try:
        pager = client.list_builds(request={
            "project_id": PROJECT_ID,
            "page_size": limit,
        })

        builds: list[dict[str, Any]] = []
        for b in pager:
            if len(builds) >= limit:
                break
            subs = dict(b.substitutions) if b.substitutions else {}
            # Filter to our service only (via trigger substitution or tags)
            service_name = subs.get("_SERVICE_NAME", "")
            if service_name and service_name != SERVICE_NAME:
                continue
            status_name = cloudbuild_v1.Build.Status(b.status).name if b.status else "UNKNOWN"
            duration_s: Optional[float] = None
            if b.start_time and b.finish_time:
                duration_s = (
                    b.finish_time.timestamp() - b.start_time.timestamp()
                )
            elif b.start_time:
                duration_s = time.time() - b.start_time.timestamp()

            builds.append({
                "id": b.id,
                "status": status_name,
                "commit_sha": subs.get("COMMIT_SHA") or subs.get("SHORT_SHA"),
                "ref_name": subs.get("REF_NAME") or subs.get("BRANCH_NAME") or subs.get("TAG_NAME"),
                "trigger_id": b.build_trigger_id,
                "service": service_name or None,
                "start_time": b.start_time.isoformat() if b.start_time else None,
                "finish_time": b.finish_time.isoformat() if b.finish_time else None,
                "duration_seconds": duration_s,
                "log_url": b.log_url,
            })

        return {"builds": builds}
    except Exception as e:
        logger.exception("list_builds failed")
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Cloud Logging
# ---------------------------------------------------------------------------

@router.get("/infra/logs", dependencies=[Depends(require_admin)])
async def list_logs(
    limit: int = Query(50, ge=1, le=200),
    severity: str = Query("ERROR"),
    hours: int = Query(24, ge=1, le=168),
) -> dict[str, Any]:
    """Recent Cloud Logging entries for prosody-api."""
    try:
        from google.cloud import logging as cloud_logging
    except ImportError as e:
        raise HTTPException(status_code=501, detail=f"google-cloud-logging not installed: {e}")

    client = cloud_logging.Client(project=PROJECT_ID)

    since = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
    severity = severity.upper()
    valid_severities = {"DEFAULT", "DEBUG", "INFO", "NOTICE", "WARNING", "ERROR", "CRITICAL", "ALERT", "EMERGENCY"}
    if severity not in valid_severities:
        raise HTTPException(status_code=400, detail=f"Invalid severity. Use one of {sorted(valid_severities)}")

    filter_str = (
        f'resource.type="cloud_run_revision" '
        f'AND resource.labels.service_name="{SERVICE_NAME}" '
        f'AND severity>="{severity}" '
        f'AND timestamp>="{since}"'
    )

    try:
        entries = []
        for entry in client.list_entries(filter_=filter_str, order_by=cloud_logging.DESCENDING, max_results=limit):
            # Extract textual payload — could be text, struct, or proto
            text = ""
            payload = getattr(entry, "payload", None)
            if isinstance(payload, str):
                text = payload
            elif isinstance(payload, dict):
                text = (
                    payload.get("message")
                    or payload.get("msg")
                    or payload.get("text")
                    or str(payload)
                )
            elif payload is not None:
                text = str(payload)

            # Fallback to text_payload/json_payload attrs on the entry itself
            if not text:
                text = (
                    getattr(entry, "text_payload", "")
                    or str(getattr(entry, "json_payload", "") or "")
                    or str(getattr(entry, "proto_payload", "") or "")
                )

            text = (text or "")[:1000]

            entries.append({
                "timestamp": entry.timestamp.isoformat() if entry.timestamp else None,
                "severity": entry.severity or "DEFAULT",
                "text": text,
                "insert_id": entry.insert_id,
                "trace": entry.trace,
                "revision": (entry.resource.labels or {}).get("revision_name") if entry.resource else None,
            })

        return {"entries": entries, "severity": severity, "hours": hours}
    except Exception as e:
        logger.exception("list_logs failed")
        raise HTTPException(status_code=500, detail=str(e))
