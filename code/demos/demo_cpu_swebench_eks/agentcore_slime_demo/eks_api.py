"""Minimal EKS API client: stdlib HTTPS + a boto3-signed bearer token. No kubectl, no `kubernetes` package.

The trainer container (agentcore_p4) has boto3 but neither the aws CLI nor kubectl, so the kubeconfig
`aws eks get-token` exec plugin is unavailable there. This does what that plugin does: an EKS token is
`k8s-aws-v1.` + base64url(a presigned STS GetCallerIdentity URL carrying the header
`x-k8s-aws-id: <cluster>`). The API server hands it to STS and maps the caller's IAM role to a
Kubernetes identity via the cluster's ACCESS ENTRIES (the role that created the cluster gets admin;
if the trainer uses another role, add an access entry for it by hand). Tokens last 15 min; we mint a new one every 10.

    api = EksApi.from_cluster("acs-swe-eks")     # endpoint + CA from eks:DescribeCluster
    api.request("GET", "/api/v1/namespaces/acs-swe/pods")

Synchronous by design (callers offload with asyncio.to_thread, like the AgentCore sandbox).
"""
from __future__ import annotations

import base64
import json
import os
import ssl
import tempfile
import threading
import time
import urllib.error
import urllib.request

_TOKEN_TTL = 600.0


class KubeError(RuntimeError):
    def __init__(self, status: int, body: str) -> None:
        super().__init__(f"k8s API {status}: {body[:400]}")
        self.status = status


class EksApi:
    def __init__(self, *, cluster: str, endpoint: str, ca_pem: bytes, region: str) -> None:
        self.cluster, self.endpoint, self.region = cluster, endpoint.rstrip("/"), region
        with tempfile.NamedTemporaryFile("wb", suffix=".crt", delete=False) as f:
            f.write(ca_pem)
        self._ssl = ssl.create_default_context(cafile=f.name)
        self._lock = threading.Lock()
        self._token, self._token_at = "", 0.0

    @classmethod
    def from_cluster(cls, cluster: str | None = None, region: str | None = None) -> "EksApi":
        import boto3

        cluster = cluster or os.environ.get("EKS_CLUSTER", "acs-swe-eks")
        region = region or os.environ.get("AWS_REGION", "us-west-2")
        c = boto3.client("eks", region_name=region).describe_cluster(name=cluster)["cluster"]
        return cls(cluster=cluster, endpoint=c["endpoint"], region=region,
                   ca_pem=base64.b64decode(c["certificateAuthority"]["data"]))

    def token(self) -> str:
        with self._lock:
            if time.time() - self._token_at > _TOKEN_TTL:
                import boto3
                from botocore.signers import RequestSigner

                sts = boto3.client("sts", region_name=self.region)
                signer = RequestSigner(sts.meta.service_model.service_id, self.region, "sts", "v4",
                                       sts._request_signer._credentials, sts.meta.events)
                url = signer.generate_presigned_url(
                    {"method": "GET", "body": {}, "context": {},
                     "url": f"https://sts.{self.region}.amazonaws.com/?Action=GetCallerIdentity&Version=2011-06-15",
                     "headers": {"x-k8s-aws-id": self.cluster}},
                    region_name=self.region, expires_in=60, operation_name="")
                self._token = "k8s-aws-v1." + base64.urlsafe_b64encode(url.encode()).decode().rstrip("=")
                self._token_at = time.time()
            return self._token

    def request(self, method: str, path: str, body: dict | None = None, *, timeout: float = 30.0,
                content_type: str = "application/json") -> dict:
        req = urllib.request.Request(
            self.endpoint + path, method=method,
            data=None if body is None else json.dumps(body).encode(),
            headers={"Authorization": f"Bearer {self.token()}", "Accept": "application/json",
                     "Content-Type": content_type})
        try:
            with urllib.request.urlopen(req, timeout=timeout, context=self._ssl) as r:
                raw = r.read()
        except urllib.error.HTTPError as e:
            raise KubeError(e.code, e.read().decode(errors="replace")) from None
        return json.loads(raw) if raw else {}


_API: EksApi | None = None
_API_LOCK = threading.Lock()


def get_api() -> EksApi:
    """Process-wide client (one DescribeCluster, one token cache) shared by every episode's pod."""
    global _API
    with _API_LOCK:
        if _API is None:
            _API = EksApi.from_cluster()
    return _API
