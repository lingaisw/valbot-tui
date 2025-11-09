from pydantic_ai import RunContext
from pydantic import BaseModel
from rich.console import Console
from typing import Optional, Dict, Any
import requests
import urllib3
from urllib3.exceptions import InsecureRequestWarning

####################################### HSD TOOLS ################################################
async def fetch_hsd_details_by_id(ctx: RunContext, hsd_id: str, fields: Optional[str] = "id,title,description") -> dict:
    """
    Fetch HSD article details by ID using Intel's HSD API.
    Args:
        hsd_id (str): The HSD article ID to fetch.
        fields (Optional[str]): Comma-separated fields to include in the response.
    Returns:
        dict: A dictionary containing the article details or error information.
    """
    console = Console()
    console.log(f"Fetching HSD article details for ID: {hsd_id}")
    result = await _fetch_article_details(hsd_id, fields=fields)
    if "error" in result:
        console.log(f"[red]Error fetching article {hsd_id}: {result['error']}[/red]")
    else:
        console.log(f"[green]Successfully fetched article {hsd_id}[/green]")
    return result

####################################### HELPERS ###############################################
async def _fetch_article_details(article_id: str, *, base: str = "https://hsdes-api.intel.com/rest", tenant: str | None = None,
                                subject: str | None = None, fields: str | None = None,
                                verbose: bool | None = None) -> Dict[str, Any]:
    url = _build_article_url(article_id, base=base, tenant=tenant, subject=subject, fields=fields, verbose=verbose)
    data = _http_get_json(url)
    if isinstance(data, dict) and data.get("__error__"):
        return {"error": data["__error__"], "url": url}
    # Extract first article if present for convenience
    data_list = data.get("data") if isinstance(data, dict) else None
    article = data_list[0] if isinstance(data_list, list) and data_list else None
    return {"article_id": article_id, "raw": data, "article": article, "url": url}

def _build_article_url(article_id: str, *, base: str = "https://hsdes-api.intel.com/rest", tenant: str | None = None,
                      subject: str | None = None, fields: str | None = None, verbose: bool | None = None) -> str:
    params = []
    if verbose: params.append("verbose=true")
    if tenant: params.append(f"tenant={tenant}")
    if subject: params.append(f"subject={subject}")
    if fields: params.append(f"fields={fields}")
    query = ("?" + "&".join(params)) if params else ""
    return f"{base}/article/{article_id}{query}"

def _http_get_json(url: str, verify: bool = False, timeout: int = 30) -> Dict[str, Any]:
    if requests is None:  # pragma: no cover
        return {"__error__": "Kerberos/GSSAPI HTTP modules not available"}
    try:
        auth = _make_auth()
        if auth is None:  # pragma: no cover
            return {"__error__": "No Kerberos/GSSAPI auth available"}
        if not verify:
            urllib3.disable_warnings(category=InsecureRequestWarning)
        resp = requests.get(url, headers={"Accept": "application/json"}, auth=auth, verify=verify, timeout=timeout)
        resp.raise_for_status()
        # Some endpoints may return text/plain JSON
        ctype = resp.headers.get("content-type", "")
        return resp.json() if "json" in ctype else {"text": resp.text}
    except Exception as e:  # pragma: no cover - network/env variability
        return {"__error__": str(e)}

def _make_auth():
    """Return an auth object for the current environment, or None if unavailable."""
    try:
        from requests_kerberos import HTTPKerberosAuth, REQUIRED
        return HTTPKerberosAuth(mutual_authentication=REQUIRED)
    except ImportError:  # pragma: no cover
        return None
