"""One stateless Chat Completions request per model decision."""
import json
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def validate_url(url):
    parts = urlsplit(url)
    local_http = parts.scheme == "http" and parts.hostname in {"127.0.0.1", "localhost", "::1"}
    if not (parts.scheme == "https" or local_http) or not parts.hostname:
        raise ValueError("Use HTTPS for the API or HTTP on localhost for a local server.")
    if parts.username or parts.password or parts.query or parts.fragment:
        raise ValueError("The API base URL must not contain credentials, query parameters or fragments.")


def choose(prompt, model, base_url, api_key, effort, count):
    messages = [
        {"role": "system", "content": 'You are a black-box optimizer. Use only the supplied task data. Return exactly one JSON object {"pick": integer} with a zero-based candidate index.'},
        {"role": "user", "content": prompt},
    ]
    body = {"model": model, "messages": messages}
    if effort:
        body["reasoning_effort"] = effort
    request = Request(base_url.rstrip("/") + "/chat/completions",
                      data=json.dumps(body).encode(),
                      headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"})
    try:
        with build_opener(NoRedirect()).open(request, timeout=120) as response:
            data = json.load(response)
    except HTTPError as error:
        raise RuntimeError(f"API HTTP {error.code}. Check endpoint, model, credentials, quota and supported parameters. Resume with the same output directory.") from None
    except (URLError, TimeoutError) as error:
        raise RuntimeError("API connection failed or timed out. Completed steps are saved.") from error
    try:
        reply = data["choices"][0]["message"]["content"].strip()
        if reply.startswith("```") and reply.endswith("```"):
            reply = reply.split("\n", 1)[1][:-3].strip()
        pick = json.loads(reply)["pick"]
        if type(pick) is not int or not 0 <= pick < count:
            raise ValueError("invalid candidate index")
    except (KeyError, IndexError, ValueError, AttributeError, TypeError):
        raise RuntimeError("API response must contain a JSON object with a valid integer pick. Check the model's text output.") from None
    return {"pick": pick, "response_id": data.get("id"), "returned_model": data.get("model"),
            "usage": data.get("usage"), "messages": messages}
