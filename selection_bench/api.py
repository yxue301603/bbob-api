"""Chat Completions client for choosing one of eight benchmark candidates."""
import json
import os
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, build_opener

from api_client import NoRedirect, validate_url


SYSTEM = (
    "You are a black-box optimizer. Use only the supplied observations and candidate "
    "information. Return exactly one JSON object with one field, \"pick\", whose value "
    "is an integer from 0 through 7 identifying the selected candidate."
)
MAX_RETRIES = 3


def choose(job, step, prompt):
    if not isinstance(job, str) or not job or type(step) is not int or step < 0:
        raise ValueError("A nonempty job name and a nonnegative integer step are required")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("The candidate prompt must be nonempty text")
    base, key, model = (os.environ.get(name, "").strip() for name in
                        ("DEEPSEEK_BASE_URL", "DEEPSEEK_API_KEY", "DEEPSEEK_MODEL"))
    if not base or not key or not model:
        raise ValueError("Set DEEPSEEK_BASE_URL, DEEPSEEK_API_KEY and DEEPSEEK_MODEL")
    if not key.isascii() or any(not 33 <= ord(char) <= 126 for char in key):
        raise ValueError("The API credential must contain only visible ASCII characters")
    validate_url(base)
    body = {
        "model": model, "temperature": 0, "thinking": {"type": "disabled"},
        "max_tokens": 32, "response_format": {"type": "json_object"},
        "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
    }
    req = Request(base.rstrip("/") + "/chat/completions",
                  data=json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode("utf-8"),
                  headers={"Content-Type": "application/json", "Authorization": "Bearer " + key})
    client = build_opener(NoRedirect())
    for attempt in range(MAX_RETRIES + 1):
        try:
            with client.open(req, timeout=60) as response:
                result = json.load(response)
            message = result["choices"][0]["message"]
            usage = result.get("usage") or {}
            reasoning_tokens = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens", 0)
            if message.get("reasoning_content") or reasoning_tokens:
                raise RuntimeError("The selected endpoint returned reasoning despite the nonthinking setting")
            parsed = json.loads(message["content"])
            if not isinstance(parsed, dict) or set(parsed) != {"pick"}:
                raise ValueError("Invalid JSON object")
            pick = parsed["pick"]
            if type(pick) is not int or not 0 <= pick <= 7:
                raise ValueError("Invalid candidate index")
            return pick
        except HTTPError as error:
            code = error.code
            error.close()
            if code not in (408, 429) and not 500 <= code < 600:
                raise RuntimeError(f"Model request rejected (HTTP {code})") from None
            reason = f"HTTP {code}"
        except (URLError, TimeoutError, OSError):
            reason = "connection failure or timeout"
        except (ValueError, KeyError, IndexError, TypeError, AttributeError):
            reason = "invalid model response"
        if attempt < MAX_RETRIES:
            time.sleep(min(2**attempt, 4))
    raise RuntimeError(f"Model selection failed after {MAX_RETRIES + 1} attempts ({reason})") from None
