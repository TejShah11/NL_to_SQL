# ================================================================
# main.py — unified version of run.py, agent.py, tools.py, safety.py, and utils.py
# ================================================================

from __future__ import annotations
import argparse, sqlite3, os, re, json, textwrap, time, threading, platform, signal, sys
from typing import Dict, Any, List, Tuple, Iterable, Sequence, Callable, Optional
from contextlib import contextmanager
from tabulate import tabulate
from dotenv import load_dotenv

# ================================================================
# ======================== utils.py ================================
# ================================================================

def format_rows(rows: Iterable[Sequence[Any]], headers: Sequence[str] | None = None, max_rows: int = 20) -> str:
    rows = list(rows)
    clipped = rows[:max_rows]
    table = tabulate(clipped, headers=headers or [], tablefmt="plain")
    if len(rows) > max_rows:
        table += f"\n... ({len(rows)-max_rows} more rows truncated)"
    return table

def shorten(s: str, n: int = 1500) -> str:
    if s is None:
        return ""
    if len(s) <= n:
        return s
    return s[:n] + f"\n... ({len(s)-n} chars truncated)"

def log_to_file(path: str, msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(path, "a", encoding="utf-8") as f:
        f.write(f"[{ts}] {msg}\n")

def retry(fn: Callable[[], Any], attempts: int = 2, backoff_s: tuple[float, ...] = (0.5, 1.5)) -> Any:
    last_exc = None
    for i in range(attempts):
        try:
            return fn()
        except Exception as e:
            last_exc = e
            if i < attempts - 1:
                time.sleep(backoff_s[min(i, len(backoff_s)-1)])
    raise last_exc

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


# ================================================================
# ======================== safety.py ================================
# ================================================================

FORBIDDEN = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|REPLACE|ATTACH|DETACH|PRAGMA|VACUUM|TRIGGER|INDEX)\b",
    flags=re.IGNORECASE
)
SELECT_WORD = re.compile(r"^\s*SELECT\b", re.IGNORECASE)
NESTED_SELECT = re.compile(r"\(\s*SELECT\b", re.IGNORECASE)
LIMIT_RX = re.compile(r"\bLIMIT\s+(\d+)\b", re.IGNORECASE)
IDENT_RX = re.compile(r"\b([A-Za-z_][A-Za-z0-9_\.]*)\b")

class ValidationError(Exception): ...
class ReadOnlyViolation(Exception): ...
class TimeoutError(Exception): ...

def _call_with_timeout(fn, seconds: int):
    """Cross-platform timeout using a worker thread."""
    result = {}
    err = {}

    def target():
        try:
            result["value"] = fn()
        except Exception as e:
            err["exc"] = e

    t = threading.Thread(target=target, daemon=True)
    t.start()
    t.join(seconds)
    if t.is_alive():
        raise TimeoutError(f"Operation exceeded {seconds}s.")
    if "exc" in err:
        raise err["exc"]
    return result.get("value")

def validate_select_query(sql: str, allowed_identifiers: set[str], max_limit: int) -> str:
    s = " ".join(sql.strip().split())
    if not SELECT_WORD.search(s):
        raise ReadOnlyViolation("Only SELECT queries are allowed.")
    if FORBIDDEN.search(s):
        raise ReadOnlyViolation("Forbidden SQL keyword detected; query must be strictly read-only.")
    if NESTED_SELECT.search(s):
        raise ValidationError("Nested SELECT/subqueries are not allowed in this assignment.")

    idents = set(m.group(1) for m in IDENT_RX.finditer(s))
    suspects = {i for i in idents if i.isidentifier() and not i.upper() in {
        "SELECT","FROM","WHERE","AND","OR","NOT","GROUP","BY","ORDER","ASC","DESC","LIMIT",
        "JOIN","ON","AS","SUM","COUNT","AVG","MIN","MAX","DISTINCT","HAVING","INNER","LEFT","RIGHT"
    }}
    unknown = {i for i in suspects if i not in allowed_identifiers and "." not in i}
    if unknown:
        pass

    m = LIMIT_RX.search(s)
    if m:
        try:
            val = int(m.group(1))
            if val > max_limit:
                s = LIMIT_RX.sub(f"LIMIT {max_limit}", s)
        except ValueError:
            s = LIMIT_RX.sub(f"LIMIT {max_limit}", s)
    else:
        s += f" LIMIT {max_limit}"
    return s

@contextmanager
def timeout(seconds: int):
    def _handle(signum, frame):
        raise TimeoutError(f"Operation exceeded {seconds}s.")
    old = signal.signal(signal.SIGALRM, _handle)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


# ================================================================
# ======================== tools.py ================================
# ================================================================

class Tool:
    name: str = ""
    description: str = ""
    params_schema: Dict[str, Any] = {}

    def __init__(self, conn: sqlite3.Connection, row_limit: int = 100, allowed_identifiers: Optional[set[str]] = None):
        self.conn = conn
        self.row_limit = row_limit
        self.allowed_identifiers = allowed_identifiers if allowed_identifiers is not None else set()

    def __call__(self, **kwargs) -> str:
        raise NotImplementedError


class ListTables(Tool):
    name = "list tables"
    description = "Lists all tables in the database."
    params_schema = {}

    def __call__(self) -> str:
        cur = self.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY 1"
        )
        rows = [r[0] for r in cur.fetchall()]
        self.allowed_identifiers.update(rows)
        return str(rows)


class DescribeTable(Tool):
    name = "describe table"
    description = "Describes a table. Args: {'table_name':'str'}"
    params_schema = {"table_name": "str"}

    def __call__(self, table_name: str) -> str:
        cur = self.conn.execute(f"PRAGMA table_info({table_name})")
        cols = cur.fetchall()
        headers = ["cid", "name", "type", "notnull", "dflt_value", "pk"]
        self.allowed_identifiers.add(table_name)
        for c in cols:
            self.allowed_identifiers.add(c[1])
            self.allowed_identifiers.add(f"{table_name}.{c[1]}")

        try:
            cnt = self.conn.execute(f"SELECT COUNT(1) FROM {table_name}").fetchone()[0]
        except Exception:
            cnt = "unknown"

        body = format_rows(cols, headers=headers, max_rows=100)
        return f"columns:\n{body}\nrow_count:{cnt}"


class QueryDatabase(Tool):
    name = "query database"
    description = "Runs a SELECT query safely."
    params_schema = {"query": "str"}

    def __call__(self, query: str) -> str:
        try:
            safe_sql = validate_select_query(query, self.allowed_identifiers, self.row_limit)
        except (ValidationError, ReadOnlyViolation) as e:
            return f"ERROR: {e}"

        try:
            def _run():
                cur = self.conn.execute(safe_sql)
                rows = cur.fetchall()
                headers = [d[0] for d in cur.description] if cur.description else []
                return (rows, headers)

            rows, headers = _call_with_timeout(_run, 10)
            return format_rows(rows, headers=headers, max_rows=20)
        except TimeoutError as e:
            return f"ERROR: {e}"
        except Exception as e:
            return f"ERROR: {e}"


# ================================================================
# ======================== agent.py ================================
# ================================================================

TRACE_FILE = "trace.log"

def _llm_provider():
    provider = os.getenv("LLM_PROVIDER", "openai").lower()
    if provider == "gemini":
        try:
            from google import genai
            client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
            return ("gemini", client)
        except Exception as e:
            raise RuntimeError("Gemini client unavailable. Set GEMINI_API_KEY and install google-genai.") from e
    elif provider == "anthropic":
        import anthropic
        return ("anthropic", anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"]))
    else:
        from openai import OpenAI
        return ("openai", OpenAI(api_key=os.environ["OPENAI_API_KEY"]))


SYSTEM_PROMPT = textwrap.dedent("""
You are a cautious SQL analysis agent. Follow this exact protocol:

FORMAT (strict):
THOUGHT: <brief plan>
ACTION: <tool name>{"arg": "value", ...}
OBSERVATION: <result>

Repeat Thought→Action→Observation until you can answer.
End with:
FINAL ANSWER: <concise answer only>

TOOLS:
- list tables: Lists all tables. Args: none. Returns: table names.
- describe table: Args: {"table_name":"str"}. Returns: columns/types (+ row count).
- query database: Args: {"query":"SELECT ... LIMIT N"}. Returns: rows as a small table.

RULES:
- Read-only. Only SELECT. No mutations. Keep queries small. Prefer LIMIT.
- If unsure of schema, call list tables or describe table first.
- Use exact FORMAT. Do not include extra text. No markdown in ACTION/OBSERVATION blocks.

IMPORTANT:
- Emit exactly ONE step per turn: THOUGHT then ACTION.
- Do NOT fabricate OBSERVATION. OBSERVATION will be appended by the system after your ACTION runs.
- Do NOT output FINAL ANSWER until after at least one OBSERVATION has been provided in this conversation.
""").strip()

FEWSHOTS = [
    {
        "role": "user",
        "content": "What columns are in the 'orders' table?"
    },
    {
        "role": "assistant",
        "content": "THOUGHT: I should describe the orders table to see its columns.\n"
                   "ACTION: describe table{\"table_name\":\"orders\"}\n"
                   "OBSERVATION: columns:\n[...omitted for brevity...]\nrow_count:200\n"
                   "FINAL ANSWER: The 'orders' table columns are: id, customer, order_date."
    },
    {
        "role": "user",
        "content": "Top 3 customers by total spend?"
    },
    {
        "role": "assistant",
        "content": "THOUGHT: I need to aggregate order_items by customer via a join.\n"
                   "ACTION: describe table{\"table_name\":\"order_items\"}\n"
                   "OBSERVATION: columns:\n[...]\nrow_count:...\n"
                   "THOUGHT: Now I can sum quantity*price grouped by customer.\n"
                   "ACTION: query database{\"query\":\"SELECT o.customer, SUM(oi.qty*oi.price) AS total "
                   "FROM order_items oi JOIN orders o ON o.id = oi.order_id "
                   "GROUP BY o.customer ORDER BY total DESC LIMIT 3\"}\n"
                   "OBSERVATION: customer  total\nAsha  12345.00\nHarsh  11223.50\nDivya  10890.25\n"
                   "FINAL ANSWER: Asha, Harsh, and Divya are the top 3 customers by total spend."
    }
]

ACTION_RX = re.compile(r"^ACTION:\s*([A-Za-z ]+)\s*(\{.*?\})\s*$", re.MULTILINE | re.DOTALL)
FINAL_RX  = re.compile(r"FINAL ANSWER:\s*(.+)", re.DOTALL)

class SQLReActAgent:
    def __init__(self, llm_client, provider: str, conn: sqlite3.Connection, max_steps=5, row_limit=100, time_budget_s=20):
        self.llm = llm_client
        self.provider = provider
        self.conn = conn
        self.max_steps = max_steps
        self.row_limit = row_limit
        self.time_budget_s = time_budget_s
        self.allowed_identifiers: set[str] = set()
        self.tools: Dict[str, Tool] = {}
        self.register_tools()

    def register_tools(self):
        self.tools = {
            "list tables": ListTables(self.conn, self.row_limit, self.allowed_identifiers),
            "describe table": DescribeTable(self.conn, self.row_limit, self.allowed_identifiers),
            "query database": QueryDatabase(self.conn, self.row_limit, self.allowed_identifiers),
        }

    def _call_llm(self, messages):
        if self.provider == "gemini":
            prompt_lines = []
            for m in messages:
                role = m["role"]
                if role == "system":
                    prompt_lines.append(f"System:\n{m['content']}\n")
                elif role == "user":
                    prompt_lines.append(f"User:\n{m['content']}\n")
                else:
                    prompt_lines.append(f"Assistant:\n{m['content']}\n")
            prompt = "\n".join(prompt_lines).strip()

            model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
            resp = self.llm.models.generate_content(model=model, contents=prompt)
            return resp.text
        elif self.provider == "anthropic":
            resp = self.llm.messages.create(
                model=os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022"),
                temperature=0,
                max_tokens=1024,
                messages=messages
            )
            return resp.content[0].text
        else:
            resp = self.llm.chat.completions.create(
                model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                temperature=0,
                messages=messages
            )
            return resp.choices[0].message.content

    def _build_messages(self, transcript: List[Dict[str, str]]) -> List[Dict[str, str]]:
        msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
        msgs.extend(FEWSHOTS)
        msgs.extend(transcript)
        return msgs

    def _parse_action_block(self, text: str):
        m = ACTION_RX.search(text)
        if not m:
            return None
        tool_name = m.group(1).strip().lower()
        args_raw = m.group(2)
        try:
            args = json.loads(args_raw)
        except json.JSONDecodeError:
            cleaned = args_raw.strip().strip("`")
            args = json.loads(cleaned)
        return tool_name, args

    def run(self, user_query: str) -> str:
        start = time.time()
        transcript = [{"role": "user", "content": user_query}]
        step = 0
        final_answer = None
        did_any_action = False

        log_to_file(TRACE_FILE, f"USER: {user_query}")

        while step < self.max_steps and (time.time() - start) < self.time_budget_s:
            step += 1
            content = self._call_llm(self._build_messages(transcript)).strip()
            log_to_file(TRACE_FILE, content)

            mfinal = FINAL_RX.search(content)
            if mfinal and not did_any_action:
                repair = ("FORMAT ERROR. Provide exactly one THOUGHT and one ACTION. "
                          "Do not give FINAL ANSWER until after an OBSERVATION from a tool.")
                transcript.append({"role": "assistant", "content": repair})
                log_to_file(TRACE_FILE, f"SYSTEM: {repair}")
                continue

            if mfinal and did_any_action:
                final_answer = mfinal.group(1).strip()
                log_to_file(TRACE_FILE, f"FINAL ANSWER: {final_answer}")
                break

            action = self._parse_action_block(content)
            if action:
                did_any_action = True
                tool_name, args = action
                tool = self.tools.get(tool_name)
                if not tool:
                    obs = f"ERROR: Unknown tool '{tool_name}'. Available: {list(self.tools.keys())}"
                else:
                    try:
                        obs = tool(**args)
                    except TypeError as e:
                        obs = f"ERROR: Bad arguments: {e}"
                    except Exception as e:
                        obs = f"ERROR: {e}"

                obs_text = f"OBSERVATION: {shorten(str(obs), 1800)}"
                log_to_file(TRACE_FILE, obs_text)
                transcript.append({"role": "assistant", "content": f"{content}\n{obs_text}"})
                if str(obs).startswith("ERROR:"):
                    final_answer = str(obs)
                    log_to_file(TRACE_FILE, f"FINAL ANSWER: {final_answer}")
                    break
                continue

            repair = ("FORMAT ERROR. Please re-emit one step: THOUGHT and ACTION only. "
                      "Wait for OBSERVATION before FINAL ANSWER.")
            transcript.append({"role": "assistant", "content": repair})
            log_to_file(TRACE_FILE, f"SYSTEM: {repair}")

        if final_answer is None:
            final_answer = (
                "I could not complete within the allotted steps/time. "
                "Please refine the question or ask me to list/describe the relevant tables first."
            )
        return final_answer


# ================================================================
# ======================== run.py ================================
# ================================================================

def main():
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True, help="Path to SQLite database file")
    ap.add_argument("--q", required=True, help="User question")
    args = ap.parse_args()

    uri = f"file:{args.db}?mode=ro"
    conn = sqlite3.connect(uri, uri=True, check_same_thread=False)

    provider, llm = _llm_provider()
    agent = SQLReActAgent(llm, provider, conn, max_steps=5, row_limit=100, time_budget_s=20)
    answer = agent.run(args.q)
    print(answer)


if __name__ == "__main__":
    main()