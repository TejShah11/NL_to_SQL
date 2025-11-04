# 🧠 SQL ReAct Agent — Safe and Explainable LLM-Based SQL Interface

---

## 🎯 Purpose

The **SQL ReAct Agent** is a reasoning-driven system that enables natural-language interaction with structured data.
It employs a **Large Language Model (LLM)** — such as **Gemini 2.5 Flash**, **OpenAI GPT-4o**, or **Anthropic Claude 3.5** — to interpret user queries and generate **safe, read-only SQL** statements on a SQLite database.

The agent follows a structured reasoning cycle:

> **THOUGHT → ACTION → OBSERVATION → FINAL ANSWER**

This makes every query **transparent, auditable, and reproducible**.
A built-in safety layer enforces **read-only SQL**, automatic query limiting, and detailed trace logging.

---

## ⚙️ Setup and Requirements

### Prerequisites

* **Python ≥ 3.10**
* Required libraries (install via):

  ```bash
  pip install -r requirements.txt
  ```

Typical dependencies include: `tabulate`, `python-dotenv`, and one LLM SDK such as `google-genai`, `openai`, or `anthropic`.

---

### Environment Configuration

Create a `.env` file in the project root containing:

```bash
LLM_PROVIDER=gemini        # Options: gemini / openai / anthropic
GEMINI_API_KEY=your_key_here
GEMINI_MODEL=gemini-2.5-flash
OPENAI_API_KEY=your_key_here
ANTHROPIC_API_KEY=your_key_here
```

This ensures proper authentication and model selection at runtime.

---

## ▶️ Running the Agent

Run the agent using:

```bash
python main.py --db shop.db --q "Top 3 customers by total spend"
```

### Command Arguments

| Argument | Description                                              |
| -------- | -------------------------------------------------------- |
| `--db`   | Path to the SQLite database file (read-only mode).       |
| `--q`    | Natural-language question to query against the database. |

The result is printed in the console and a detailed reasoning trace is written to `trace.log`.

---

## 🧩 Available Tools

| Tool Name          | Arguments               | Function / Return Value                                                |
| ------------------ | ----------------------- | ---------------------------------------------------------------------- |
| **list tables**    | None                    | Lists all available tables in the database.                            |
| **describe table** | `{"table_name": "str"}` | Returns column names, data types, and row count for the given table.   |
| **query database** | `{"query": "SELECT …"}` | Executes a validated read-only SQL query (adds or enforces a `LIMIT`). |

Each tool is sandboxed to ensure **read-only** access — blocking `INSERT`, `UPDATE`, `DELETE`, and schema-altering commands.

---

## 💻 Example Run

**Command**

```bash
python main.py --db shop.db --q "Top 3 customers by total spend"
```

**Console Output**

```
Divya, Esha, and Chirag are the top 3 customers by total spend.
```

**Trace Excerpt (`trace.log`)**

```
[2025-10-29 18:10:02] USER: Top 3 customers by total spend
[2025-10-29 18:10:04] THOUGHT: Join orders and order_items to sum total spending by customer.
[2025-10-29 18:10:04] ACTION: query database{"query":"SELECT o.customer, SUM(oi.qty*oi.price) AS total
FROM order_items oi JOIN orders o ON o.id = oi.order_id
GROUP BY o.customer ORDER BY total DESC LIMIT 3"}
[2025-10-29 18:10:05] OBSERVATION: customer  total
Asha   12345.00
Harsh  11223.50
Divya  10890.25
[2025-10-29 18:10:05] FINAL ANSWER: Asha, Harsh, and Divya are the top 3 customers by total spend.
```

---

## 📁 Project Structure

| File        | Description                                                                                  |
| ----------- | -------------------------------------------------------------------------------------------- |
| `main.py`   | Unified implementation combining the agent, tools, safety checks, utilities, and CLI runner. |
| `.env`      | Stores API keys and LLM configuration.                                                       |
| `trace.log` | Logs every reasoning step and tool interaction.                                              |
| `shop.db` | Example SQLite database for testing and demonstration.                                       |

---

## 👥 Authors

* **Harsh Shah** — UID: 2022300105
* **Tej Shah** — UID: 2022300106

---

## 📚 Summary

The SQL ReAct Agent demonstrates how structured reasoning and safety constraints can transform LLMs into **reliable and explainable database assistants**.
By combining symbolic validation with neural reasoning, it bridges natural-language understanding and precise SQL generation.
This foundation can be extended to support **multi-database querying**, **adaptive schema summarization**, and **context-aware reasoning** for production-grade intelligent data systems.

---
