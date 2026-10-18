"""
任務資料庫操作。

負責：
1. 建立 SQLite 資料表
2. 保存 API 任務
3. 實作 platform_task_id 冪等性
4. 更新 PENDING / PROCESSING / SUCCEEDED / FAILED
5. 清除超過 24 小時的結果
"""

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import aiosqlite


# 專案根目錄。避免從不同工作目錄啟動 Uvicorn 時，
# SQLite 被建立在無法預期的位置。
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 可以透過環境變數覆蓋資料庫位置。
_configured_db_path = os.getenv("TASK_DB_PATH")
DB_PATH = (
    Path(_configured_db_path).expanduser()
    if _configured_db_path
    else PROJECT_ROOT / "runtime" / "tasks.db"
)

# 任務完成後保留幾小時
TASK_TTL_HOURS = int(
    os.getenv("TASK_TTL_HOURS", "24")
)


CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS tasks (
    task_id TEXT PRIMARY KEY,
    platform_task_id TEXT NOT NULL UNIQUE,
    status TEXT NOT NULL,
    progress INTEGER,
    request_json TEXT NOT NULL,
    result_json TEXT,
    error_json TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    completed_at TEXT
);
"""


class TaskPlantConflict(Exception):
    """同一平台任務編號不可改用另一植物模型。"""


def utc_now() -> datetime:
    """取得包含時區的 UTC 現在時間。"""
    return datetime.now(timezone.utc)


def utc_now_string() -> str:
    """取得適合存入 SQLite 的時間字串。"""
    return utc_now().isoformat()


def serialize_request(request: Any) -> dict:
    """
    將 Pydantic model 或一般 dict 轉成可儲存的 dict。

    CreateTaskRequest 是 Pydantic model，
    model_dump(mode="json") 可以處理 datetime 與 HttpUrl。
    """
    if hasattr(request, "model_dump"):
        return request.model_dump(mode="json")

    if isinstance(request, dict):
        return request

    raise TypeError("request 必須是 Pydantic model 或 dict")


def row_to_task(row: aiosqlite.Row) -> dict:
    """將 SQLite Row 轉成程式使用的 task dict。"""
    task = dict(row)

    request_json = task.pop("request_json", None)
    result_json = task.pop("result_json", None)
    error_json = task.pop("error_json", None)

    task["request"] = (
        json.loads(request_json)
        if request_json
        else None
    )
    task["result"] = (
        json.loads(result_json)
        if result_json
        else None
    )
    task["error"] = (
        json.loads(error_json)
        if error_json
        else None
    )

    return task


async def fetch_one(
    db: aiosqlite.Connection,
    sql: str,
    parameters: tuple = (),
):
    """執行查詢並取得第一筆資料。"""
    async with db.execute(sql, parameters) as cursor:
        return await cursor.fetchone()


async def init_db() -> None:
    """
    初始化資料庫。

    FastAPI 啟動時執行一次。
    runtime/ 不存在時會自動建立。
    """
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    async with aiosqlite.connect(str(DB_PATH), timeout=5.0) as db:
        # WAL 讓讀取和寫入比較不容易互相阻塞
        await db.execute("PRAGMA journal_mode=WAL;")

        await db.execute(CREATE_TABLE_SQL)

        # 加速依狀態及完成時間查詢
        await db.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_tasks_status
            ON tasks(status);
            """
        )

        await db.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_tasks_completed_at
            ON tasks(completed_at);
            """
        )

        await db.commit()


async def create_or_get_task(request: Any) -> tuple[dict, bool]:
    """
    建立任務，或取得相同 platform_task_id 的既有任務。

    Returns:
        (task, is_new)

        is_new=True：
            本次建立了新任務，應放入 worker queue。

        is_new=False：
            platform_task_id 已存在，不可重複推論。
    """
    payload = serialize_request(request)
    platform_task_id = payload["platform_task_id"]

    new_task_id = f"acad_{uuid4().hex[:12]}"
    now = utc_now_string()

    request_json = json.dumps(
        payload,
        ensure_ascii=False,
    )

    async with aiosqlite.connect(str(DB_PATH), timeout=5.0) as db:
        db.row_factory = aiosqlite.Row

        # 鎖定寫入交易，避免兩個相同請求同時建立任務
        await db.execute("BEGIN IMMEDIATE;")

        existing = await fetch_one(
            db,
            """
            SELECT *
            FROM tasks
            WHERE platform_task_id = ?
            """,
            (platform_task_id,),
        )

        if existing is not None:
            stored = row_to_task(existing)
            old_plant = stored["request"].get("task", {}).get("plant", "annona")
            new_plant = payload.get("task", {}).get("plant", "annona")
            if old_plant != new_plant:
                raise TaskPlantConflict(platform_task_id)
            await db.commit()
            return stored, False

        await db.execute(
            """
            INSERT INTO tasks (
                task_id,
                platform_task_id,
                status,
                progress,
                request_json,
                result_json,
                error_json,
                created_at,
                updated_at,
                completed_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                new_task_id,
                platform_task_id,
                "PENDING",
                0,
                request_json,
                None,
                None,
                now,
                now,
                None,
            ),
        )

        created = await fetch_one(
            db,
            """
            SELECT *
            FROM tasks
            WHERE task_id = ?
            """,
            (new_task_id,),
        )

        await db.commit()

    return row_to_task(created), True


async def get_task(task_id: str) -> dict | None:
    """
    依 task_id 查詢任務。

    如果任務完成已超過 24 小時：
    1. 刪除任務
    2. 回傳 None
    3. API 層再將 None 轉成 TASK_NOT_FOUND
    """
    async with aiosqlite.connect(str(DB_PATH), timeout=5.0) as db:
        db.row_factory = aiosqlite.Row

        row = await fetch_one(
            db,
            """
            SELECT *
            FROM tasks
            WHERE task_id = ?
            """,
            (task_id,),
        )

        if row is None:
            return None

        task = row_to_task(row)
        completed_at = task.get("completed_at")

        if completed_at is not None:
            completed_time = datetime.fromisoformat(completed_at)
            expires_time = completed_time + timedelta(
                hours=TASK_TTL_HOURS
            )

            if utc_now() >= expires_time:
                await db.execute(
                    """
                    DELETE FROM tasks
                    WHERE task_id = ?
                    """,
                    (task_id,),
                )
                await db.commit()
                return None

        return task


async def mark_processing(
    task_id: str,
    progress: int = 0,
) -> None:
    """將任務更新成 PROCESSING。"""
    safe_progress = max(0, min(int(progress), 100))

    async with aiosqlite.connect(str(DB_PATH), timeout=5.0) as db:
        await db.execute(
            """
            UPDATE tasks
            SET
                status = 'PROCESSING',
                progress = ?,
                updated_at = ?
            WHERE task_id = ?
            """,
            (
                safe_progress,
                utc_now_string(),
                task_id,
            ),
        )
        await db.commit()


async def mark_succeeded(
    task_id: str,
    result: dict,
) -> None:
    """將任務更新成 SUCCEEDED，並保存模型結果。"""
    now = utc_now_string()

    result_json = json.dumps(
        result,
        ensure_ascii=False,
    )

    async with aiosqlite.connect(str(DB_PATH), timeout=5.0) as db:
        await db.execute(
            """
            UPDATE tasks
            SET
                status = 'SUCCEEDED',
                progress = 100,
                result_json = ?,
                error_json = NULL,
                updated_at = ?,
                completed_at = ?
            WHERE task_id = ?
            """,
            (
                result_json,
                now,
                now,
                task_id,
            ),
        )
        await db.commit()


async def mark_failed(
    task_id: str,
    error: dict,
) -> None:
    """將任務更新成 FAILED，並保存錯誤內容。"""
    now = utc_now_string()

    error_json = json.dumps(
        error,
        ensure_ascii=False,
    )

    async with aiosqlite.connect(str(DB_PATH), timeout=5.0) as db:
        await db.execute(
            """
            UPDATE tasks
            SET
                status = 'FAILED',
                error_json = ?,
                result_json = NULL,
                updated_at = ?,
                completed_at = ?
            WHERE task_id = ?
            """,
            (
                error_json,
                now,
                now,
                task_id,
            ),
        )
        await db.commit()


async def list_unfinished_tasks() -> list[dict]:
    """
    取得尚未完成的任務。

    若服務上次在 PROCESSING 時中斷，
    啟動後先改回 PENDING，讓 worker 重新處理。
    """
    now = utc_now_string()

    async with aiosqlite.connect(str(DB_PATH), timeout=5.0) as db:
        db.row_factory = aiosqlite.Row

        await db.execute(
            """
            UPDATE tasks
            SET
                status = 'PENDING',
                progress = 0,
                updated_at = ?
            WHERE status = 'PROCESSING'
            """,
            (now,),
        )

        async with db.execute(
            """
            SELECT *
            FROM tasks
            WHERE status = 'PENDING'
            ORDER BY created_at ASC
            """
        ) as cursor:
            rows = await cursor.fetchall()

        await db.commit()

    return [row_to_task(row) for row in rows]


async def delete_expired_tasks() -> int:
    """
    刪除完成時間超過 24 小時的任務。

    Returns:
        實際刪除的任務數量。
    """
    cutoff = (
        utc_now() - timedelta(hours=TASK_TTL_HOURS)
    ).isoformat()

    async with aiosqlite.connect(str(DB_PATH), timeout=5.0) as db:
        cursor = await db.execute(
            """
            DELETE FROM tasks
            WHERE completed_at IS NOT NULL
              AND completed_at <= ?
            """,
            (cutoff,),
        )

        deleted_count = cursor.rowcount
        await db.commit()

    return deleted_count
