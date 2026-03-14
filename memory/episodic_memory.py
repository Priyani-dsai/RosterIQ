import sqlite3
from datetime import datetime
import math

class EpisodicMemory:


   def __init__(self, db_path="memory/episodic_memory.db"):

    self.conn = sqlite3.connect(db_path, check_same_thread=False)
    self.cursor = self.conn.cursor()

    self.cursor.execute("""
    CREATE TABLE IF NOT EXISTS memory (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_query TEXT,
        agent_response TEXT,
        timestamp TEXT,
        importance REAL
    )
    """)

    self.conn.commit()


   def store_interaction(self, query, response, base_importance=1.0):

    timestamp = datetime.now().isoformat()

    self.cursor.execute(
        "INSERT INTO memory (user_query, agent_response, timestamp, importance) VALUES (?, ?, ?, ?)",
        (query, response, timestamp, base_importance)
    )

    self.conn.commit()


   def fetch_recent(self, limit=5, decay_lambda=0.05):

    self.cursor.execute(
        "SELECT user_query, agent_response, timestamp, importance FROM memory"
    )

    rows = self.cursor.fetchall()

    scored_memories = []

    now = datetime.now()

    for q, a, t, importance in rows:

        past_time = datetime.fromisoformat(t)

        delta_seconds = (now - past_time).total_seconds()

        # exponential decay based on time difference
        decayed_score = importance * math.exp(-decay_lambda * delta_seconds / 3600)

        scored_memories.append((decayed_score, q, a))


    # sort memories by decayed importance
    scored_memories.sort(reverse=True, key=lambda x: x[0])

    selected = scored_memories[:limit]

    history = []

    for _, q, a in selected:
        history.append(f"User: {q}\nAgent: {a}")

    return "\n".join(history)

