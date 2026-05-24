PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS sessions (
  id TEXT PRIMARY KEY,
  goal TEXT NOT NULL,
  config_json TEXT NOT NULL DEFAULT '{}',
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS hypotheses (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
  content TEXT NOT NULL,
  summary TEXT NOT NULL,
  detailed_description TEXT,
  mechanism TEXT,
  impacted_pathways TEXT NOT NULL DEFAULT '[]',
  experimental_plan TEXT,
  safety_notes TEXT,
  testable_predictions TEXT NOT NULL DEFAULT '[]',
  elo INTEGER NOT NULL DEFAULT 1200,
  parent_ids TEXT NOT NULL DEFAULT '[]',
  source_strategy TEXT,
  created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_hypotheses_session_elo
  ON hypotheses(session_id, elo DESC);

CREATE TABLE IF NOT EXISTS reviews (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
  hypothesis_id INTEGER NOT NULL REFERENCES hypotheses(id) ON DELETE CASCADE,
  type TEXT NOT NULL,
  score REAL,
  content TEXT NOT NULL,
  created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_reviews_hypothesis
  ON reviews(hypothesis_id);

CREATE TABLE IF NOT EXISTS matches (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
  hypo_a_id INTEGER NOT NULL REFERENCES hypotheses(id) ON DELETE CASCADE,
  hypo_b_id INTEGER NOT NULL REFERENCES hypotheses(id) ON DELETE CASCADE,
  winner_id INTEGER NOT NULL REFERENCES hypotheses(id) ON DELETE CASCADE,
  transcript TEXT NOT NULL,
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS tasks (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
  agent TEXT NOT NULL,
  action TEXT NOT NULL,
  target_id INTEGER,
  priority INTEGER NOT NULL,
  status TEXT NOT NULL,
  payload_json TEXT NOT NULL DEFAULT '{}',
  result_json TEXT,
  error TEXT,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_tasks_session_status_priority
  ON tasks(session_id, status, priority DESC);

CREATE TABLE IF NOT EXISTS feedback (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
  round INTEGER NOT NULL,
  content TEXT NOT NULL,
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS overview (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
  round INTEGER NOT NULL,
  content TEXT NOT NULL,
  top_hypothesis_ids TEXT NOT NULL DEFAULT '[]',
  created_at TEXT NOT NULL
);
