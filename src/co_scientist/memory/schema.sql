PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS sessions (
  id TEXT PRIMARY KEY,
  goal TEXT NOT NULL,
  config_json TEXT NOT NULL DEFAULT '{}',
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS research_plans (
  session_id TEXT PRIMARY KEY REFERENCES sessions(id) ON DELETE CASCADE,
  goal TEXT NOT NULL,
  preferences TEXT NOT NULL DEFAULT '[]',
  attributes TEXT NOT NULL DEFAULT '[]',
  constraints TEXT NOT NULL DEFAULT '[]',
  idea_attributes TEXT NOT NULL DEFAULT '[]',
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
  meta_review_round INTEGER,
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

CREATE INDEX IF NOT EXISTS idx_matches_session_pair
  ON matches(session_id, hypo_a_id, hypo_b_id);

CREATE TABLE IF NOT EXISTS elo_checkpoints (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
  match_id INTEGER REFERENCES matches(id) ON DELETE SET NULL,
  top_k INTEGER NOT NULL,
  avg_elo REAL NOT NULL,
  created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_elo_checkpoints_session_created
  ON elo_checkpoints(session_id, created_at DESC, id DESC);

CREATE TABLE IF NOT EXISTS hypothesis_embeddings (
  hypothesis_id INTEGER PRIMARY KEY REFERENCES hypotheses(id) ON DELETE CASCADE,
  session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
  embedding_json TEXT NOT NULL,
  updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_hypothesis_embeddings_session
  ON hypothesis_embeddings(session_id);

CREATE TABLE IF NOT EXISTS proximity_edges (
  session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
  hypo_a_id INTEGER NOT NULL REFERENCES hypotheses(id) ON DELETE CASCADE,
  hypo_b_id INTEGER NOT NULL REFERENCES hypotheses(id) ON DELETE CASCADE,
  similarity REAL NOT NULL,
  updated_at TEXT NOT NULL,
  PRIMARY KEY(session_id, hypo_a_id, hypo_b_id),
  CHECK(hypo_a_id < hypo_b_id)
);

CREATE INDEX IF NOT EXISTS idx_proximity_edges_session_similarity
  ON proximity_edges(session_id, similarity DESC);

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

CREATE TABLE IF NOT EXISTS citations (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  session_id TEXT REFERENCES sessions(id) ON DELETE CASCADE,
  dedupe_key TEXT,
  source TEXT NOT NULL,
  title TEXT NOT NULL,
  url TEXT,
  doi TEXT,
  pmid TEXT,
  arxiv_id TEXT,
  semantic_scholar_id TEXT,
  year INTEGER,
  raw_json TEXT NOT NULL DEFAULT '{}',
  created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_citations_session_source
  ON citations(session_id, source);

CREATE TABLE IF NOT EXISTS citation_links (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
  citation_id INTEGER NOT NULL REFERENCES citations(id) ON DELETE CASCADE,
  artifact_type TEXT NOT NULL,
  artifact_id INTEGER NOT NULL,
  source_task_id INTEGER REFERENCES tasks(id) ON DELETE SET NULL,
  evidence_index INTEGER NOT NULL,
  created_at TEXT NOT NULL,
  UNIQUE(session_id, citation_id, artifact_type, artifact_id, evidence_index)
);

CREATE INDEX IF NOT EXISTS idx_citation_links_artifact
  ON citation_links(session_id, artifact_type, artifact_id);

CREATE TABLE IF NOT EXISTS tool_cache (
  cache_key TEXT PRIMARY KEY,
  source TEXT NOT NULL,
  query TEXT NOT NULL,
  max_results INTEGER NOT NULL,
  options_hash TEXT NOT NULL,
  status TEXT NOT NULL,
  result_json TEXT NOT NULL,
  created_at TEXT NOT NULL,
  expires_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_tool_cache_expires
  ON tool_cache(expires_at);
