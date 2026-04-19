CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE documents (
    id          SERIAL PRIMARY KEY,
    service     TEXT NOT NULL,
    service_id  TEXT NOT NULL,
    meta        JSONB,
    text        TEXT,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),

    UNIQUE (service, service_id)
);

CREATE TABLE files (
    id          SERIAL PRIMARY KEY,
    document_id INT NOT NULL REFERENCES documents (id) ON DELETE CASCADE,
    service_id  TEXT,
    kind        TEXT,
    text        TEXT,
    meta        JSONB
);

CREATE TABLE embeddings (
    id          SERIAL PRIMARY KEY,
    document_id INT NOT NULL REFERENCES documents (id) ON DELETE CASCADE,
    chunk_index INT NOT NULL,
    chunk_text  TEXT NOT NULL,
    embedding   VECTOR(1024)
);

