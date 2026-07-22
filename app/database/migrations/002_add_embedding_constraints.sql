DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'uq_embedding_chunk_id'
    ) THEN
        ALTER TABLE embedding
        ADD CONSTRAINT uq_embedding_chunk_id
        UNIQUE (chunk_id);
    END IF;
END
$$;

ALTER TABLE embedding
DROP CONSTRAINT IF EXISTS embedding_chunk_id_fkey;

ALTER TABLE embedding
ADD CONSTRAINT embedding_chunk_id_fkey
FOREIGN KEY (chunk_id)
REFERENCES chunk (id)
ON DELETE CASCADE;
