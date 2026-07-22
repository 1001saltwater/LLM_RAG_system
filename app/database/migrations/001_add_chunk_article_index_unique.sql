DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'uq_chunk_article_index'
    ) THEN
        ALTER TABLE chunk
        ADD CONSTRAINT uq_chunk_article_index
        UNIQUE (article_id, chunk_index);
    END IF;
END
$$;
