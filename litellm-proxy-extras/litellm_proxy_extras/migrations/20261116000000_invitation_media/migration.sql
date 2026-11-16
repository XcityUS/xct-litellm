ALTER TABLE "LiteLLM_AssetVerification"
    ADD COLUMN IF NOT EXISTS photo_url TEXT,
    ADD COLUMN IF NOT EXISTS photo_asset_id TEXT,
    ADD COLUMN IF NOT EXISTS photo_submitted_at TIMESTAMPTZ(6);
