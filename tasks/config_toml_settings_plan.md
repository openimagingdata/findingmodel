# Plan: Split Secrets (env/.env) and Non‑Secrets (config.toml) with Pydantic Settings

## Goal
Use Pydantic Settings to load **non‑secret configuration** from a shared `config.toml` file (sectioned per package), while **secrets remain in environment variables / `.env`**. Env values should override TOML.

Packages in scope:
- `findingmodel`
- `findingmodel-ai`
- `anatomic-locations`
- `oidm-common`
- `oidm-maintenance`

## Desired Behavior
- `config.toml` contains open config values (paths, URLs, model names, etc.).
- `.env` contains only secrets (API keys, tokens, credentials).
- All settings classes pull from both sources with **env precedence**.
- One shared base settings class handles TOML + env source ordering.
- No breaking change for users who only set `.env` (env still works).

---

## Phase 1 — Design the TOML schema + base settings source

### Tasks
1. Define a standard TOML schema with one section per package:
   ```toml
   [findingmodel]
   duckdb_index_path = "/data/findingmodels.duckdb"
   remote_manifest_url = "https://findingmodelsdata.t3.storage.dev/manifest.json"
   openai_embedding_model = "text-embedding-3-small"
   openai_embedding_dimensions = 512

   [anatomic_locations]
   manifest_url = "https://anatomiclocationdata.t3.storage.dev/manifest.json"

   [findingmodel_ai]
   default_model = "openai:gpt-5-mini"
   default_model_full = "openai:gpt-5.2"
   default_model_small = "openai:gpt-5-nano"
   tavily_search_depth = "advanced"
   ollama_base_url = "http://localhost:11434/v1"

   [oidm_maintenance]
   s3_endpoint_url = "https://fly.storage.tigris.dev"
   s3_bucket = "findingmodelsdata"
   anatomic_s3_bucket = "anatomiclocationdata"
   ```
2. Decide a file location strategy:
   - Default to `config.toml` in repo root.
   - Allow override via env var `OIDM_CONFIG_PATH`.
3. Implement a shared base settings class (e.g., `oidm_common/settings.py`) with:
   - `BaseSettings` subclass
   - TOML source that selects a single section from the file
   - `settings_customise_sources()` order: `init > env > dotenv > toml > file_secrets`

### Implementation notes (Pydantic Settings v2.11.0)
- Use `TomlConfigSettingsSource` to parse TOML.
- Wrap it to select only the `toml_section` for each package.
- Keep `env_file=".env"` and `env_nested_delimiter="__"` consistent with existing usage.

### Phase 1 completion rubric (objective)
- A shared settings base class exists and can be imported.
- `settings_customise_sources()` returns sources with TOML included **after** env/dotenv.
- The TOML wrapper extracts one section only (not the whole file).
- `OIDM_CONFIG_PATH` is documented as the config file override.

---

## Phase 2 — Integrate settings classes in each package

### Tasks
1. Update each package’s settings class to inherit from the shared base class.
2. Set `toml_section` per package:
   - `findingmodel` → `"findingmodel"`
   - `anatomic-locations` → `"anatomic_locations"`
   - `findingmodel-ai` → `"findingmodel_ai"`
   - `oidm-maintenance` → `"oidm_maintenance"`
3. Preserve existing env prefixes where they already exist (e.g., `ANATOMIC_`, `OIDM_MAINTAIN_`).
4. Keep all secrets as `SecretStr` and ensure they still load from env.

### Phase 2 completion rubric (objective)
- Each package settings class inherits from the shared base class.
- Each settings class has a `toml_section` string that matches its TOML section.
- Env-only values still work (no settings regressions in code paths that only use `.env`).
- No references to TOML in code outside the shared base class.

---

## Phase 3 — Enforce “secrets only in env” (optional but recommended)

### Tasks
1. Add a validator in the shared base class (or per class) that rejects `SecretStr` fields loaded from TOML.
2. Provide a clear error message listing any secret fields populated from TOML.

### Phase 3 completion rubric (objective)
- If a `SecretStr` field is set in `config.toml`, instantiation raises a clear exception.
- Secret fields still load normally from env/.env.

---

## Phase 4 — Documentation and samples

### Tasks
1. Add `config.sample.toml` in repo root with non‑secret settings only.
2. Update `.env.sample` to include only secrets (remove non‑secret values).
3. Update docs to reflect the split:
   - `docs/configuration.md`
   - package READMEs (findingmodel, findingmodel‑ai, anatomic‑locations)
   - any doc sections that currently describe non‑secret `.env` usage
4. Add a minimal “enable config.toml” snippet:
   - “Create `config.toml` with package sections.”
   - “Set `OIDM_CONFIG_PATH` to point to it.”

### Phase 4 completion rubric (objective)
- `config.sample.toml` exists and contains only non‑secret values.
- `.env.sample` contains only secrets.
- Docs clearly indicate TOML vs env responsibilities.
- Docs mention `OIDM_CONFIG_PATH`.

---

## Phase 5 — Smoke checks

### Tasks
1. Verify a minimal TOML works for each package without env overrides.
2. Verify env overrides TOML for at least one field per package.
3. Verify missing secrets are still caught where required (existing validation behavior).

### Phase 5 completion rubric (objective)
- Instantiating each settings class with TOML only yields expected non‑secret values.
- Setting a conflicting env var overrides TOML.
- Secrets are still required only via env/.env and fail with clear errors when missing.

---

## Acceptance Criteria (overall)
- All packages read open configuration from `config.toml` sections.
- All secrets are expected from env/`.env` only.
- Env overrides TOML consistently across packages.
- Documentation and samples reflect the split.
- No breaking change for users who rely on `.env` only.

---

## Implementation Hints (for assigned agent)
- Search for settings classes:
  - `rg -n "BaseSettings|SettingsConfigDict" packages`
- Files likely to touch:
  - `packages/*/src/*/config.py`
  - `packages/oidm-common/src/oidm_common/` (new shared base)
  - `docs/configuration.md`
  - `README.md` and package READMEs
  - `.env.sample` (trim to secrets)
- Pydantic Settings version is 2.11.0 (TOML source available).
