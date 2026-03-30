-- samples from the datasets
CREATE TABLE IF NOT EXISTS samples (
    id                 SERIAL PRIMARY KEY,
    chart_source       TEXT NOT NULL, -- dataset name
    chart_type         TEXT NOT NULL, -- e.g. "bar", "line"
    sample_question    TEXT NOT NULL, -- from the dataset
	sample_answer      TEXT NOT NULL,

    raw_chart          UUID NOT NULL, -- without preprocessing
    original_width     INTEGER,
    original_height    INTEGER,
    preprocess_meta    JSONB,         -- preprocessing metadata, no fixed format
    clean_chart        UUID,          -- cleaned by preprocessing module

    created_at         TIMESTAMP NOT NULL DEFAULT NOW()
);

-- clean answers associated with each chart
CREATE TABLE IF NOT EXISTS clean_answers (
    id                 SERIAL PRIMARY KEY,
    sample_id          INTEGER NOT NULL REFERENCES samples(id) ON DELETE CASCADE,

    clean_answer_model TEXT NOT NULL,
    clean_answer       TEXT NOT NULL,

    created_at         TIMESTAMP NOT NULL DEFAULT NOW(),

    UNIQUE (sample_id, clean_answer_model)
);
 
-- target answers associated with each clean answer
CREATE TABLE IF NOT EXISTS target_answers (
    id               SERIAL PRIMARY KEY,
    clean_answer_id  INTEGER NOT NULL REFERENCES clean_answers(id) ON DELETE CASCADE,

    target_answer    TEXT NOT NULL,
    target_strategy  TEXT,

    created_at       TIMESTAMP NOT NULL DEFAULT NOW(),

    UNIQUE (clean_answer_id, target_strategy)
);

-- adversarial charts associated with each clean chart/target answer pair
CREATE TABLE IF NOT EXISTS adversarial_charts (
    id                SERIAL PRIMARY KEY,
    target_answer_id  INTEGER NOT NULL REFERENCES target_answers(id) ON DELETE CASCADE,

    adversarial_chart UUID NOT NULL,
    attack_method     TEXT NOT NULL, -- the attack method used, e.g. "AttackVLM"
    attack_surrogate  TEXT NOT NULL, -- the model uses as surrogate in the attack, e.g. "CLIP"
    attack_meta       JSONB,

    created_at        TIMESTAMP NOT NULL DEFAULT NOW(),

    UNIQUE (target_answer_id, attack_method, target_surrogate)
);

-- response from an eval model on each adversarial graph
CREATE TABLE IF NOT EXISTS adversarial_answers (
    id                       SERIAL PRIMARY KEY,
    adversarial_chart_id     INTEGER NOT NULL REFERENCES adversarial_charts(id) ON DELETE CASCADE,

    adversarial_answer_model TEXT NOT NULL,
    answer_text              TEXT NOT NULL,

    attack_succeeded         BOOLEAN NOT NULL,
	eval_meta                JSONB,

    created_at               TIMESTAMP NOT NULL DEFAULT NOW(),

    UNIQUE (adversarial_chart_id, adversarial_answer_model)
);

CREATE INDEX idx_clean_answers_sample
ON clean_answers(sample_id);

CREATE INDEX idx_target_answers_clean
ON target_answers(clean_answer_id);

CREATE INDEX idx_adv_charts_target
ON adversarial_charts(target_answer_id);

CREATE INDEX idx_adv_answers_chart
ON adversarial_answers(adversarial_chart_id);
