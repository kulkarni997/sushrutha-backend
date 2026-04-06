-- ============================================================
-- Sushrutha AI — Supabase Database Schema
-- ============================================================

-- ============================================================
-- 1. users
-- ============================================================
CREATE TABLE users (
    id                UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    email             TEXT        NOT NULL UNIQUE,
    hashed_password   TEXT        NOT NULL,
    role              TEXT        NOT NULL CHECK (role IN ('patient', 'doctor')),
    full_name         TEXT        NOT NULL,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ============================================================
-- 2. doctors
-- Extends users for doctor-specific fields.
-- ============================================================
CREATE TABLE doctors (
    id                  UUID        PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
    bams_number         TEXT        NOT NULL UNIQUE,
    verified            BOOLEAN     NOT NULL DEFAULT FALSE,
    subscription_active BOOLEAN     NOT NULL DEFAULT FALSE,
    specialization      TEXT,
    clinic_name         TEXT,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ============================================================
-- 3. scans
-- Medical images uploaded by authenticated patients.
-- ============================================================
CREATE TABLE scans (
    id          UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    patient_id  UUID        NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    doctor_id   UUID        REFERENCES users(id) ON DELETE SET NULL,
    scan_type   TEXT        NOT NULL CHECK (scan_type IN ('xray', 'mri', 'retina', 'skin', 'other')),
    image_url   TEXT        NOT NULL,
    status      TEXT        NOT NULL DEFAULT 'pending'
                            CHECK (status IN ('pending', 'processing', 'completed', 'failed')),
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_scans_patient_id ON scans(patient_id);
CREATE INDEX idx_scans_doctor_id  ON scans(doctor_id);

-- ============================================================
-- 4. guest_scans
-- Scans submitted by unauthenticated (guest) users.
-- ============================================================
CREATE TABLE guest_scans (
    id          UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id  TEXT        NOT NULL,
    scan_type   TEXT        NOT NULL CHECK (scan_type IN ('xray', 'mri', 'retina', 'skin', 'other')),
    image_url   TEXT        NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_guest_scans_session_id ON guest_scans(session_id);

-- ============================================================
-- 5. results
-- Diagnosis output for both authenticated and guest scans.
-- Exactly one of scan_id or guest_scan_id must be set.
-- ============================================================
CREATE TABLE results (
    id               UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    scan_id          UUID        REFERENCES scans(id) ON DELETE CASCADE,
    guest_scan_id    UUID        REFERENCES guest_scans(id) ON DELETE CASCADE,
    diagnosis        TEXT        NOT NULL,
    severity         TEXT        NOT NULL DEFAULT 'mild'
                                 CHECK (severity IN ('mild', 'moderate', 'severe')),
    confidence       FLOAT       CHECK (confidence >= 0.0 AND confidence <= 1.0),
    recommendations  TEXT,
    raw_output       JSONB,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now(),

    CONSTRAINT results_single_scan_source CHECK (
        (scan_id IS NOT NULL AND guest_scan_id IS NULL) OR
        (scan_id IS NULL AND guest_scan_id IS NOT NULL)
    )
);

CREATE INDEX idx_results_scan_id       ON results(scan_id);
CREATE INDEX idx_results_guest_scan_id ON results(guest_scan_id);

-- ============================================================
-- 6. pulse_readings
-- Biosignal data captured via the pulse/ECG route.
-- ============================================================
CREATE TABLE pulse_readings (
    id           UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    patient_id   UUID        NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    heart_rate   INTEGER     CHECK (heart_rate > 0 AND heart_rate < 300),
    spo2         FLOAT       CHECK (spo2 >= 0.0 AND spo2 <= 100.0),
    label        TEXT        CHECK (label IN ('normal', 'tachycardia', 'bradycardia', 'afib', 'unknown')),
    confidence   FLOAT       CHECK (confidence >= 0.0 AND confidence <= 1.0),
    raw_data     JSONB,
    recorded_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_pulse_readings_patient_id ON pulse_readings(patient_id);

-- ============================================================
-- 7. messages
-- Doctor-patient direct messages.
-- ============================================================
CREATE TABLE messages (
    id           UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    sender_id    UUID        NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    receiver_id  UUID        NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    content      TEXT        NOT NULL,
    is_read      BOOLEAN     NOT NULL DEFAULT FALSE,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),

    CONSTRAINT messages_no_self_message CHECK (sender_id <> receiver_id)
);

CREATE INDEX idx_messages_sender_id   ON messages(sender_id);
CREATE INDEX idx_messages_receiver_id ON messages(receiver_id);

-- ============================================================
-- 8. notifications
-- In-app notifications for any user.
-- ============================================================
CREATE TABLE notifications (
    id          UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id     UUID        NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    title       TEXT        NOT NULL,
    body        TEXT        NOT NULL,
    is_read     BOOLEAN     NOT NULL DEFAULT FALSE,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_notifications_user_id ON notifications(user_id);
