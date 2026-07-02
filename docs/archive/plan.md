# DJIA Implementation Plan

## Overview

**Goal:** Build an analytics system that extracts audio features (BPM, key, mood, structure) from techno tracks and provides DJ mixing insights.

**Pipeline Architecture:**
```
Audio Files → Ingestion → DSP Features → AI Vibe Classification → SQLite Store → Traktor Export
```

**Success Criteria:**
- Process audio files from `data/` without crashes
- Extract accurate BPM, key, and structural features
- Generate mood/vibe classifications
- Export compatible data to Traktor NML format

---

## Phase 1: Ingestion & Library Scanning *(Weeks 1)*

**Deliverables:**
- `src/ingestion/scanner.py` — recursive directory scanner and file validation
- `src/ingestion/loader.py` — audio file decoder and normalizer
- `tests/test_ingestion.py` — unit tests for file loading

**Tasks:**
1. Create `src/ingestion/` module structure
2. Implement directory scanner (pathlib) to find `.mp3`, `.wav`, `.aiff`, `.flac` files
3. Implement audio loader using `librosa` or `soundfile`:
   - Read metadata with `mutagen` (Artist, Title, Album)
   - Resample to mono at 22,050 Hz for consistency
   - Validate file integrity (handle corrupted files gracefully)
4. Add logging for processed/skipped files
5. Write unit tests for file detection and loading

**Dependencies:** None (phase 1 prerequisite)

**Success Criteria:**
- Loads all audio files from `data/` without errors
- Metadata extraction works for MP3/WAV files
- Resampling produces consistent mono output

---

## Phase 2: DSP Core (Feature Extraction) *(Weeks 2-3)*

**Overview:** 4-step analysis pipeline that extracts the complete "DNA" of a track. Each step builds on librosa + scipy to extract geometric, rhythmic, harmonic, and semantic features.

**Deliverables:**
- `src/dsp/extractor.py` — Master pipeline orchestrating all 4 steps
- `src/dsp/phrasing_engine.py` — Step 1: Structural & Geometric Analysis
- `src/dsp/groove_engine.py` — Step 2: Temporal & Micro-Rhythmic Analysis
- `src/dsp/mood_engine.py` — Step 3: Spectral & Harmonic Analysis
- `src/dsp/curation_engine.py` — Step 4: High-Level Semantic Analysis
- `src/features/schema.py` — standardized feature data structure
- `tests/test_dsp.py` — feature extraction tests

**Dependencies:** Phase 1 (ingestion)

**Success Criteria (Phase 2 Overall):**
- All 4 engines extract without crashes
- Hot-cue timestamps automatically generate (within ±500ms of manual)
- BPM detection within ±2% of ground truth
- Key detection matches manual assessment (>85% agreement)
- Energy curves visualize correctly

---

### 2.1: Structural & Geometric Analysis (The "Phrasing" Engine) 🟨

**What Analysis to Run:**
- Novelty Curve via `librosa.effects.percussive_harmonic_source_separation()`
- Self-Similarity Matrix to detect structural boundaries
- Onset Detection Edge Detection

**What Python Looks For:**
- Sudden changes in frequency energy, rhythmic texture, or harmonic shifts
- Peaks in the novelty curve = section boundaries

**Output (The Actionable Data):**
- **Exact Timestamps of Section Changes:** List of times (e.g., `[0.00, 60.95, 121.90, 182.85]`) showing:
  - Where Intro ends
  - Where Breakdown drops
  - Where Outro begins
- **Automation Blueprint:** Automatically calculate Hot Cue positions (Pad 1, Pad 2, Pad 4) without manual listening
- **Segment Labels:** Classify each section (intro, build, main, breakdown, outro)

**Implementation Sketch:**
```python
# Extract novelty curve (change detection)
novelty_curve = compute_novelty_curve(audio)
boundaries = detect_peaks(novelty_curve, threshold=0.3)
segments = [(timestamps[i], segment_label[i]) for i in range(len(boundaries))]
# Output: [(0.0, 'intro'), (60.95, 'build'), (121.90, 'main'), (182.85, 'outro')]
```

---

### 2.2: Temporal & Micro-Rhythmic Analysis (The "Groove" Engine) 🟪

**What Analysis to Run:**
- Onset Detection via `librosa.onset.onset_detect()`
- Dynamic Tempo Estimation via `librosa.beat.beat_track()`
- Swing/Groove Scoring from inter-beat distance variance

**What Python Looks For:**
- Sudden spikes in energy when a kick drum hits (transients)
- Deviations from the average beat grid (swing)

**Output (The Actionable Data):**
- **Exact Decimal BPM:** Instead of "126 BPM", output 126.04 BPM to prevent micro-drifting during long, hypnotic minimal transitions
- **Beat Grid & Onset Times:** Frame-level timestamps of every detected beat
- **Groove "Swing" Detection:** Score based on distance between primary beat and off-beat hi-hats
  - High swing score = groovy, bouncy, human-feel track
  - Low swing score = stiff, industrial, driving, robotic techno
- **Rhythmic Stability:** Flag tracks with tempo drift (problematic for DJ mixing)

**Implementation Sketch:**
```python
# Extract BPM and beat positions
bpm, beats = librosa.beat.beat_track(y=audio, sr=sr)
onsets = librosa.onset.onset_detect(y=audio, sr=sr)
swing_score = compute_swing(beats, onsets)  # Variance of inter-beat distance
groove_character = "bouncy" if swing_score > 0.5 else "stiff"
# Output: {bpm: 126.04, beats: [array], swing: 0.72, character: "bouncy"}
```

---

### 2.3: Spectral & Harmonic Analysis (The "Mood" Engine) 🟧

**What Analysis to Run:**
- Chromagram Extraction via `librosa.feature.chroma_stft()` or `chroma_cqt()`
- Spectral Centroid via `librosa.feature.spectral_centroid()`
- Harmonic-Percussive Source Separation (HPSS)

**What Python Looks For:**
- Distribution of pitch classes (notes) over time
- The "center of mass" of the frequency spectrum

**Output (The Actionable Data):**
- **Musical Key:** Root key (e.g., A minor or 7A in Camelot scale), enabling perfect harmonic mixing
- **Key Confidence Score:** How certain the algorithm is (some tracks are polytonal or don't fit traditional scales)
- **Brightness/Darkness Score:** Spectral Centroid ranges 0-1
  - **High score (0.7-1.0):** Crisp hi-hats, cutting synths, energetic open-air vibes
  - **Low score (0.0-0.3):** Deep, subby, muddy bass, perfect for 4:00 AM dark basement vibes
- **Harmonic Density:** Measure of harmonic complexity (useful for filtering minimal vs. rich tracks)

**Implementation Sketch:**
```python
# Extract chromagram and spectral properties
chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
key = detect_key_from_chroma(chroma)  # Root pitch class
camelot_key = convert_to_camelot(key)  # E.g., "7A"
spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
brightness = normalize(spectral_centroid)  # 0.0 (dark) to 1.0 (bright)
# Output: {key: "A minor", camelot: "7A", brightness: 0.78}
```

---

### 2.4: High-Level Semantic Analysis (The "Curation" Engine) 🟦

**What Analysis to Run:**
- Feature Classification & Complexity Scoring
- Energy Profiling via `librosa.feature.rms()` (Root Mean Square)
- Spectral Flux & Entropy measurement

**What Python Looks For:**
- Overall loudness variation and signal complexity
- Energy distribution across time (are there peaks or is it flat?)

**Output (The Actionable Data):**
- **"Danceability" Coefficient:** Percentage score of how steady and dance-friendly the rhythm is
  - High danceability = consistent kick drum, minimal variations
  - Low danceability = atmospheric, ambient, or chaotic breaks
- **Energy Curve Mapping:** Plot track's volume density over time
  - Flat loops = repetitive, hypnotic minimal (good for transitions)
  - Giant energy peaks = explosive drops, crowd-pleaser bangers
  - Gradual rise = build-up focused tracks
- **Semantic Tags:** Auto-generated labels (e.g., "high-energy", "steady-groove", "peak-heavy")
- **Playlist Filtering:** Quickly scan 100 tracks and separate:
  - Hypnotic loop tracks (flat energy curve)
  - Peak-driven tracks (big energy spikes)
  - Consistent grooves (steady energy)

**Implementation Sketch:**
```python
# Compute energy and rhythmic properties
rms_energy = librosa.feature.rms(y=audio)
danceability = compute_danceability(rms_energy, bpm, swing_score)  # 0.0-1.0
energy_curve = smooth(rms_energy)  # Over-time plot
energy_variance = np.var(energy_curve)
curve_type = "flat" if energy_variance < 0.1 else "dynamic"
# Output: {danceability: 0.82, energy_type: "dynamic", tags: ["high-energy", "peak-heavy"]}
```

---

**Phase 2 Architecture Diagram:**

```
Audio Input
    │
    ├──→ 🟨 Step 1: Phrasing Engine ──→ Section Timestamps, Hot Cues
    │
    ├──→ 🟪 Step 2: Groove Engine ──→ BPM, Beat Grid, Swing Score
    │
    ├──→ 🟧 Step 3: Mood Engine ──→ Key, Camelot Scale, Brightness
    │
    └──→ 🟦 Step 4: Curation Engine ──→ Danceability, Energy Curve, Tags
         │
         └──→ Combined Feature Vector (Track "DNA")
              │
              └──→ Store to Database (Phase 4)
```

**Testing Strategy:**
- Unit test each engine independently on sample minimal techno tracks
- Validate BPM against ground truth (Traktor analysis)
- Cross-check key detection with human ear
- Benchmark performance (should process 1 track in <30 seconds on CPU)

---

## Phase 3: AI Layer & Stem Separation *(Weeks 4-6)*

**Deliverables:**
- `src/ai/stem_separator.py` — Advanced Demucs/HTDemucs source separation
- `src/ai/classifier.py` — Deep learning mood/vibe classification
- `src/ai/segmentation.py` — Neural network-based structural analysis
- `src/ai/processor.py` — orchestrate Phase 2 DSP on separated stems
- `tests/test_ai.py` — integration tests

### 3.1: Advanced Source Separation (AI Demixing)

**Problem Solved:** Traditional DSP cannot isolate a vocal from a bassline once mixed. AI models trained on studio multi-tracks learn the spectral "signature" of isolated instruments.

**Tasks:**
1. Set up Demucs/HTDemucs (Meta’s source separation):
   - Install `demucs` or `spleeter` package
   - Implement stem splitting into Drums, Bass, Vocals, Melody
   - Configure for minimal bleeding and clean splits
   - Cache separated stems to disk to avoid re-processing
2. Post-processing on separated stems:
   - Normalize stem loudness for consistent analysis
   - Detect and flag heavily overlapping stems (confidence scores)
3. Run Phase 2 DSP on individual stems:
   - Extract BPM from drums (more accurate than full mix)
   - Extract key from isolated bass + melodic stems
   - Extract harmonic content separately from percussive

**DJ Benefit:** Automatically isolate buried vocal samples or synths in remixes without manual listening.

**Success Criteria:**
- Stem separation produces clean splits with <10% bleeding
- Stems can be exported for re-mixing/remixing workflows
- BPM extracted from drum stem is ±1% more accurate than full mix

### 3.2: Semantic Vibe & Mood Detection

**Problem Solved:** Traditional code detects brightness; it cannot classify psychological states like "hypnotic," "industrial," or "dark."

**Tasks:**
1. Integrate deep learning mood classifier:
   - Option A: Essentia TensorFlow models (pre-trained for style/mood/danceability)
   - Option B: OpenAI Audio Embeddings for abstract semantic similarity
2. Classify tracks across multiple dimensions:
   - Mood: dark, hypnotic, euphoric, aggressive, industrial, minimal
   - Energy: low, medium, high
   - Danceability: binary score + confidence
   - Spectral character: bright/dark, clean/dirty, organic/synthetic
3. Store confidence scores for each classification
4. Enable DJ filtering: "Show me all hypnotic tracks with low spectral contrast variance"

**DJ Benefit:** Automatically tag your library by psychological state. Filter for 3:00 AM minimal vibes or aggressive industrial builds.

**Success Criteria:**
- Mood classifications validated against manual tagging (>85% accuracy)
- Confidence scores reflect true prediction certainty
- Mood filters return sensible track selections

### 3.3: Intelligent Structural Segmentation (Auto-Cueing)

**Problem Solved:** Traditional algorithms confuse quiet bridges for breakdowns. Neural networks understand tension/release arcs.

**Tasks:**
1. Implement CNN or Transformer-based structural analysis:
   - Analyze drum stem’s energy profile over time
   - Map onset density and spectral changes globally
   - Identify narrative arc of the track
2. Auto-detect key structural moments:
   - **Intro/Outro:** Low energy, minimal percussion
   - **Main Drop:** Sudden energy spike, full drum hit
   - **Breakdown:** Energy valley, stripped-back instrumentation
   - **Teasing/Build:** Gradual energy rise with tension elements
   - **Bridge:** Harmonic or melodic shift without energy collapse
3. Write Hot Cues automatically:
   - Generate Traktor-compatible cue points at each structural landmark
   - Assign semantic names ("Drop", "Breakdown", "Outro")
4. Confidence scoring: Flag uncertain detections for manual review

**DJ Benefit:** Skip manual cueing. Traktor/Rekordbox hot cues auto-populated with 95%+ accuracy on structural points.

**Success Criteria:**
- Structural detection >90% accurate on test tracks
- Hot cues written to Traktor NML without manual intervention
- Edge cases (long intros, multiple drops) handled gracefully

**Dependencies:** Phase 2 (DSP features)

**Success Criteria (Phase 3 Overall):**
- Stem separation produces clean drum/bass/vocal splits with minimal bleeding
- Mood classifications match manual labeling (>85% agreement)
- Structural segmentation identifies drops/breakdowns with >90% accuracy
- Traktor export includes auto-generated hot cues

---

## Phase 4: Data Store & Export *(Weeks 6-7)*

**Deliverables:**
- `src/database/schema.py` — SQLite schema and ORM models
- `src/database/store.py` — write/query track features
- `src/traktor/exporter.py` — NML file writer
- `src/matching/similarity.py` — track similarity engine (Cosine Similarity)
- `tests/test_database.py`, `tests/test_traktor.py` — integration tests

**Tasks:**
1. Design SQLite schema:
   - `tracks` table: id, file_path, artist, title, album, analysis_date
   - `features` table: track_id, bpm, key, spectral_centroid, mfcc_vector, etc.
   - `mood` table: track_id, danceable, techno, aggressive, hypnotic (scores)
   - `stems` table: track_id, stem_name, stem_file_path
2. Implement database layer:
   - Create ORM models with SQLAlchemy or raw sqlite3
   - Write insert/query functions
   - Store MFCC vectors as serialized arrays (JSON or pickle)
3. Implement track similarity search:
   - See **Track Similarity Algorithm** subsection below
4. Implement Traktor NML exporter:
   - Parse user’s Traktor collection (`collection.nml`)
   - Write computed features as Cue points and metadata
   - Export back to `.nml` format
5. Create CLI or notebook to run full pipeline:
   - Ingest → DSP → AI → Store → Export
6. Write integration tests

**Dependencies:** Phase 3 (AI features)

### 4.1: Track Similarity Algorithm (The Matchmaker Tool)

**Problem:** How to find "similar-sounding" tracks when a DJ can’t manually curate every connection.

**Solution:** Convert tracks into mathematical vectors and measure geometric distance using Cosine Similarity.

#### Step 1: Vector Extraction (The DNA Profile)

Each track is converted into a feature vector capturing its timbral "DNA":

**Features to Extract & Normalize:**
- **MFCCs (Mel-Frequency Cepstral Coefficients):** Analyzes timbre (texture). Differentiates a metallic, clicky minimal track from a warm, organic house track. Store 13-20 MFCC coefficients per track.
- **Spectral Contrast:** Measures peaks vs. valleys in frequencies (reveals aggression of hats/percussion). One scalar value per track.
- **Tempo & Rhythmic Strength:** Ensures speed and "drive" match. Store BPM + beat confidence.
- **Spectral Centroid:** Brightness score (low = dark/subby, high = bright/percussive).
- **Chromatic Content:** Harmonic profile (key + chord family). Store as chroma vector.

**Vector Normalization:**
- Standardize all features to mean=0, std=1 (Z-score normalization) so one feature doesn’t dominate (BPM shouldn’t overshadow MFCC).
- Combine into single vector: `[mfcc_1...mfcc_13, spectral_contrast, bpm_normalized, spectral_centroid, chroma_1...chroma_12]`

#### Step 2: The Distance Formula (Cosine Similarity)

Once tracks are vectors, measure angular distance using **Cosine Similarity**:

```
similarity(Track A, Track B) = (A · B) / (||A|| × ||B||)

Range: [0.0, 1.0]
- 1.0 = identical direction (perfect match)
- 0.5 = moderately similar
- 0.0 = completely different
```

**Why Cosine Similarity?**
- Invariant to vector magnitude (a loud remix and quiet original can still match if timbre is similar)
- Fast to compute (perfect for real-time DJ database queries)
- Interpretable (ranges 0-1, easy to set thresholds)

#### Step 3: Ranking & Filtering

For a query track:
1. Compute similarity score against all tracks in database
2. Rank by descending similarity (1.0 → 0.0)
3. Apply optional filters (BPM range, key compatibility, mood match)
4. Return top-K results (typically top-5 for DJ workflow)

**Example Query:**
```python
matches = similarity_engine.find_similar(
    track_id=42,                    # Query: "Aphex Twin - Windowlicker"
    top_k=5,                        # Top 5 matches
    bpm_tolerance=2,                # Within ±2 BPM
    mood_filter="hypnotic"          # Same vibe
)
# Returns: [(Track_B, 0.87), (Track_C, 0.84), ...]
```

**DJ Benefit:** Instant "what should I play next?" suggestions. No manual crate digging.

**Success Criteria:**
- Similarity scores correlate with manual DJ assessment (inter-rater reliability >0.75)
- Top-5 results are playable together in a DJ set
- Query latency <100ms for 1000-track library

---

**Success Criteria (Phase 4 Overall):**
- SQLite stores 100+ tracks without performance issues
- Track similarity matches DJ intuition (validated by A/B testing)
- Traktor NML export imports cleanly into Traktor Pro with hot-cues and metadata

---

## Phase 5: Advanced AI Features — Generative Playlists & Transition Mapping *(Weeks 8-10, Optional)*

**Deliverables:**
- `src/ai/transition_mapper.py` — Graph Neural Networks for DJ transitions
- `src/ai/playlist_generator.py` — Generative playlist sequencing
- `src/ai/reinforcement_learner.py` — RL agent for optimal DJ flow
- `tests/test_advanced_ai.py` — integration tests

**Tasks:**
1. Build Transition Mapper (Graph Neural Network):
   - Model your track library as a directed graph
   - Nodes = tracks with feature vectors (BPM, key, mood, energy)
   - Edges = transition quality scores (energy smoothness, key harmonic distance, groove continuity)
   - Score transitions using: BPM compatibility, key harmonic distance, mood continuity, energy arc
2. Implement Reinforcement Learning DJ Agent:
   - State: current track + remaining library + transition path
   - Action: select next track from library
   - Reward: smooth energy progression, harmonic coherence, mood flow
   - Train on simulated DJ sets to learn optimal sequencing
3. Generative Playlist Builder:
   - Input: starting track + target ending track + number of steps
   - Output: optimal transition path through your library
   - Example: "Bridge me from 124 BPM minimal to 127 BPM Afro Acid in 5 steps"
4. Transition visualization:
   - Plot energy/key/mood arc over proposed DJ set
   - Show predicted "floor response" for each transition

**DJ Benefit:** AI co-pilot suggests optimal track sequences. Automatically builds smooth transitions between incompatible BPMs/keys.

**Success Criteria:**
- Generated playlists are musically coherent and playable
- Transition quality scores correlate with manual DJ feedback
- Can bridge two tracks with ≥5 BPM difference smoothly

**Dependencies:** Phase 4 (database of track features)

**Note:** This phase is advanced/optional. Core DJIA value is delivered in Phases 1-4. Phase 5 enables generative "co-pilot" features.

---

## Pre-Built AI Frameworks (No-Code / Low-Code Alternatives)

If you want advanced AI features without writing deep learning code:

- **Essentia TensorFlow Models:** Pre-trained classifiers for style/mood/danceability. Drop-in replacement for custom Phase 3.2 classifier.
- **OpenAI Audio Embeddings:** Feed audio → get semantic "meaning vectors" for superior similarity matching vs. hand-tuned features.
- **Spotify Audio Analysis API:** Use Spotify's pre-computed track features if your library is on Spotify (seamless integration with `spotipy` SDK).

---

## Implementation Order

1. **Phase 1** — Baseline: Read audio files, extract metadata
2. **Phase 2** — Core features: BPM, key, spectral analysis (testable standalone)
3. **Phase 3** — AI layer: Stem separation, mood classification, structural segmentation (depends on Phase 2)
4. **Phase 4** — Storage & export: Database, matching, Traktor export (depends on Phase 3)
5. **Phase 5** *(Optional)* — Advanced AI: Generative playlists, transition mapping (depends on Phase 4)

---

## File Structure (to Create)

```
src/
  __init__.py
  ingestion/
    __init__.py
    scanner.py        # Directory scanner
    loader.py         # Audio file decoder
  dsp/
    __init__.py
    extractor.py      # BPM, key, spectral features
  features/
    __init__.py
    schema.py         # Data classes for features
  ai/
    __init__.py
    stem_separator.py # Demucs/HTDemucs integration (Phase 3.1)
    classifier.py     # Mood/vibe classification (Phase 3.2)
    segmentation.py   # Structural segmentation & auto-cueing (Phase 3.3)
    processor.py      # Run DSP on stems
    transition_mapper.py      # GNN transition scoring (Phase 5, optional)
    playlist_generator.py     # Generative playlist sequencing (Phase 5, optional)
    reinforcement_learner.py  # RL DJ agent (Phase 5, optional)
  database/
    __init__.py
    schema.py         # SQLite schema
    store.py          # Query/insert functions
  traktor/
    __init__.py
    exporter.py       # NML writer
  matching/
    __init__.py
    similarity.py     # Track similarity engine
  cli.py              # Main entry point (or main.py)

tests/
  __init__.py
  test_ingestion.py
  test_dsp.py
  test_ai.py
  test_database.py
  test_traktor.py
  test_advanced_ai.py  # Phase 5 integration tests
```

---

## Dependencies to Add

**Core (Phases 1-4):**
```
librosa>=0.9.0          # Audio feature extraction
scipy                   # Signal processing
numpy                   # Numerical computing
pandas                  # Data manipulation
mutagen                 # Metadata reading
demucs                  # Advanced stem separation (AI demixing)
sqlalchemy              # ORM (optional; sqlite3 works standalone)
scikit-learn            # Similarity metrics & cosine similarity
pytest                  # Testing
```

**Phase 3.2 — Mood Classification (Choose One):**
```
essentia-tensorflow     # Pre-trained models for mood/style/danceability
# OR
openai                  # For Audio Embeddings API (requires API key)
```

**Phase 5 — Advanced AI (Optional):**
```
torch                   # PyTorch for Neural Networks
torch-geometric         # Graph Neural Networks
stable-baselines3       # RL agents for playlist generation
```

---

## Milestones

| Week | Phase | Deliverable |
|------|-------|-------------|
| 1 | Phase 1 | Audio ingestion & metadata parsing working |
| 2-3 | Phase 2 | BPM, key, spectral features extracted & tested |
| 4-5 | Phase 3.1 | Stem separation (Demucs) producing clean splits |
| 6 | Phase 3.2-3.3 | Mood detection + structural segmentation integrated |
| 7 | Phase 4 | SQLite database + Traktor NML export with auto hot-cues |
| 8-10 | Phase 5 *(Optional)* | GNN transition mapper + RL playlist generator |

**MVP (Minimum Viable Product):** End of Week 7 (Phases 1-4). Full DJIA pipeline working.
**Extended (with Generative AI):** End of Week 10 (Phases 1-5). AI co-pilot features enabled.

---

## Open Questions / Decisions

**Phase 3 (AI Layer):**
- [ ] Use SQLAlchemy ORM or raw sqlite3? (sqlite3 = simpler, SQLAlchemy = scalable)
- [ ] Which mood classifier? Essentia TensorFlow models, OpenAI Embeddings, or Spotify API?
- [ ] Cache processed stems to disk or re-process each run? (Disk = faster iteration, RAM = simplicity)
- [ ] Support Serato/Rekordbox export in addition to Traktor?

**Phase 5 (Advanced AI, Optional):**
- [ ] Use Graph Neural Networks (PyTorch Geometric) or simpler rule-based transition scoring?
- [ ] Train RL agent on synthetic DJ sets or crowdsourced real DJ mixes?
- [ ] Expose playlist generator as REST API or CLI-only?
- [ ] MVP without Phase 5, or prioritize playlist generator before Traktor export?