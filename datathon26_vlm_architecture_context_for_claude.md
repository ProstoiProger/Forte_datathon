# Datathon 26 — Local Video Compliance Architecture Context

## Project Goal

We are building a **fully local AI service for detecting cash-discipline / cashier-procedure violations from video recordings** for Datathon 26.

The user input is **video only**.

The system must analyze the video, find visually observable suspicious events, identify the relevant violation type, return the timestamp and evidence frames, and connect the finding to the relevant regulatory rule or source.

The final system must operate as an **open-world video compliance system**, not as a fixed closed-set classifier.

---

# Hard Constraints

1. **Input is video only.**
2. **Audio is forbidden** and must not be used.
3. **Cloud APIs are forbidden.**
4. All VLMs, embedding models, retrievers, databases, and reasoning components must run **locally**.
5. The provided training dataset contains roughly **40 videos**.
6. The provided labels include:
   - violation / no violation;
   - timestamp of the violation.
7. The organizer does **not** provide the violation type.
8. The test set may contain **violation types that never appear in the 40 training videos**.
9. Therefore, a traditional classifier such as:

   ```text
   video -> class_1 / class_2 / class_3
   ```

   is fundamentally insufficient.
10. The service must return:
    - overall verdict;
    - violation type;
    - timestamp;
    - 1–3 evidence frames;
    - description of what is visibly happening;
    - relevant regulatory source / rule.
11. The system must finish processing a test video within roughly **5 minutes**.
12. Deployment hardware:
    - RTX 5090 Laptop GPU;
    - 24 GB VRAM;
    - 64 GB system RAM;
    - about 200 GB free SSD space.

---

# Core Architectural Principle

The system should separate knowledge into three layers:

```text
1. FOUNDATION VISUAL KNOWLEDGE
   pretrained local VLM

2. EMPIRICAL CASE KNOWLEDGE
   40 provided training videos

3. REGULATORY KNOWLEDGE
   parsed regulations from Adilet / NBRK / other relevant Kazakhstan sources
```

These three layers solve different problems.

---

# 1. Foundation Visual Knowledge

We should use a strong pretrained **local Video/Vision-Language Model** as the primary perception model.

Its role is not to memorize all legal violations.

Its role is to answer:

> What objectively happened in the video?

For example:

```json
{
  "events": [
    {
      "start_sec": 21.1,
      "end_sec": 23.8,
      "actor": "cashier",
      "action": "opens_cash_drawer"
    },
    {
      "start_sec": 24.0,
      "end_sec": 27.1,
      "actor": "second_person",
      "action": "hands_object_to_cashier"
    }
  ]
}
```

The primary VLM should detect or describe:

- people;
- cashier/customer/incassator roles when inferable from context;
- objects;
- hand-object interactions;
- drawer opening/closing;
- object transfer;
- person entering/leaving the cashier zone;
- temporal ordering of actions;
- visually observable scene changes;
- candidate suspicious time intervals.

The VLM should **not be trusted to invent legal rules or citations**.

---

# 2. Empirical Knowledge from the 40 Videos

The 40 training videos should still be used heavily, but **not as a complete class taxonomy**.

The test set may contain unseen violation types, therefore the training set cannot define the full output space.

Instead, the 40 videos should become a **domain-specific multimodal case memory**.

For each training video, extract and store:

- original video ID;
- violation / no_violation ground truth;
- provided violation timestamp;
- clip around the timestamp;
- keyframes;
- multimodal / video embeddings;
- detected objects;
- observable visual events;
- temporal action sequence;
- local scene description;
- manually reviewed interpretation where possible;
- candidate regulatory rule IDs after legal analysis.

Example:

```json
{
  "video_id": "video_017",
  "ground_truth": {
    "violation": true,
    "timestamp_sec": 44.2
  },
  "clip": {
    "start_sec": 39.0,
    "end_sec": 49.0
  },
  "visual_events": [
    "second_person_approaches_cashier",
    "object_transferred_to_cashier",
    "cash_drawer_opened",
    "object_placed_inside_drawer"
  ],
  "candidate_rule_ids": [
    "RULE_027"
  ]
}
```

This becomes a **case-retrieval database** for test-time reasoning.

---

# Why the 40 Videos Should Not Define the Violation Classes

Suppose the training set contains only:

```text
A
B
C
```

A traditional classifier has a fixed output space:

```text
[A, B, C]
```

If the test set contains a new violation:

```text
D
```

the model cannot naturally output D because D is not part of its learned classification head.

Therefore, the system must not be designed as:

```text
video -> violation_class
```

Instead, it should be designed as:

```text
video
  ->
observable facts
  ->
candidate rules
  ->
rule verification
```

This is the core open-world design.

---

# Optional Training on the 40 Videos

The 40 videos may still be used for **lightweight domain adaptation**.

Recommended options:

- LoRA;
- linear probing;
- temporal localization head;
- anomaly / suspiciousness head;
- contrastive learning;
- normal-vs-suspicious clip scoring.

The goal is not:

> teach the model every possible violation.

The goal is:

> adapt the pretrained visual model to this specific CCTV / cashier environment.

For example:

```text
pretrained video/VLM encoder
        frozen
          |
          v
      embeddings
          |
   -------------------
   |                 |
anomaly head    temporal head
```

Possible training target:

```text
clip -> suspiciousness score
```

rather than:

```text
clip -> legal violation class
```

This lets the 40 videos help identify **where to look**, while the regulatory layer determines **what the event means legally**.

---

# 3. Regulatory Knowledge Base

The regulatory knowledge should be stored **outside the VLM weights**.

Reasons:

- regulations may change;
- legal citations must be precise;
- the model must not hallucinate law numbers;
- unseen violation types must remain detectable;
- legal rules should be independently auditable and updateable.

Pipeline:

```text
Adilet / NBRK / other relevant Kazakhstan regulations
        |
        v
document parsing
        |
        v
section extraction
        |
        v
atomic rule extraction
        |
        v
manual validation
        |
        v
Local Regulatory Knowledge Base
```

Each rule should ideally be normalized into something similar to:

```json
{
  "rule_id": "RULE_027",
  "document": "...",
  "section": "...",
  "text": "...",
  "conditions": [],
  "obligations": [],
  "exceptions": [],
  "observability": "VIDEO",
  "valid_from": "...",
  "valid_to": null
}
```

Possible observability categories:

```text
VIDEO
VIDEO_PLUS_OCR
EXTERNAL_DATA_REQUIRED
NOT_VISUALLY_OBSERVABLE
```

Only rules that can be reasonably verified from the available video should be used for direct video compliance decisions.

---

# Retrieval Does Not Require a Text Query from the User

The user uploads only a video.

The system itself generates the retrieval query.

Example:

```text
VIDEO
  ->
VLM observations:
  - cashier opens drawer
  - second person approaches
  - object is transferred
  - object placed inside drawer
```

These facts can be used to retrieve:

1. similar historical cases from the 40-video case memory;
2. relevant regulatory rules.

Therefore, this is not a traditional user-text RAG system.

It is better described as:

> **Multimodal Case Retrieval + Regulatory Retrieval + Video Rule Verification**

---

# Multimodal Retrieval

We plan to use **Qwen3-VL-Embedding** as the open-source multimodal embedding model.

The embedding layer should handle:

- text;
- images;
- video clips;
- multimodal content.

This allows us to store both:

```text
training video clips -> embeddings
```

and:

```text
regulatory text -> embeddings
```

and then retrieve useful context for a new video clip.

Important architectural point:

The embedding model and the primary VLM do **not** need to be from the same model family.

Using Qwen3-VL-Embedding does not technically require the primary VLM to also be Qwen.

There is no assumption that:

```text
Qwen VLM + Qwen embedding model
```

means they share the same latent space internally.

The embedding model produces vectors for retrieval.

The VLM receives the retrieved cases / rules afterward as context.

Therefore both MiniCPM and Qwen are compatible with Qwen3-VL-Embedding.

---

# Candidate Primary VLMs

We are currently comparing:

## MiniCPM-V 4.7 35B-A3B

Current local build:

```text
Quantization: IQ4_XS
Model weights: ~18.73 GB
Vision projector: ~1.12 GB
Base total: ~19.85 GB
```

Architecture:

```text
35B total parameters
MoE
~3B active parameters per token
```

Potential advantage:

- lower active compute during decoding;
- potentially better generation latency;
- strong local multimodal/video model;
- attractive MoE architecture.

Main disadvantage on our hardware:

```text
24 GB VRAM - 19.85 GB ~= 4.15 GB remaining
```

That remaining memory must also hold:

- KV cache;
- visual tokens;
- CUDA/runtime buffers;
- video-frame processing;
- driver / desktop overhead;
- temporary tensors.

For video workloads, ~4 GB of headroom is uncomfortable.

This creates a higher risk of:

- out-of-memory errors;
- CPU/RAM offloading;
- reduced context size;
- fewer frames;
- lower throughput;
- unstable inference.

---

## Qwen3.8-27B

Current local build:

```text
Quantization: Q4_K_M
Model weights: ~16.81 GB
Vision component/projector: ~0.93 GB
Base total: ~17.74 GB
```

Remaining VRAM:

```text
24 GB - 17.74 GB ~= 6.26 GB
```

That gives approximately 2.1 GB more VRAM headroom than the MiniCPM build.

For a normal LLM this difference may be modest.

For a video VLM on a 24 GB GPU, it is significant because additional visual context consumes substantial runtime memory.

---

# Primary VLM Recommendation

For the current hardware, choose:

> **Qwen3.8-27B Q4_K_M**

as the primary VLM.

The main reasons are:

1. Lower base VRAM usage.
2. More room for video tokens and KV cache.
3. Lower probability of OOM.
4. Less risk of RAM/CPU offloading.
5. More freedom to increase sampled frames or context.
6. Better operational margin under the 5-minute inference limit.
7. Easier integration with the planned local inference pipeline.

MiniCPM-V 4.7 remains a strong bake-off candidate, especially if real-world tests show a major latency or video-understanding advantage.

However, on exactly 24 GB VRAM, the extra ~2.1 GB consumed by the selected MiniCPM quantization makes the system much tighter.

---

# Latency Considerations

MiniCPM has an important theoretical advantage:

```text
35B total
~3B active
MoE
```

whereas Qwen3.8-27B is a dense model.

Therefore MiniCPM may have significantly lower **decode compute per generated token**.

However, our task does not generate thousands of tokens.

Typical output should be structured and relatively short:

```json
{
  "events": [],
  "violation": true,
  "rule_id": "...",
  "timestamp": 42.1
}
```

The expensive part of our task is likely to be:

```text
video decoding
+
visual encoding
+
multimodal prefill
+
processing many visual tokens
+
temporal reasoning
```

Therefore, MiniCPM's MoE decode advantage does not automatically imply proportionally faster end-to-end video processing.

Real project latency must be measured on actual Datathon videos.

---

# Accuracy Considerations

We should not choose solely from generic VLM benchmarks.

Our real task is unusual:

- top-down CCTV;
- small objects;
- hands;
- cashier/customer/incassator interactions;
- object passing;
- drawer state;
- person entering/leaving zones;
- subtle temporal ordering;
- no audio;
- evidence timestamping.

A model that performs better on generic visual QA may still perform worse on this specific CCTV domain.

Therefore, final selection should be based on a **project-specific bake-off**.

Recommended benchmark:

```text
10 known violation clips
10 normal clips
10 difficult/ambiguous clips
```

Measure:

```text
35% action/event correctness
20% temporal ordering
20% timestamp localization
15% hallucination rate
10% latency
```

The primary model must be tested using exactly the same sampled frames and output schema.

---

# Qwen3-VL-Embedding: 2B vs 8B

For our 24 GB VRAM machine, prefer:

> **Qwen3-VL-Embedding-2B**

for the runtime pipeline.

Reason:

The embedding model does not need to perform deep legal reasoning.

Its job is retrieval:

```text
clip
  ->
embedding
  ->
similar cases / relevant rules
```

The 8B version may produce better retrieval quality, but it uses substantially more memory.

The 2B model is much easier to fit into the workflow.

---

# Important: Do Not Keep Both Models Resident in VRAM

Even with Qwen3-VL-Embedding-2B, we should not assume that the embedder and the 27B VLM will permanently coexist in the 24 GB GPU.

Use sequential model residency.

Recommended runtime:

```text
INPUT VIDEO
    |
    v
temporal segmentation / clip generation
    |
    v
load Qwen3-VL-Embedding-2B
    |
    v
embed candidate clips
    |
    v
retrieve:
  - similar train cases
  - relevant regulatory rules
    |
    v
unload embedding model from GPU
    |
    v
load / run Qwen3.8-27B Q4_K_M
    |
    v
deep multimodal analysis only on selected clips
    |
    v
rule verification
    |
    v
final structured result
```

The regulatory database and case-memory embeddings are precomputed offline.

At runtime, only the new test video needs to be embedded.

---

# Recommended Inference Pipeline

```text
                        INPUT VIDEO
                             |
                             v
                  video decode / sampling
                             |
                             v
                 temporal segmentation
                             |
                             v
              candidate clips / windows
                             |
                             v
              Qwen3-VL-Embedding-2B
                             |
                  multimodal embeddings
                             |
                +------------+------------+
                |                         |
                v                         v
       40-video Case Memory       Regulatory KB
                |                         |
                +------------+------------+
                             |
                    top-k retrieved context
                             |
                             v
                   unload embedding model
                             |
                             v
                  Qwen3.8-27B Q4_K_M
                     PRIMARY LOCAL VLM
                             |
                             v
                   observable visual facts
                             |
                             v
                     rule verification
                             |
                             v
                 evidence-frame selection
                             |
                             v
                       FINAL RESULT
```

Final result:

```json
{
  "verdict": "violation",
  "violations": [
    {
      "type": "...",
      "timecode_sec": 44.2,
      "description": "...",
      "rule_id": "...",
      "source": "...",
      "evidence_frames": []
    }
  ]
}
```

---

# Known vs Unseen Violations

The architecture must support both.

## Seen-like violation

```text
test clip
  ->
embedding search
  ->
very similar training case found
  ->
relevant rule found
  ->
VLM verifies the current video independently
```

The 40-video case memory gives strong few-shot guidance.

---

## Unseen violation

Suppose no similar training case exists.

The pipeline can still work:

```text
test video
  ->
VLM extracts observable events
  ->
regulatory retrieval finds an applicable rule
  ->
VLM verifies whether the events violate the rule
```

Therefore, lack of a train example does not automatically prevent detection.

This is the main reason for keeping empirical case knowledge and regulatory knowledge separate.

---

# Avoid Pure Anomaly Detection as the Final Decision Layer

An anomaly model can be useful for candidate localization:

```text
video -> suspiciousness score
```

But it should not produce the final legal decision.

A visually unusual action is not necessarily a regulatory violation.

Likewise, a legal violation may look visually ordinary.

Therefore:

```text
anomaly detection = optional candidate-generation tool
```

not:

```text
anomaly detection = compliance verdict
```

---

# Avoid Letting the VLM Invent Regulatory Citations

The VLM should not freely generate:

```text
"Article 17 paragraph 4..."
```

from memory.

Instead:

```text
retriever
  ->
returns exact stored rules
  ->
VLM is only allowed to choose / verify among retrieved rules
```

This reduces hallucinated citations.

Legal source fields should ideally come directly from structured metadata in the Regulatory KB, not from free-form model generation.

---

# Recommended Role of the Primary VLM

Qwen3.8-27B should be treated as an **expensive deep analyzer**, not as a brute-force frame scanner.

Do not process every frame of a 145-second video at maximum resolution through the 27B model.

Better:

```text
video
  ->
cheap segmentation / sampling / embedding
  ->
candidate intervals
  ->
27B deep analysis only on relevant clips
```

This improves:

- latency;
- VRAM usage;
- temporal focus;
- reasoning quality;
- ability to meet the 5-minute deadline.

---

# Final Recommended Stack

## Primary VLM

```text
Qwen3.8-27B
Q4_K_M
~17.74 GB base including vision component
```

Role:

- deep video understanding;
- temporal reasoning;
- structured event extraction;
- rule verification;
- final explanation.

---

## Multimodal Embedder

```text
Qwen3-VL-Embedding-2B
```

Role:

- video/clip embedding;
- text/rule embedding;
- similar-case retrieval;
- regulatory retrieval.

---

## Knowledge Stores

### Case Memory

Contains:

```text
40 training videos
clips
timestamps
embeddings
keyframes
visual events
manually reviewed annotations
candidate rules
```

### Regulatory KB

Contains:

```text
atomic rules
exact source metadata
sections/paragraphs
conditions
exceptions
validity dates
observability metadata
embeddings
```

---

## Vector Store

Possible options:

```text
FAISS
Qdrant
```

For a Datathon-sized dataset, either is sufficient.

---

# Short Decision Summary

On RTX 5090 Laptop 24 GB VRAM:

```text
MiniCPM-V 4.7 35B-A3B IQ4_XS
base ~= 19.85 GB

Qwen3.8-27B Q4_K_M
base ~= 17.74 GB
```

MiniCPM has the theoretical MoE latency advantage.

Qwen leaves approximately 2.1 GB more VRAM headroom.

Because our workload is video-heavy and must fit within a 24 GB GPU, runtime memory headroom is extremely valuable.

Therefore the current recommended production candidate is:

> Qwen3.8-27B Q4_K_M

paired with:

> Qwen3-VL-Embedding-2B

using sequential loading and a multimodal case-memory + regulatory-retrieval architecture.

The 40 training videos are not discarded.

They become:
- domain adaptation data;
- temporal localization supervision;
- normal/suspicious examples;
- multimodal few-shot case memory.

The regulatory corpus defines the broader open-world violation space.

The pretrained local VLM provides general visual understanding.

The system combines all three.
