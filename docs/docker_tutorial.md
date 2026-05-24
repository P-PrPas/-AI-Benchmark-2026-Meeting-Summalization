# Docker Introduction & Model Submission Guide
> AI Thailand Benchmark Programs 2026 — Theerasit Issaranon

---

## 1. Docker Environment

| Component | Description |
|---|---|
| **Docker Desktop** | Management software (*licensed) |
| **Docker Engine** | Core system component |
| **Docker Image** | Self-contained environment |
| **Dockerfile** | Image-construction file (filename: `Dockerfile`) |
| **Docker Compose** | Complex composite Dockerfile (filename: `docker-compose.yml`) |
| **Docker Hub** | Public repository — https://hub.docker.com/ |

---

## 2. Docker Image

Pull a base image from Docker Hub:

```bash
docker image pull python:3.11-slim
```

Reference: https://hub.docker.com/layers/library/python/3.11-slim

---

## 3. Dockerfile

### Basic Syntax

```dockerfile
FROM python:3.11-slim

COPY requirements.txt ./
COPY model /model
RUN pip3 install --no-cache-dir -r requirements.txt

WORKDIR /model
CMD python3 run.py
```

### Command Reference

| Instruction | Purpose |
|---|---|
| `FROM <image>` | Create a new build stage from a base image |
| `COPY <src> <dst>` | Copy files or directories from local into image |
| `RUN <command>` | Execute a terminal/build command |
| `WORKDIR <path>` | Change the current working directory inside image |
| `CMD <command>` | Specify the default command to run when container starts |

Full reference: https://docs.docker.com/reference/dockerfile/

### What Gets Copied Into the Image

**Local file structure:**
```
model/
  run.py
Dockerfile
requirements.txt
```

**Resulting image file structure:**
```
model/
  run.py
requirements.txt
```

- `COPY requirements.txt ./` → copies `requirements.txt` to image root
- `COPY model /model` → copies local `model/` directory to `/model` in image
- `RUN pip3 install --no-cache-dir -r requirements.txt` → installs Python dependencies
- `WORKDIR /model` → sets working directory to `/model`
- `CMD python3 run.py` → runs `run.py` on container start

### Build & Run Commands

```bash
# Build image named "da" from current directory Dockerfile
docker build -t da .

# Run image "da" (remove container after exit)
docker run --rm da
```

---

## 4. Docker Compose

### Basic `docker-compose.yml` Syntax

```yaml
services:
  eval:
    image: "db"
    build: .
    volumes:
      - ./host_dir:/docker_dir
    working_dir: /model
    ports:
      - "8000:80"
```

### Field Reference

| Field | Purpose |
|---|---|
| `image: "db"` | Name to give the built Docker image |
| `build: .` | Path to Dockerfile (`.` = current directory; no filename needed for default `Dockerfile`) |
| `volumes:` | Mount host directories into container |
| `- ./host_dir:/docker_dir` | Format: `host_path:container_path` |
| `working_dir: /model` | Change working directory inside container |
| `ports:` | Expose/map ports |
| `- "8000:80"` | Format: `host_port:container_port` |

Full reference: https://docs.docker.com/reference/compose-file/

### Docker Compose Build & Run Commands

```bash
# Build image
docker compose build

# Run image
docker compose up

# Rebuild (no cache)
docker compose build --no-cache

# Build and run in one step
docker compose up --build
```

### GPU / Other Platform Support

Add to the service definition:

```yaml
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    platform: "linux/amd64"
```

---

## 5. Docker Compose — Volume Mapping Example

**Local file structure:**
```
model/
  run.py
file/
  config
Dockerfile
docker-compose.yml
requirements.txt
```

**After volume mount (`./file:/mount_volume`):**

Image sees:
```
model/
  run.py
mount_volume/       ← mapped from local ./file/
  config
requirements.txt
```

---

## 6. Submission Workflow

### 6.1 File Submission Workflow

```
USER    → runs model on released test data
USER    → uploads model output (as specified by task)
SYSTEM  → auto-places file at /result/{files}
SERVER  → evaluates and returns score
SYSTEM  → displays score and ranking
```

**Internal detail:**
```
USER    → produces model output
USER    → uploads file (as specified)
          ↓ mount
SYSTEM  → /result/{files}  (auto)
SERVER  → evaluates
```

### 6.2 Model Submission Workflow (Docker Image)

**Phase 1 — Push:**
```
USER    → docker image push → registry
SERVER  → ack
```

**Phase 2 — Inference:**
```
USER    → submits Docker image via web UI
SYSTEM  → enqueues and waits for inference server
SERVER  → runs Docker image
SYSTEM  → stores solution file
```

**Phase 3 — Evaluation:**
```
SYSTEM  → enqueues and waits for evaluation server
SERVER  → evaluates and returns score
SYSTEM  → displays score and ranking
```

---

## 7. Model Submission — Key Paths & Rules

> ⚠️ These paths are **fixed by the evaluation system**. Your code MUST use them exactly.

### I/O Paths

| Direction | Path | Note |
|---|---|---|
| **Input** (READ) | `/model/test/{files}` | Do **NOT** include test data in your image — this dir is overridden by the system at runtime |
| **Output** (WRITE) | `/result/{files}` | Write all results here |
| **Progress** (EXEC) | `/benchmark_lib/progress <n>` | Must call at the end with total count `n` |

### Progress Reporting

Call after completing each item (intermediate calls optional, final call required):

```python
# After item 1  (optional)
subprocess.run(["/benchmark_lib/progress", "1"])

# After item 2  (optional)
subprocess.run(["/benchmark_lib/progress", "2"])

# ...

# After item n  ← REQUIRED — signals completion to the system
subprocess.run(["/benchmark_lib/progress", str(n)])
```

The `/benchmark_lib/` directory is **mounted by the server** at runtime (read-only).

### Additional Constraints

- **No Internet connection** inside the container — image must be fully self-contained
- **Do not pre-populate `/model/test/`** — it will be overridden by the confidential test dataset
- **Restricted resources / limited runtime** — depends on the specific task

---

## 8. Model Submission — docker-compose.yml Volume Mapping

When testing locally, mirror the server's mount structure:

```yaml
services:
  run_service:
    image: "db"
    build: .
    volumes:
      - ./test_data:/model/test:ro         # test dataset (read-only)
      - ./result:/result:rw                # output directory (read-write)
      - ./benchmark_lib:/benchmark_lib/:ro # progress binary (read-only)
    working_dir: /model
```

### Resulting Image File Structure (at runtime)

```
model/                    ← from Dockerfile COPY
  run.py
test/                     ← mounted by server (confidential test data)
  {files}
result/                   ← mounted by server (write output here)
  {files}
benchmark_lib/            ← mounted by server (read-only)
  progress
requirements.txt
```

---

## 9. Pushing Image to Registry

### Step-by-step

```bash
# 1. Login (once)
docker login registry-dev.ai.in.th
#    Username: <your_urid>
#    Password: <your_password>

# 2. Pull or build your image
docker image pull python:3.11-slim

# 3. List images to get IMAGE_ID
docker images

# 4. Tag the image with registry path
docker image tag <IMAGE_ID> registry.ai.in.th/workshop_dc_docker_image/<group_id>/<urid>:<tag>

# Example:
docker image tag e67db9b14d09 \
  registry.ai.in.th/workshop_dc_docker_image/urteamid/theerasit.urid:the_best

# 5. Push to registry
docker image push registry.ai.in.th/workshop_dc_docker_image/<group_id>/<urid>:<tag>

# Example:
docker image push \
  registry.ai.in.th/workshop_dc_docker_image/urteamid/theerasit.urid:the_best
```

### Tag Format

```
registry.ai.in.th/workshop_dc_docker_image/<group_id>/<urid>:<tag>
```

- `<group_id>` = your team/group ID
- `<urid>` = your user ID (e.g. `theerasit.urid`)
- `<tag>` = any label (e.g. `latest`, `the_best`)

---

## 10. Summary Checklist for Model Submission

```
[ ] Dockerfile builds successfully
[ ] Image reads input from /model/test/{files}  (do NOT hardcode test data in image)
[ ] Image writes output to /result/{files}
[ ] Script calls /benchmark_lib/progress <n> at the very end
[ ] No internet calls in the script
[ ] All model weights / dependencies bundled inside the image
[ ] Image tagged with correct registry path
[ ] Image pushed to registry
[ ] Image selected and submitted via benchmark web UI
```
