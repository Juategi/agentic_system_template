# Guia de Configuracion y Despliegue

Guia paso a paso para poner en funcionamiento el sistema de agentes IA en tu proyecto de GitHub.

---

## Tabla de Contenidos

1. [Prerequisitos](#1-prerequisitos)
2. [Configuracion Inicial](#2-configuracion-inicial)
3. [Configurar GitHub](#3-configurar-github)
4. [Configurar el Proveedor LLM](#4-configurar-el-proveedor-llm)
5. [Preparar la Memoria del Proyecto](#5-preparar-la-memoria-del-proyecto)
6. [Opciones de Ejecucion](#6-opciones-de-ejecucion)
   - [Opcion A: Local con Docker Compose](#opcion-a-local-con-docker-compose)
   - [Opcion B: Local con Ollama (sin API de pago)](#opcion-b-local-con-ollama-sin-api-de-pago)
   - [Opcion C: Nube con Kubernetes](#opcion-c-nube-con-kubernetes)
   - [Opcion D: VPS / Servidor Dedicado](#opcion-d-vps--servidor-dedicado)
7. [Crear tu Primera Issue](#7-crear-tu-primera-issue)
8. [Monitorizacion](#8-monitorizacion)
9. [Comandos Utiles](#9-comandos-utiles)
10. [Solucion de Problemas](#10-solucion-de-problemas)

---

## 1. Prerequisitos

### Software necesario (todas las opciones)

| Software         | Version minima | Uso                          |
|------------------|----------------|------------------------------|
| Docker           | 20.10+         | Contenedores de agentes      |
| Docker Compose   | 2.0+           | Orquestacion local           |
| Git              | 2.30+          | Control de versiones         |
| Python           | 3.11+          | Solo si ejecutas sin Docker  |

### Cuentas necesarias

- **GitHub**: Cuenta con acceso al repositorio objetivo
- **Proveedor LLM** (una de estas):
  - Anthropic (Claude) — recomendado
  - OpenAI (GPT-4)
  - Ollama (local, gratuito, sin API key)

---

## 2. Configuracion Inicial

### Paso 1: Clonar el template

```bash
git clone <url-de-este-repositorio> mi-proyecto-agentes
cd mi-proyecto-agentes
```

### Paso 2: Ejecutar el script de inicializacion (interactivo)

```bash
./scripts/init_project.sh
```

Este script te pedira:
- Nombre del proyecto
- Repositorio de GitHub (`owner/repo`)
- Token de GitHub
- Proveedor LLM y API key

Al finalizar, creara el archivo `.env`, las carpetas necesarias y opcionalmente construira las imagenes Docker.

### Paso 2 (alternativo): Configuracion manual

Si prefieres configurar manualmente:

```bash
# Crear .env desde el template
cp .env.template .env

# Crear directorios necesarios
mkdir -p memory/features logs output repo
```

Edita `.env` con tus valores. Las variables **obligatorias** son:

```env
# Identificacion del proyecto
PROJECT_ID=mi-proyecto
PROJECT_NAME="Mi Proyecto"

# GitHub
GITHUB_TOKEN=ghp_tu_token_aqui
GITHUB_REPO=tu-usuario/tu-repo

# LLM (elige uno)
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-tu_key_aqui
# o bien:
# LLM_PROVIDER=openai
# OPENAI_API_KEY=sk-tu_key_aqui
```

---

## 3. Configurar GitHub

### 3.1 Crear un Personal Access Token (PAT)

1. Ve a **GitHub > Settings > Developer settings > Personal access tokens > Tokens (classic)**
2. Genera un nuevo token con estos permisos:
   - `repo` (acceso completo al repositorio)
   - `write:discussion` (comentarios en issues)
3. Copia el token y ponlo en `GITHUB_TOKEN` en `.env`

> **Alternativa: GitHub App** — Para entornos de produccion, puedes usar una GitHub App en lugar de un PAT. Configura estas variables en `.env`:
> ```env
> GITHUB_APP_ID=12345
> GITHUB_APP_PRIVATE_KEY_PATH=/ruta/a/private-key.pem
> GITHUB_APP_INSTALLATION_ID=67890
> ```

### 3.2 Crear las labels automaticamente

El sistema usa labels en las issues para rastrear el estado del flujo de trabajo. Ejecuta:

```bash
./scripts/setup_github.sh
```

Esto creara las siguientes labels en tu repositorio:

**Labels de estado del flujo:**

| Label           | Color    | Significado                        |
|-----------------|----------|------------------------------------|
| `READY`         | Verde    | Tarea lista para asignar a agente  |
| `IN_PROGRESS`   | Amarillo | Agente trabajando                  |
| `QA`            | Azul     | Validando calidad                  |
| `QA_FAILED`     | Rosa     | QA fallo, vuelve a desarrollo      |
| `REVIEW`        | Morado   | En revision de codigo              |
| `BLOCKED`       | Rojo     | Requiere intervencion humana       |
| `DONE`          | Verde    | Completado exitosamente            |

**Labels de tipo:**

| Label      | Significado                            |
|------------|----------------------------------------|
| `feature`  | Feature compleja (sera descompuesta)   |
| `task`     | Tarea individual de desarrollo         |
| `bug`      | Bug a corregir                         |
| `subtask`  | Subtarea de una feature                |

**Labels de prioridad:** `priority:critical`, `priority:high`, `priority:medium`, `priority:low`

### 3.3 (Opcional) Configurar Webhooks

Por defecto el sistema usa **polling** (revisa GitHub cada 30 segundos). Si quieres respuesta inmediata, configura webhooks:

1. En tu repositorio: **Settings > Webhooks > Add webhook**
2. **Payload URL**: La URL donde corre tu orquestador (ej: `https://tu-dominio.com/webhooks/github`)
3. **Content type**: `application/json`
4. **Secret**: Pon el mismo valor que `GITHUB_WEBHOOK_SECRET` en `.env`
5. **Events**: Selecciona `Issues`, `Issue comments`, `Labels`

En `.env`, activa webhooks:

```env
ENABLE_WEBHOOKS=true
GITHUB_WEBHOOK_SECRET=tu_secreto_aqui
```

> **Para desarrollo local** puedes usar ngrok como tunel. Ejecuta `make start-dev` para levantar el orquestador con un tunel ngrok automatico (necesitas configurar `NGROK_AUTHTOKEN` en `.env`).

---

## 4. Configurar el Proveedor LLM

### Opcion 1: Anthropic (Claude) — Recomendado

```env
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-tu_key_aqui

# Modelos por agente (puedes personalizar)
LLM_MODEL_PLANNER=claude-sonnet-4-20250514
LLM_MODEL_DEVELOPER=claude-sonnet-4-20250514
LLM_MODEL_QA=claude-sonnet-4-20250514
LLM_MODEL_REVIEWER=claude-sonnet-4-20250514
LLM_MODEL_DOC=claude-haiku-4-20250514
```

### Opcion 2: OpenAI (GPT-4)

```env
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-tu_key_aqui

LLM_MODEL_PLANNER=gpt-4o
LLM_MODEL_DEVELOPER=gpt-4o
LLM_MODEL_QA=gpt-4o
LLM_MODEL_REVIEWER=gpt-4o
LLM_MODEL_DOC=gpt-4o-mini
```

### Opcion 3: Ollama (Local, gratuito)

Ollama te permite ejecutar modelos LLM en tu propia maquina, sin API keys ni costes.

1. **Instalar Ollama**: https://ollama.ai
2. **Descargar un modelo**:
   ```bash
   ollama pull llama3.2
   # Para vision (analisis de imagenes):
   ollama pull llava
   ```
3. **Configurar en `.env`**:
   ```env
   LLM_PROVIDER=ollama
   OLLAMA_BASE_URL=http://localhost:11434
   OLLAMA_MODEL=llama3.2
   ```

> **Nota**: Cuando Docker Compose ejecuta los contenedores, usa `http://host.docker.internal:11434` para conectar con Ollama en la maquina host. Esto ya esta configurado automaticamente en `docker-compose.yml`.

### Parametros LLM comunes

```env
LLM_TEMPERATURE=0.2    # Baja temperatura = respuestas mas deterministas
LLM_MAX_TOKENS=8192    # Maximo de tokens por respuesta
```

---

## 5. Preparar la Memoria del Proyecto

El sistema usa archivos Markdown como "memoria" del proyecto. Los agentes leen estos archivos para entender el contexto. Edita estos archivos en la carpeta `memory/`:

### `memory/PROJECT.md` (obligatorio)

Describe tu proyecto: que hace, cual es su proposito, tecnologias usadas, estructura general.

### `memory/ARCHITECTURE.md`

Documenta la arquitectura tecnica: componentes principales, como se comunican, diagramas si los tienes.

### `memory/CONVENTIONS.md`

Convenciones de codigo: estilo, patrones, nombrado de archivos, estructura de carpetas.

### `memory/CONSTRAINTS.md`

Restricciones tecnicas: versiones de lenguaje, dependencias obligatorias, limitaciones de infraestructura.

> **Importante**: Cuanto mejor documentada este tu memoria, mejores resultados daran los agentes. Dedicale tiempo a escribir estos archivos con detalle.

### Repositorio del proyecto

Clona o vincula el repositorio de tu proyecto en la carpeta `repo/`:

```bash
git clone https://github.com/tu-usuario/tu-repo.git repo/
```

Este directorio se monta como volumen en los contenedores de los agentes.

---

## 6. Opciones de Ejecucion

### Opcion A: Local con Docker Compose

Esta es la opcion mas sencilla para empezar. Todo corre en tu maquina local.

#### Requisitos

- Docker Desktop (Windows/Mac) o Docker Engine (Linux)
- Al menos 8 GB de RAM libre
- Acceso a internet (para GitHub API y LLM API)

#### Pasos

```bash
# 1. Construir las imagenes Docker
make build
# o equivalente:
docker-compose build

# 2. Arrancar el orquestador + Redis
make start
# o equivalente:
docker-compose up -d orchestrator

# 3. Verificar que esta corriendo
make status
docker-compose ps

# 4. Ver los logs en tiempo real
make logs
docker-compose logs -f orchestrator
```

#### Que se levanta

| Servicio       | Descripcion                              | Puerto |
|----------------|------------------------------------------|--------|
| `orchestrator` | Proceso principal, corre 24/7            | 8080   |
| `redis`        | Backend de estado y colas (auto)         | 6379   |
| `agent-runner` | Template de agentes (no se arranca solo) | -      |

El orquestador lanzara contenedores de agentes automaticamente segun sea necesario.

#### Perfiles opcionales

```bash
# Con tunel ngrok para webhooks en desarrollo
docker-compose --profile dev up -d

# Con todos los servicios (incluye PostgreSQL)
docker-compose --profile full up -d

# Con stack de monitorizacion (Prometheus + Grafana)
docker-compose --profile monitoring up -d
```

#### Parar el sistema

```bash
make stop
# o:
docker-compose down

# Para eliminar tambien los volumenes (resetea estado):
docker-compose down -v
```

---

### Opcion B: Local con Ollama (sin API de pago)

Ideal para probar el sistema sin gastar en APIs.

#### Requisitos adicionales

- Ollama instalado: https://ollama.ai
- GPU recomendada (NVIDIA con al menos 8 GB VRAM para modelos de 7B)
- Sin GPU: funciona con CPU pero mucho mas lento

#### Pasos

```bash
# 1. Iniciar Ollama y descargar modelo
ollama serve  # Si no esta corriendo como servicio
ollama pull llama3.2

# 2. Configurar .env
# LLM_PROVIDER=ollama
# OLLAMA_BASE_URL=http://localhost:11434
# OLLAMA_MODEL=llama3.2

# 3. Construir y arrancar (igual que Opcion A)
make build
make start
```

> **Modelos recomendados para Ollama:**
> - `llama3.2` — Bueno para tareas generales (8B params)
> - `codellama` — Optimizado para codigo
> - `deepseek-coder` — Alternativa para codigo
> - `llava` — Soporta analisis de imagenes (multimodal)
> - `mistral` — Buen balance calidad/velocidad

---

### Opcion C: Nube con Kubernetes

Para ejecucion en produccion con escalado automatico.

#### Requisitos

- Cluster de Kubernetes (EKS, GKE, AKS, o self-managed)
- `kubectl` configurado
- Un Container Registry (ECR, GCR, Docker Hub)
- Secretos de Kubernetes para tokens y API keys

#### Paso 1: Construir y publicar imagenes

```bash
# Definir tu registry
export REGISTRY=tu-registry.com/ai-agents

# Construir imagenes
docker build -f docker/Dockerfile.orchestrator -t $REGISTRY/orchestrator:latest .
docker build -f docker/Dockerfile.agent -t $REGISTRY/ai-agent:latest .

# Publicar
docker push $REGISTRY/orchestrator:latest
docker push $REGISTRY/ai-agent:latest
```

#### Paso 2: Crear secretos de Kubernetes

```bash
kubectl create namespace ai-agents

kubectl create secret generic ai-agent-secrets \
  --namespace ai-agents \
  --from-literal=GITHUB_TOKEN=ghp_tu_token \
  --from-literal=ANTHROPIC_API_KEY=sk-ant-tu_key \
  --from-literal=GITHUB_WEBHOOK_SECRET=tu_secreto
```

#### Paso 3: Desplegar

Crea un archivo `k8s/deployment.yaml`:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: orchestrator
  namespace: ai-agents
spec:
  replicas: 1
  selector:
    matchLabels:
      app: orchestrator
  template:
    metadata:
      labels:
        app: orchestrator
    spec:
      serviceAccountName: orchestrator
      containers:
        - name: orchestrator
          image: tu-registry.com/ai-agents/orchestrator:latest
          ports:
            - containerPort: 8080
          env:
            - name: PROJECT_ID
              value: "mi-proyecto"
            - name: GITHUB_REPO
              value: "tu-usuario/tu-repo"
            - name: LLM_PROVIDER
              value: "anthropic"
            - name: ORCHESTRATOR_STATE_BACKEND
              value: "redis"
            - name: REDIS_URL
              value: "redis://redis-svc:6379/0"
            - name: DOCKER_AGENT_IMAGE
              value: "tu-registry.com/ai-agents/ai-agent:latest"
          envFrom:
            - secretRef:
                name: ai-agent-secrets
          volumeMounts:
            - name: memory
              mountPath: /memory
            - name: repo
              mountPath: /repo
          resources:
            requests:
              memory: "512Mi"
              cpu: "250m"
            limits:
              memory: "2Gi"
              cpu: "1"
      volumes:
        - name: memory
          persistentVolumeClaim:
            claimName: memory-pvc
        - name: repo
          persistentVolumeClaim:
            claimName: repo-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: orchestrator-svc
  namespace: ai-agents
spec:
  selector:
    app: orchestrator
  ports:
    - port: 8080
      targetPort: 8080
  type: ClusterIP
---
# Redis para estado
apiVersion: apps/v1
kind: Deployment
metadata:
  name: redis
  namespace: ai-agents
spec:
  replicas: 1
  selector:
    matchLabels:
      app: redis
  template:
    metadata:
      labels:
        app: redis
    spec:
      containers:
        - name: redis
          image: redis:7-alpine
          command: ["redis-server", "--appendonly", "yes"]
          ports:
            - containerPort: 6379
          volumeMounts:
            - name: redis-data
              mountPath: /data
      volumes:
        - name: redis-data
          persistentVolumeClaim:
            claimName: redis-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: redis-svc
  namespace: ai-agents
spec:
  selector:
    app: redis
  ports:
    - port: 6379
```

> **Nota sobre ejecucion de agentes en Kubernetes**: El orquestador necesita lanzar contenedores de agentes. En Kubernetes, en lugar de montar el Docker socket, debes configurar el orquestador para crear **Kubernetes Jobs** para cada ejecucion de agente. Esto requiere que el ServiceAccount del orquestador tenga permisos para crear/listar/borrar Jobs en el namespace.

```bash
kubectl apply -f k8s/deployment.yaml
```

#### Paso 4: Exponer webhooks (opcional)

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: orchestrator-ingress
  namespace: ai-agents
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
spec:
  rules:
    - host: agentes.tu-dominio.com
      http:
        paths:
          - path: /webhooks
            pathType: Prefix
            backend:
              service:
                name: orchestrator-svc
                port:
                  number: 8080
```

---

### Opcion D: VPS / Servidor Dedicado

Para ejecutar en un servidor Linux remoto (DigitalOcean, Hetzner, AWS EC2, etc.)

#### Requisitos

- Servidor Linux (Ubuntu 22.04+ recomendado)
- Al menos 4 GB RAM, 2 vCPUs
- Docker y Docker Compose instalados
- Acceso SSH

#### Pasos

```bash
# 1. Conectar al servidor
ssh usuario@tu-servidor

# 2. Instalar Docker (si no esta instalado)
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER
# Reconectar SSH para que aplique el grupo

# 3. Instalar Docker Compose
sudo apt install docker-compose-plugin

# 4. Clonar el repositorio del template
git clone <url-del-template> /opt/ai-agents
cd /opt/ai-agents

# 5. Configurar (igual que Opcion A)
cp .env.template .env
nano .env  # Editar con tus valores

# 6. Construir y arrancar
docker compose build
docker compose up -d

# 7. Configurar reinicio automatico (ya esta en docker-compose.yml con restart: unless-stopped)
```

#### Configurar como servicio systemd (opcional)

Crea `/etc/systemd/system/ai-agents.service`:

```ini
[Unit]
Description=AI Agent Development System
Requires=docker.service
After=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=/opt/ai-agents
ExecStart=/usr/bin/docker compose up -d
ExecStop=/usr/bin/docker compose down
TimeoutStartSec=0

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable ai-agents
sudo systemctl start ai-agents
```

#### Configurar webhooks con Caddy (reverse proxy + HTTPS automatico)

```bash
# Instalar Caddy
sudo apt install -y debian-keyring debian-archive-keyring apt-transport-https
curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' | sudo gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' | sudo tee /etc/apt/sources.list.d/caddy-stable.list
sudo apt update && sudo apt install caddy
```

Edita `/etc/caddy/Caddyfile`:

```
agentes.tu-dominio.com {
    reverse_proxy localhost:8080
}
```

```bash
sudo systemctl restart caddy
```

Ahora puedes configurar el webhook de GitHub apuntando a `https://agentes.tu-dominio.com/webhooks/github`.

---

## 7. Crear tu Primera Issue

Una vez el sistema esta corriendo, crea una issue en tu repositorio de GitHub para que los agentes empiecen a trabajar.

### Feature (se descompone en subtareas)

Crea una issue con el label `feature` + `READY`:

```markdown
Titulo: [Feature] Sistema de autenticacion de usuarios

## Description
Implementar un sistema completo de autenticacion con registro,
login y recuperacion de contrasena.

## Acceptance Criteria
- [ ] Los usuarios pueden registrarse con email y contrasena
- [ ] Los usuarios pueden hacer login
- [ ] Las contrasenas se almacenan hasheadas
- [ ] Existe endpoint de recuperacion de contrasena
- [ ] Tests unitarios cubren los casos principales

## Technical Considerations
- Usar bcrypt para hashing de contrasenas
- JWT para sesiones
- Compatible con la arquitectura existente
```

El **Planner Agent** descompondra esta feature en subtareas mas pequenas.

### Task (se implementa directamente)

Crea una issue con el label `task` + `READY`:

```markdown
Titulo: [Task] Agregar endpoint GET /health

## Description
Crear un endpoint HTTP GET /health que devuelva el estado del servicio.

## Acceptance Criteria
- [ ] GET /health devuelve 200 con {"status": "ok"}
- [ ] Incluye version del servicio en la respuesta
- [ ] Test unitario para el endpoint
```

El **Developer Agent** implementara esta tarea directamente.

### Flujo automatico

1. El orquestador detecta la issue con label `READY`
2. Segun el tipo (`feature` o `task`), lanza el agente apropiado
3. El agente trabaja, actualiza la issue con comentarios
4. Pasa por QA, review y documentacion automaticamente
5. Si QA falla, vuelve a desarrollo (maximo 5 iteraciones)
6. Si no puede resolverse, se marca como `BLOCKED` para intervencion humana

---

## 8. Monitorizacion

### Logs del orquestador

```bash
make logs
# o
docker-compose logs -f orchestrator
```

### Stack de Prometheus + Grafana

```bash
# Levantar con perfil de monitorizacion
docker-compose --profile monitoring up -d
```

- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000 (usuario: `admin`, contrasena: valor de `GRAFANA_PASSWORD` en `.env`, por defecto `admin`)

### Health check

```bash
curl http://localhost:8080/health
```

### Audit trail

Los logs de auditoria se escriben en `logs/` en formato JSONL. Cada accion de los agentes queda registrada.

---

## 9. Comandos Utiles

| Comando                   | Descripcion                                |
|---------------------------|--------------------------------------------|
| `make build`              | Construir imagenes Docker                  |
| `make start`              | Arrancar orquestador                       |
| `make stop`               | Parar todo                                 |
| `make restart`            | Reiniciar servicios                        |
| `make logs`               | Ver logs del orquestador                   |
| `make status`             | Estado de los servicios                    |
| `make health`             | Health check del sistema                   |
| `make start-monitoring`   | Arrancar con Prometheus + Grafana          |
| `make start-dev`          | Arrancar con tunel ngrok                   |
| `make clean`              | Limpiar contenedores y cache               |
| `make run-planner ISSUE_NUMBER=5`   | Ejecutar planner manualmente    |
| `make run-developer ISSUE_NUMBER=5` | Ejecutar developer manualmente  |
| `make run-qa ISSUE_NUMBER=5`        | Ejecutar QA manualmente         |

---

## 10. Solucion de Problemas

### El orquestador no arranca

```bash
# Verificar logs
docker-compose logs orchestrator

# Verificar que .env tiene las variables necesarias
grep -c "GITHUB_TOKEN" .env
grep -c "GITHUB_REPO" .env
```

### Los agentes no se lanzan

El orquestador necesita acceso al Docker socket para lanzar contenedores de agentes:

```bash
# Verificar que el socket esta montado
docker-compose exec orchestrator ls -la /var/run/docker.sock

# Verificar que la imagen del agente existe
docker images | grep ai-agent
```

### Error de autenticacion con GitHub

```bash
# Probar el token manualmente
curl -H "Authorization: Bearer tu_token" https://api.github.com/repos/tu-usuario/tu-repo
```

Asegurate de que el token tiene permiso `repo`.

### Ollama no conecta desde Docker

Cuando ejecutas en Docker, los contenedores no pueden acceder a `localhost` del host directamente. La configuracion de `docker-compose.yml` ya mapea `host.docker.internal` automaticamente:

```env
OLLAMA_BASE_URL=http://host.docker.internal:11434
```

En **Linux**, esto puede requerir el flag `--add-host` o la directiva `extra_hosts` (ya configurada en `docker-compose.yml`).

### Una issue se queda en BLOCKED

Esto significa que los agentes no pudieron completar la tarea tras el maximo de iteraciones, o hubo un error irrecuperable.

1. Lee el comentario del agente en la issue para entender el problema
2. Corrige las instrucciones o agrega contexto
3. Quita el label `BLOCKED` y pon `READY` para reintentar

### Limites de recursos

Cada agente tiene limites por defecto:

```env
DOCKER_AGENT_CPU_LIMIT=2        # CPUs
DOCKER_AGENT_MEMORY_LIMIT=4g    # RAM
AGENT_TIMEOUT=1800               # 30 minutos maximo
AGENT_MAX_ITERATIONS=5           # Intentos QA antes de BLOCKED
```

Ajusta estos valores en `.env` segun tu hardware.

---

## Resumen de Variables de Entorno

### Obligatorias

| Variable            | Ejemplo                   | Descripcion                    |
|---------------------|---------------------------|--------------------------------|
| `PROJECT_ID`        | `mi-proyecto`             | Identificador del proyecto     |
| `GITHUB_TOKEN`      | `ghp_xxxx`                | Token de acceso a GitHub       |
| `GITHUB_REPO`       | `usuario/repo`            | Repositorio objetivo           |
| `LLM_PROVIDER`      | `anthropic`               | Proveedor LLM                  |
| `ANTHROPIC_API_KEY`  | `sk-ant-xxxx`            | Key de Anthropic (si aplica)   |
| `OPENAI_API_KEY`     | `sk-xxxx`                | Key de OpenAI (si aplica)      |

### Opcionales (con valores por defecto)

| Variable                           | Default       | Descripcion                       |
|------------------------------------|---------------|-----------------------------------|
| `ENVIRONMENT`                      | `development` | Entorno de ejecucion              |
| `ORCHESTRATOR_POLL_INTERVAL`       | `30`          | Segundos entre polls a GitHub     |
| `ORCHESTRATOR_MAX_CONCURRENT_AGENTS` | `3`         | Agentes en paralelo               |
| `ORCHESTRATOR_STATE_BACKEND`       | `file`        | Backend de estado (file/redis)    |
| `AGENT_MAX_ITERATIONS`             | `5`           | Max reintentos antes de BLOCKED   |
| `AGENT_TIMEOUT`                    | `1800`        | Timeout de agente en segundos     |
| `LLM_TEMPERATURE`                  | `0.2`         | Temperatura del LLM               |
| `LLM_MAX_TOKENS`                   | `8192`        | Max tokens por respuesta          |
| `LOG_LEVEL`                        | `INFO`        | Nivel de log                      |
| `ENABLE_WEBHOOKS`                  | `false`       | Activar modo webhook              |
| `ENABLE_METRICS`                   | `true`        | Activar metricas Prometheus       |
| `OLLAMA_BASE_URL`                  | `http://localhost:11434` | URL de Ollama           |
| `OLLAMA_MODEL`                     | `llama3.2`    | Modelo de Ollama                  |
| `DRY_RUN`                          | `false`       | Modo simulacion (sin cambios)     |

---

## Comparativa de Opciones de Ejecucion

| Caracteristica         | Local Docker Compose | Local + Ollama | Kubernetes           | VPS                  |
|------------------------|----------------------|----------------|----------------------|----------------------|
| Dificultad de setup    | Baja                 | Baja           | Alta                 | Media                |
| Coste infraestructura  | Ninguno              | Ninguno        | Variable (cloud)     | Desde ~5 USD/mes     |
| Coste LLM              | Por uso API          | Gratuito       | Por uso API          | Por uso API / Gratis |
| Disponibilidad 24/7    | Solo si PC encendido | Solo si PC on  | Si                   | Si                   |
| Escalabilidad          | Limitada             | Limitada       | Alta                 | Media                |
| Calidad de resultados  | Alta (Claude/GPT)    | Media (local)  | Alta                 | Alta / Media         |
| Uso recomendado        | Desarrollo/pruebas   | Experimentar   | Produccion a escala  | Produccion basica    |