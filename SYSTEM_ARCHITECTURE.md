# Arquitectura Completa del Sistema de Agentes IA

Documento educativo que explica en profundidad cada componente, clase y funcion del sistema.

---

## Tabla de Contenidos

1. [Vision General](#1-vision-general)
2. [Estructura de Archivos](#2-estructura-de-archivos)
3. [El Orquestador](#3-el-orquestador)
   - 3.1 [Punto de Entrada (main.py)](#31-punto-de-entrada-mainpy)
   - 3.2 [Motor del Engine](#32-motor-del-engine)
   - 3.3 [Integracion con GitHub](#33-integracion-con-github)
   - 3.4 [Nodos del Workflow](#34-nodos-del-workflow)
   - 3.5 [Scheduler (Planificador)](#35-scheduler-planificador)
4. [Los Agentes](#4-los-agentes)
   - 4.1 [Base Compartida](#41-base-compartida)
   - 4.2 [Planner Agent](#42-planner-agent)
   - 4.3 [Developer Agent](#43-developer-agent)
   - 4.4 [QA Agent](#44-qa-agent)
   - 4.5 [Reviewer Agent](#45-reviewer-agent)
   - 4.6 [Doc Agent](#46-doc-agent)
5. [Monitorizacion](#5-monitorizacion)
6. [Infraestructura Docker](#6-infraestructura-docker)
7. [Flujo de Datos Completo](#7-flujo-de-datos-completo)

---

## 1. Vision General

El sistema es una plataforma de desarrollo de software autonomo basada en agentes de IA. Su proposito es tomar features definidas como GitHub Issues y producir codigo funcional, testeado y documentado sin intervencion humana (excepto cuando se bloquea).

### Principios de Diseno

- **Efimero**: Los agentes se ejecutan en contenedores Docker que se crean y destruyen por tarea. No mantienen estado entre ejecuciones.
- **Imagen unica**: Todos los agentes comparten la misma imagen Docker. El comportamiento se diferencia por la variable de entorno `AGENT_TYPE`.
- **Estado en GitHub**: Las GitHub Issues son la fuente de verdad. Las labels representan el estado del flujo de trabajo.
- **Memoria en Markdown**: El contexto del proyecto se almacena en archivos Markdown que los agentes leen para entender el proyecto.
- **Recuperable**: El orquestador persiste su estado y puede recuperarse tras un reinicio.

### Componentes Principales

```
                    +-------------------+
                    |   GitHub Issues   |
                    |   (Backlog)       |
                    +--------+----------+
                             |
                    +--------v----------+
                    |   ORQUESTADOR     |
                    |  (corre 24/7)     |
                    |                   |
                    |  - LangGraph      |
                    |  - LangChain      |
                    |  - State Manager  |
                    |  - Queue Manager  |
                    +--------+----------+
                             |
              +--------------+--------------+
              |              |              |
        +-----v----+  +-----v----+  +------v---+
        | Planner  |  |Developer |  |    QA    |
        | Agent    |  |  Agent   |  |  Agent   |
        +----------+  +----------+  +----------+
              |              |              |
        +-----v----+  +-----v----+
        | Reviewer |  |   Doc    |
        |  Agent   |  |  Agent   |
        +----------+  +----------+
              |
        +-----v----+
        |  Memory  |
        | (Markdown)|
        +----------+
```

### Flujo de Estados

```
[New Issue] --> TRIAGE --> PLANNING --> AWAIT_SUBTASKS --> DOCUMENTATION --> DONE
                  |                                            ^
                  v                                            |
               DEVELOPMENT --> QA --> QA_PASSED --> REVIEW ----+
                  ^            |                     |
                  |            v                     v
                  +---- QA_FAILED              CHANGES_REQUESTED
                            |
                    (max iteraciones?)
                            |
                            v
                         BLOCKED
```

---

## 2. Estructura de Archivos

```
agentic_system_template/
|
|-- orchestrator/                    # Orquestador central (corre 24/7)
|   |-- __init__.py                  # Exports del paquete
|   |-- main.py                      # Punto de entrada, clase Orchestrator
|   |-- engine/                      # Motor core
|   |   |-- langchain_setup.py       # Configuracion LLM, clientes, herramientas
|   |   |-- langgraph_workflow.py    # Maquina de estados LangGraph
|   |   |-- state_manager.py         # Persistencia de estado (File/Redis/PostgreSQL)
|   |-- github/                      # Integracion con GitHub
|   |   |-- client.py                # Cliente HTTP de bajo nivel para GitHub API
|   |   |-- issue_manager.py         # Gestor de issues a nivel de workflow
|   |   |-- webhook_handler.py       # Receptor de webhooks (skeleton)
|   |-- nodes/                       # Nodos del grafo de estados
|   |   |-- _base.py                 # Utilidades compartidas por todos los nodos
|   |   |-- triage_node.py           # Clasifica issues (feature/task/bug)
|   |   |-- planning_node.py         # Descompone features en subtareas
|   |   |-- development_node.py      # Lanza el Developer Agent
|   |   |-- qa_node.py               # Lanza el QA Agent
|   |   |-- qa_failed_node.py        # Maneja fallo de QA
|   |   |-- review_node.py           # Lanza el Reviewer Agent
|   |   |-- documentation_node.py    # Lanza el Doc Agent
|   |   |-- await_subtasks_node.py   # Espera a subtareas completadas
|   |   |-- done_node.py             # Marca issue como DONE
|   |   |-- blocked_node.py          # Marca issue como BLOCKED
|   |-- scheduler/                   # Planificacion y ejecucion
|       |-- agent_launcher.py        # Lanza y gestiona contenedores Docker
|       |-- queue_manager.py         # Cola de prioridad para issues
|
|-- agents/                          # Agentes de IA
|   |-- base/                        # Infraestructura compartida
|   |   |-- agent_interface.py       # Clase base abstracta AgentInterface
|   |   |-- llm_client.py            # Cliente LLM multi-proveedor
|   |   |-- context_loader.py        # Carga contexto desde env/volumenes
|   |   |-- output_handler.py        # Escribe resultados estructurados
|   |   |-- image_utils.py           # Procesamiento de imagenes de issues
|   |-- planner/                     # Agente Planificador
|   |   |-- planner_agent.py         # Descompone features en tareas
|   |-- developer/                   # Agente Desarrollador
|   |   |-- developer_agent.py       # Implementa codigo
|   |-- qa/                          # Agente de QA
|   |   |-- qa_agent.py              # Valida implementaciones
|   |-- reviewer/                    # Agente Revisor
|   |   |-- reviewer_agent.py        # Revisa calidad de codigo
|   |-- doc/                         # Agente de Documentacion
|       |-- doc_agent.py             # Actualiza documentacion/memoria
|
|-- monitoring/                      # Monitorizacion
|   |-- logger.py                    # Logging estructurado + audit trail
|   |-- metrics.py                   # Metricas Prometheus + alertas + health
|
|-- config/                          # Configuracion
|   |-- orchestrator.yaml            # Config del orquestador
|   |-- agents.yaml                  # Definicion de agentes
|   |-- github.yaml                  # Config de GitHub
|
|-- docker/                          # Dockerfiles
|   |-- Dockerfile.orchestrator      # Imagen del orquestador
|   |-- Dockerfile.agent             # Imagen unica de agentes
|   |-- entrypoint.sh                # Entrypoint de agentes
|
|-- memory/                          # Memoria del proyecto (Markdown)
|   |-- PROJECT.md                   # Descripcion general
|   |-- ARCHITECTURE.md              # Arquitectura tecnica
|   |-- CONVENTIONS.md               # Convenciones de codigo
|   |-- CONSTRAINTS.md               # Restricciones tecnicas
|   |-- features/                    # Memoria por feature
|       |-- _TEMPLATE.md             # Plantilla
|
|-- docker-compose.yml               # Orquestacion de servicios
|-- Makefile                         # Comandos comunes
|-- .env.template                    # Plantilla de variables de entorno
```

---

## 3. El Orquestador

El orquestador es el componente central que corre continuamente (24/7). Su trabajo es detectar issues nuevas en GitHub, procesarlas a traves de una maquina de estados, y lanzar agentes en contenedores Docker para cada paso del workflow.

### 3.1 Punto de Entrada (main.py)

#### Funcion `main()`

El punto de entrada del programa. Se ejecuta con `python -m orchestrator.main`.

1. Parsea argumentos de linea de comandos (`--config`, `--debug`, `--dry-run`)
2. Configura logging
3. Llama a `asyncio.run(async_main(...))`

#### Funcion `async_main(config_path, debug)`

1. Carga la configuracion con `load_config()`
2. Crea instancia de `Orchestrator`
3. Registra signal handlers para shutdown graceful
4. Llama a `orchestrator.setup()` y luego `orchestrator.run()`

#### Funcion `load_config(config_path) -> Dict`

Combina tres fuentes de configuracion en orden de prioridad:

1. **Valores por defecto** (hardcoded en la funcion)
2. **Archivo YAML** (`config/orchestrator.yaml`)
3. **Variables de entorno** (maxima prioridad)

Mapea variables de entorno a secciones de config. Por ejemplo:
- `GITHUB_TOKEN` -> `config["github"]["token"]`
- `LLM_PROVIDER` -> `config["llm"]["provider"]`
- `POLL_INTERVAL` -> `config["orchestrator"]["poll_interval"]`

#### Clase `Orchestrator`

La clase principal que coordina todo el sistema.

**Constructor `__init__(self, config)`**

Inicializa atributos pero **no crea componentes** todavia. Los componentes se crean en `setup()` porque requieren operaciones async.

- `self._running: bool` - Flag del bucle principal
- `self._shutdown_event: asyncio.Event` - Para shutdown graceful
- `self._tasks: List[asyncio.Task]` - Tareas async en ejecucion

**Metodo `async setup()`**

Inicializa los 7 componentes del sistema en orden:

| Orden | Componente | Clase | Funcion |
|-------|------------|-------|---------|
| 1 | GitHub Client | `GitHubClient` | Comunicacion con GitHub API |
| 2 | Issue Manager | `IssueManager` | Gestion de issues a nivel workflow |
| 3 | State Manager | `StateManager` | Persistencia de estado |
| 4 | LangChain Engine | `LangChainEngine` | Interaccion con LLMs |
| 5 | Agent Launcher | `AgentLauncher` | Lanzamiento de contenedores |
| 6 | Workflow Engine | `WorkflowEngine` | Maquina de estados LangGraph |
| 7 | Queue Manager | `QueueManager` | Cola de issues por procesar |

Ademas llama a `issue_manager.ensure_labels_exist()` para crear las labels de workflow en el repositorio si no existen.

**Metodo `async run()`**

El bucle principal. Se ejecuta indefinidamente hasta que se llame a `stop()`.

```
while self._running:
    1. queue_manager.refresh_queue()     # Busca nuevas issues READY en GitHub
    2. self._process_queue()             # Procesa issues de la cola
    3. Espera poll_interval segundos (o shutdown)
```

Usa `asyncio.wait_for(shutdown_event.wait(), timeout=poll_interval)` para implementar un sleep interrumpible.

**Metodo `async _process_queue()`**

Extrae issues de la cola y las procesa en paralelo:

1. Comprueba si puede lanzar mas agentes (`current < max_concurrent`)
2. Obtiene la siguiente issue de la cola (`queue_manager.get_next()`)
3. Crea una `asyncio.Task` para procesarla en background
4. Limpia tareas completadas de la lista

**Metodo `async process_issue(issue_number)`**

Procesa una issue individual:

1. Ejecuta el workflow completo: `workflow_engine.run(issue_number)`
2. Determina el resultado (completed, blocked, in_progress)
3. Actualiza el estado en la cola

**Metodo `async _recover_state()`**

Se ejecuta al inicio para recuperar issues que quedaron en progreso tras un reinicio. Llama a `issue_manager.recover_orphaned_issues()`.

**Metodo `async stop()`**

Shutdown graceful:
1. Pone `_running = False`
2. Dispara `_shutdown_event`
3. Cancela todas las tareas async
4. Espera a que terminen

---

### 3.2 Motor del Engine

#### 3.2.1 State Manager (`state_manager.py`)

Gestiona la persistencia del estado del workflow. Cada issue tiene un `WorkflowState` que registra su estado actual, historial de transiciones, output del ultimo agente, etc.

##### Enum `IssueState`

Define los estados posibles de una issue:

| Estado | Descripcion |
|--------|-------------|
| `TRIAGE` | Recien creada, pendiente de clasificacion |
| `PLANNING` | Siendo descompuesta en subtareas |
| `AWAIT_SUBTASKS` | Esperando que las subtareas terminen |
| `DEVELOPMENT` | En desarrollo |
| `QA` | En validacion de calidad |
| `QA_FAILED` | Fallo QA, volviendo a desarrollo |
| `REVIEW` | En revision de codigo |
| `DOCUMENTATION` | Actualizando documentacion |
| `DONE` | Completada exitosamente |
| `BLOCKED` | Requiere intervencion humana |

##### Dataclass `TransitionRecord`

Registro de una transicion de estado para el audit trail:

- `timestamp`: Momento de la transicion
- `from_state`: Estado origen
- `to_state`: Estado destino
- `trigger`: Que causo la transicion (ej: "qa_passed")
- `agent_type`: Agente involucrado (si aplica)
- `details`: Informacion adicional

##### Dataclass `WorkflowState`

El objeto principal de estado que fluye por el workflow:

- `issue_number: int` - Numero de issue
- `issue_state: str` - Estado actual (valor del enum `IssueState`)
- `issue_type: str` - Tipo (feature, task, bug)
- `current_agent: Optional[str]` - Agente ejecutandose actualmente
- `iteration_count: int` - Iteraciones Dev->QA realizadas
- `max_iterations: int` - Maximo antes de BLOCKED (default 5)
- `last_agent_output: Optional[Dict]` - Output del ultimo agente
- `error_message: Optional[str]` - Mensaje de error
- `parent_issue: Optional[int]` - Issue padre (para subtareas)
- `child_issues: List[int]` - Issues hijas (para features)
- `history: List[Dict]` - Historial de transiciones
- `metadata: Dict` - Datos adicionales
- `created_at / updated_at: str` - Timestamps
- `version: int` - Version para control de concurrencia

Metodo clave: `add_transition()` - Anade un `TransitionRecord` al historial, actualiza el estado, incrementa la version.

##### Interfaz `StateManagerInterface` (ABC)

Define el contrato que deben cumplir todos los backends de persistencia:

| Metodo | Descripcion |
|--------|-------------|
| `get_state(issue_number)` | Recuperar estado |
| `save_state(issue_number, state)` | Guardar estado |
| `delete_state(issue_number)` | Eliminar estado |
| `list_active_issues()` | Listar issues activas |
| `get_issues_by_state(workflow_state)` | Filtrar por estado |
| `health_check()` | Verificar salud del backend |

##### Backend `FileStateManager`

Almacena estado en un archivo JSON. Adecuado para desarrollo y baja carga.

**Estructura del archivo:**
```json
{
  "issues": {
    "123": { "issue_state": "DEVELOPMENT", ... },
    "124": { "issue_state": "QA", ... }
  },
  "metadata": {
    "version": "1.0",
    "last_updated": "2024-01-01T00:00:00Z"
  }
}
```

Caracteristicas:
- **Cache con hash MD5**: Lee del disco solo si el hash del archivo cambio
- **Escritura atomica**: Escribe a un `.tmp` y luego renombra
- **Backups automaticos**: Crea copias de seguridad antes de cada escritura
- **Thread-safe**: Usa `threading.RLock` para acceso concurrente

##### Backend `RedisStateManager`

Almacena estado en Redis. Recomendado para produccion.

**Estructura de claves:**
- `orchestrator:{project_id}:state:{issue_number}` -> JSON del estado
- `orchestrator:{project_id}:index:state:{workflow_state}` -> Set de issues en ese estado
- `orchestrator:{project_id}:all_issues` -> Set de todas las issues

Caracteristicas:
- **Indices por estado**: Permite consultar issues por estado en O(1) usando Redis Sets
- **Transacciones**: Usa `pipeline(transaction=True)` para operaciones atomicas
- **TTL opcional**: Puede expirar estados automaticamente

##### Backend `PostgreSQLStateManager`

Almacena estado en PostgreSQL. Para necesidades complejas de consulta.

**Schema:**
```sql
CREATE TABLE workflow_state (
    issue_number INTEGER NOT NULL,
    project_id VARCHAR(255) NOT NULL,
    workflow_state VARCHAR(50) NOT NULL,
    state_json JSONB NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    PRIMARY KEY (project_id, issue_number)
);
```

Caracteristicas:
- **UPSERT**: Usa `ON CONFLICT DO UPDATE` para crear o actualizar
- **Connection pool**: Usa `asyncpg` con pool configurable
- **Auto-migration**: Crea la tabla automaticamente si no existe

##### Clase `StateManager` (Factory + Wrapper)

Envuelve el backend seleccionado y anade metodos de alto nivel:

- `create(config)` -> Factory que instancia el backend segun `config["persistence"]["backend"]`
- `create_initial_state(issue_number, issue_type)` -> Crea el estado inicial en TRIAGE
- `transition_state(issue_number, to_state, trigger, ...)` -> Transicion con historial y validacion
- `increment_iteration(issue_number)` -> Incrementa el contador de iteraciones
- `set_agent_output(issue_number, agent_type, output)` -> Almacena output del agente
- `set_error(issue_number, error_message)` -> Registra un error

##### Funcion `recover_orphaned_states(state_manager, agent_launcher)`

Busca issues que quedaron en estados activos (DEVELOPMENT, QA, REVIEW, DOCUMENTATION) pero cuyo contenedor de agente ya no esta corriendo. Las devuelve a su estado anterior.

##### Funcion `cleanup_completed_states(state_manager, retention_days=7)`

Elimina estados de issues terminadas (DONE, BLOCKED) que tienen mas de `retention_days` dias.

---

#### 3.2.2 LangChain Engine (`langchain_setup.py`)

Gestiona las interacciones con LLMs y las herramientas que el orquestador usa para tomar decisiones.

##### Enum `LLMProvider`

Proveedores soportados: `ANTHROPIC`, `OPENAI`, `AZURE`, `OLLAMA`.

##### Dataclass `LLMConfig`

Configuracion del LLM:
- `provider`, `model`, `temperature`, `max_tokens`, `timeout`, `max_retries`
- `api_base`, `api_version` (para Azure)

##### Dataclass `ToolCall`

Representa una llamada a herramienta del LLM: `id`, `name`, `arguments`.

##### Dataclass `ToolResult`

Resultado de ejecutar una herramienta: `tool_call_id`, `output`, `error`.

##### Dataclass `LLMResponse`

Respuesta del LLM: `content`, `tool_calls`, `input_tokens`, `output_tokens`, `model`, `latency_ms`.

##### Dataclass `LLMMetrics`

Metricas acumuladas: `total_requests`, `total_input_tokens`, `total_output_tokens`, `total_latency_ms`, `errors`.

##### Constante `TOOL_SCHEMAS`

Define 7 herramientas que el LLM del orquestador puede usar:

| Herramienta | Parametros | Funcion |
|-------------|------------|---------|
| `read_issue` | `issue_number` | Lee una issue de GitHub |
| `update_issue` | `issue_number, labels, state, comment` | Actualiza una issue |
| `launch_agent` | `agent_type, issue_number, context` | Lanza un agente |
| `read_memory` | `file_name` | Lee un archivo de memoria |
| `check_agent_status` | `container_id` | Verifica estado de un agente |
| `list_issues` | `state, labels` | Lista issues |
| `create_subtask` | `parent_issue, title, body, labels` | Crea una subtarea |

##### Prompts

Tres prompts predefinidos para el orquestador:

- **`TRIAGE_PROMPT`**: Clasifica una issue como feature, task o bug. Analiza titulo, body y labels para decidir.
- **`TRANSITION_PROMPT`**: Decide la siguiente transicion de estado basandose en el output del agente, feedback de QA, iteraciones, etc.
- **`PLANNING_PROMPT`**: Guia la descomposicion de features en subtareas.

##### Interfaz `LLMClientInterface` (ABC)

Define el contrato para clientes LLM:

- `invoke(messages, tools, **kwargs) -> LLMResponse`
- `get_model_name() -> str`

##### Clientes concretos

Cada cliente implementa `LLMClientInterface`:

**`AnthropicClient`**: Usa la API de Anthropic (Claude). Convierte herramientas al formato `input_schema` que espera la API de Anthropic. Extrae `tool_use` blocks de la respuesta.

**`OpenAIClient`**: Usa la API de OpenAI (GPT). Convierte herramientas al formato `function` de OpenAI. Parsea `tool_calls` de la respuesta.

**`AzureOpenAIClient`**: Similar a OpenAI pero con endpoint y version de API de Azure.

**`OllamaClient`**: Usa la API REST de Ollama (`/api/chat`). Compatible con modelos locales. Usa `aiohttp` para llamadas async.

##### Clase `ToolRegistry`

Registro de herramientas disponibles:

- `register_schema(name, schema)` - Registra la definicion de una herramienta
- `register_handler(name, handler)` - Registra la funcion que la ejecuta
- `execute(tool_call) -> ToolResult` - Ejecuta una herramienta

##### Clase `LangChainEngine`

La clase principal del motor LLM:

**`_init_client()`**: Crea el cliente LLM adecuado segun `provider`:
```
anthropic -> AnthropicClient
openai    -> OpenAIClient
azure     -> AzureOpenAIClient
ollama    -> OllamaClient
```

**`invoke(messages, tools, execute_tools, max_tool_iterations)`**: Envia mensajes al LLM, opcionalmente con herramientas. Si el LLM devuelve tool calls, las ejecuta y reenvia los resultados en un bucle hasta que el LLM responda sin tool calls (o se alcance el limite).

**`_execute_tool_loop(messages, response, tool_schemas, max_iterations)`**: Implementa el bucle de herramientas:
1. El LLM responde con tool_calls
2. Se ejecutan las herramientas via `ToolRegistry`
3. Los resultados se anaden a los mensajes
4. Se vuelve a invocar al LLM
5. Se repite hasta que no haya tool_calls

**`triage_issue(issue_number, title, body, labels, project_context)`**: Usa `TRIAGE_PROMPT` para clasificar una issue. Retorna `{"issue_type": "feature"|"task"|"bug", "reasoning": "..."}`.

**`determine_transition(issue_number, current_state, issue_type, iteration_count, ...)`**: Usa `TRANSITION_PROMPT` para decidir el siguiente estado. Retorna `{"next_state": "...", "reasoning": "..."}`.

**`plan_subtasks(issue_number, title, body, architecture, conventions)`**: Usa `PLANNING_PROMPT` para descomponer features. Retorna una lista de subtareas.

---

#### 3.2.3 LangGraph Workflow (`langgraph_workflow.py`)

Define la maquina de estados que orquesta todo el flujo de trabajo.

##### TypedDict `GraphState`

El estado que fluye por el grafo de LangGraph. Contiene todos los campos necesarios para cada nodo:

- `issue_number`, `issue_state`, `issue_type`
- `issue_title`, `issue_body`, `issue_labels`
- `current_agent`, `iteration_count`, `max_iterations`
- `last_agent_output`, `qa_result`, `review_result`
- `error_message`, `parent_issue`, `child_issues`
- `history`, `metadata`, `created_at`, `updated_at`

##### Dataclass `NodeResult`

Resultado de un nodo: `success`, `next_state`, `error_message`, `agent_output`, `metadata_updates`.

##### Clase `WorkflowEngine`

La clase central que define y ejecuta el workflow.

**Constructor**: Recibe todos los componentes del sistema (`langchain_engine`, `state_manager`, `agent_launcher`, `github_client`).

**`_setup_node_handlers()`**: Registra los handlers de cada nodo:
```python
self._node_handlers = {
    "triage":          self._triage_node,
    "planning":        self._planning_node,
    "await_subtasks":  self._await_subtasks_node,
    "development":     self._development_node,
    "qa":              self._qa_node,
    "qa_failed":       self._qa_failed_node,
    "review":          self._review_node,
    "documentation":   self._documentation_node,
    "done":            self._done_node,
    "blocked":         self._blocked_node,
}
```

**`_build_graph()`**: Construye el grafo de LangGraph:

```
triage ---[feature]---> planning ---[success]---> await_subtasks ---[complete]---> documentation
  |                        |                            |
  |--[task/bug]----> development ---[success]---> qa ---[pass]---> review ---[approved]---> documentation
                         ^                         |                  |
                         |                   [fail_retriable]   [changes_requested]
                         |                         |                  |
                         +---- qa_failed <---------+                  |
                         |         |                                  |
                         +---------+----------------------------------+
                                   |
                            [max_iterations]
                                   |
                                   v
                               blocked
```

Registra nodos con `graph.add_node()`, configura transiciones con `graph.add_conditional_edges()`, y compila con `graph.compile()`.

**`run(issue_number)`**: Ejecuta el workflow completo:
1. Inicializa el estado (`_initialize_state`)
2. Obtiene datos de la issue de GitHub
3. Intenta ejecutar con LangGraph (`_run_with_langgraph`)
4. Si LangGraph no esta disponible, usa fallback manual (`_run_fallback`)
5. Persiste el estado final

**`_run_fallback(state)`**: Implementacion manual del workflow sin LangGraph:
```python
while current_node != "done" and current_node != "blocked":
    handler = self._node_handlers[current_node]
    state = await handler(state)
    current_node = self._get_next_node(current_node, state)
```

##### Nodos internos del WorkflowEngine

Cada nodo es un metodo async que recibe y devuelve `GraphState`:

**`_triage_node(state)`**: Usa `langchain_engine.triage_issue()` para clasificar la issue. Actualiza `issue_type` y `issue_state`.

**`_planning_node(state)`**: Lanza el Planner Agent via `_launch_agent(AgentType.PLANNER, ...)`. Almacena las subtareas creadas en `child_issues`.

**`_development_node(state)`**: Lanza el Developer Agent. Incluye QA feedback previo si es un reintento. Actualiza labels a `IN_PROGRESS`.

**`_qa_node(state)`**: Lanza el QA Agent. Almacena `qa_result` (pass/fail). Actualiza labels a `QA`.

**`_qa_failed_node(state)`**: Incrementa `iteration_count`. Si se alcanzo `max_iterations`, transiciona a BLOCKED. Si no, vuelve a DEVELOPMENT.

**`_review_node(state)`**: Lanza el Reviewer Agent. Almacena `review_result` (approved/changes_requested).

**`_documentation_node(state)`**: Lanza el Doc Agent. Actualiza la memoria del proyecto.

**`_done_node(state)`**: Cierra la issue en GitHub, pone label `DONE`, posta comentario de resumen.

**`_blocked_node(state)`**: Pone label `BLOCKED`, posta comentario explicando por que se bloqueo.

##### Routers

Funciones que determinan el siguiente nodo basandose en el estado:

**`_triage_router(state)`**: Retorna `"feature"` (-> planning), `"task"`/`"bug"` (-> development), o `"invalid"` (-> blocked).

**`_qa_router(state)`**: Retorna `"pass"` (-> review), `"fail_retriable"` (-> qa_failed), o `"fail_blocked"` (-> blocked).

**`_review_router(state)`**: Retorna `"approved"` (-> documentation) o `"changes_requested"` (-> development).

**`_await_subtasks_router(state)`**: Retorna `"complete"` (-> documentation), `"waiting"` (sigue esperando), o `"error"` (-> blocked).

##### Helper `_launch_agent(agent_type, issue_number, context)`

1. Llama a `agent_launcher.launch_agent()` para crear el contenedor
2. Llama a `agent_launcher.wait_for_completion()` para esperar resultado
3. Retorna `AgentResult` con output, logs y estado

---

### 3.3 Integracion con GitHub

#### 3.3.1 GitHub Client (`client.py`)

Cliente HTTP de bajo nivel para la API de GitHub.

##### Excepciones

| Excepcion | Uso |
|-----------|-----|
| `GitHubAPIError` | Base, incluye `status_code` y `response` |
| `RateLimitError` | Rate limit alcanzado, incluye `reset_time` |
| `NotFoundError` | Recurso no encontrado (404) |
| `AuthenticationError` | Token invalido (401/403) |
| `ValidationError` | Datos invalidos (422), incluye `errors` |

##### Clase `GitHubClient`

**Constructor**: Configura token, repo, base_url, timeout, retry. Crea `requests.Session` con reintentos automaticos y headers de autenticacion.

**`_create_session()`**: Configura:
- `Authorization: Bearer {token}`
- `Accept: application/vnd.github+json`
- `X-GitHub-Api-Version: 2022-11-28`
- Retry con `HTTPAdapter` (3 reintentos, backoff 0.5s, en status 429/500/502/503/504)

**Operaciones de Issues:**

| Metodo | Endpoint | Descripcion |
|--------|----------|-------------|
| `get_issue(n)` | `GET /issues/{n}` | Lee issue completa |
| `create_issue(title, body, labels)` | `POST /issues` | Crea issue nueva |
| `update_issue(n, ...)` | `PATCH /issues/{n}` | Actualiza issue |
| `close_issue(n)` | `PATCH /issues/{n}` | Cierra issue (state=closed) |
| `list_issues(state, labels, ...)` | `GET /issues` | Lista issues con filtros |
| `list_all_issues(...)` | Paginado auto | Todas las issues (autopaginado) |

**Operaciones de Labels:**

| Metodo | Descripcion |
|--------|-------------|
| `add_labels(n, labels)` | Anade labels a issue |
| `remove_label(n, label)` | Quita una label |
| `set_labels(n, labels)` | Reemplaza todas las labels |
| `create_label(name, color, desc)` | Crea label en el repo |
| `get_or_create_label(name, color, desc)` | Crea si no existe |

**Operaciones de Comentarios:**

| Metodo | Descripcion |
|--------|-------------|
| `add_comment(n, body)` | Anade comentario |
| `update_comment(id, body)` | Edita comentario |
| `delete_comment(id)` | Elimina comentario |
| `list_comments(n)` | Lista comentarios |

**Rate Limiting:**

- `_check_rate_limit()`: Antes de cada request, verifica si quedan suficientes requests. Si queda menos de `RATE_LIMIT_THRESHOLD` (10), espera hasta el reset.
- `_update_rate_limit(response)`: Actualiza contadores desde los headers `X-RateLimit-Remaining` y `X-RateLimit-Reset`.

#### 3.3.2 Issue Manager (`issue_manager.py`)

Gestor de alto nivel que traduce el workflow del sistema a operaciones de GitHub.

##### Enum `WorkflowLabel`

Todos los labels de estado: `READY`, `IN_PROGRESS`, `PLANNING`, `DEVELOPMENT`, `QA`, `QA_FAILED`, `REVIEW`, `DOCUMENTATION`, `BLOCKED`, `DONE`, `AWAIT_SUBTASKS`.

##### Enum `IssueType`

Tipos de issue: `FEATURE`, `TASK`, `BUG`, `SUBTASK`.

##### Constante `COMMENT_TEMPLATES`

Templates para los comentarios que los agentes dejan en las issues. Incluye templates para:
- `agent_start` / `agent_complete`
- `qa_passed` / `qa_failed`
- `blocked` / `done`
- `subtask_created` / `planning_complete`
- `review_approved` / `review_changes_requested`

##### Clase `IssueManager`

**Metodo `_transition(issue_number, to_state, additional_labels, remove_states, keep_labels)`**

El metodo privado central para transiciones. Hace tres cosas:
1. Quita todos los labels de estado existentes (excepto `keep_labels`)
2. Anade el nuevo label de estado
3. Anade labels adicionales si se especifican

**Transiciones de estado** (metodos publicos):

Cada metodo es un wrapper de `_transition()` con la logica especifica:

- `transition_to_ready(n)` -> Label `READY`
- `transition_to_in_progress(n, agent_type)` -> Label `IN_PROGRESS` + `agent:{type}` + comentario
- `transition_to_qa(n)` -> Label `QA`
- `transition_to_qa_failed(n, feedback, iteration)` -> Label `QA_FAILED` + comentario con feedback
- `transition_to_review(n)` -> Label `REVIEW`
- `transition_to_blocked(n, reason)` -> Label `BLOCKED` + comentario
- `transition_to_done(n, summary)` -> Label `DONE` + cierra issue + comentario

**Comunicacion con agentes:**

- `post_agent_start(n, agent_type, iteration)` - Comenta que un agente empezo a trabajar
- `post_agent_complete(n, agent_type, result)` - Comenta que termino
- `post_qa_result(n, passed, details)` - Comenta resultado de QA con checklist
- `post_review_result(n, approved, notes)` - Comenta resultado de review
- `post_planning_complete(n, subtasks)` - Comenta las subtareas creadas

**Creacion de issues:**

- `create_subtask(parent_number, title, body, labels)` - Crea una subtarea vinculada al padre con label `subtask`
- `create_issue(title, body, labels)` - Crea issue generica

**Recuperacion:**

- `recover_orphaned_issues()` - Busca issues con `IN_PROGRESS` que no tienen agente activo, las devuelve a `READY`
- `get_workflow_summary()` - Retorna conteo de issues por estado

---

### 3.4 Nodos del Workflow

Cada nodo es una funcion async que recibe `(state: Dict, ctx: NodeContext) -> Dict`. El `NodeContext` es una dataclass que proporciona acceso a todos los componentes del sistema.

#### Utilidades Compartidas (`_base.py`)

##### Dataclass `NodeContext`

Contiene referencias a todos los componentes necesarios:
- `github_client`, `issue_manager`, `langchain_engine`, `state_manager`, `agent_launcher`
- Properties: `max_iterations`, `agent_timeout`, `memory_path`, `repo_path`

##### Funciones de utilidad

- **`launch_agent_and_wait(ctx, agent_type, issue_number, context, timeout)`**: Lanza un agente en contenedor Docker y espera a que termine. Retorna `AgentResult`.

- **`update_labels(ctx, issue_number, add, remove)`**: Actualiza labels de una issue.

- **`post_comment(ctx, issue_number, body)`**: Anade comentario a una issue.

- **`add_history_entry(state, node, action, details)`**: Anade una entrada al historial de la issue en el state.

- **`extract_acceptance_criteria(body)`**: Parsea el markdown del body buscando lineas con `- [ ]` o `- [x]` bajo la seccion "Acceptance Criteria".

- **`load_memory_file(memory_path, filename)`**: Carga un archivo de la carpeta memory.

- **`load_project_context(memory_path)`**: Carga `PROJECT.md`, `ARCHITECTURE.md`, `CONVENTIONS.md` y `CONSTRAINTS.md`.

#### Nodo `triage_node`

Clasifica una issue nueva. Usa el LLM (`langchain_engine.triage_issue()`) para analizar titulo, body y labels, y determinar si es feature, task o bug.

Router `triage_router`: Retorna `"feature"`, `"task"`, `"bug"` o `"invalid"` segun `issue_type`.

#### Nodo `planning_node`

1. Carga contexto del proyecto (memory files)
2. Lanza el Planner Agent
3. Crea subtareas como issues de GitHub via `issue_manager.create_subtask()`
4. Crea archivo de feature memory en `memory/features/`
5. Comenta en la issue padre con resumen de la planificacion

Validador `validate_planner_output(output)`: Verifica que el output tenga `tasks` (lista no vacia) y `summary`.

#### Nodo `development_node`

1. Carga QA feedback previo (si es reintento)
2. Carga contexto del proyecto
3. Lanza el Developer Agent con toda la informacion
4. Almacena output (archivos modificados, notas de implementacion)

Validador `validate_developer_output(output)`: Verifica que tenga `modified_files` (lista no vacia).

#### Nodo `qa_node`

1. Obtiene la lista de archivos modificados del output del Developer
2. Lanza el QA Agent
3. Almacena `qa_result` (PASS o FAIL) y feedback

Router `qa_router`: Si `qa_result == "PASS"` -> `"pass"`. Si fallo y `iteration_count < max_iterations` -> `"fail_retriable"`. Si se alcanzaron las iteraciones -> `"fail_blocked"`.

Helper `format_qa_comment(qa_result, checklist, test_results, feedback)`: Formatea un comentario markdown con los resultados de QA.

#### Nodo `qa_failed_node`

1. Incrementa `iteration_count`
2. Posta comentario con feedback de QA para el developer
3. Actualiza labels a `QA_FAILED`

#### Nodo `review_node`

1. Lanza el Reviewer Agent
2. Almacena `review_result` (APPROVED o CHANGES_REQUESTED)

Router `review_router`: Si `review_result == "APPROVED"` -> `"approved"`. Si `"CHANGES_REQUESTED"` -> `"changes_requested"`. Otro -> `"error"`.

#### Nodo `documentation_node`

1. Lanza el Doc Agent
2. El agente actualiza feature memory, changelog, etc.
3. El nodo transiciona a DONE independientemente del resultado (fallo de doc no bloquea)

#### Nodo `done_node`

1. Cierra la issue en GitHub
2. Pone label `DONE`
3. Posta comentario de resumen final

#### Nodo `blocked_node`

1. Pone label `BLOCKED`
2. Posta comentario explicando el motivo del bloqueo
3. Registra el error en el state

#### Nodo `await_subtasks_node`

Para features descompuestas en subtareas:
1. Lee las child issues de GitHub
2. Verifica si todas tienen label `DONE`
3. Si todas completadas -> `"complete"` (va a documentation)
4. Si alguna BLOCKED -> `"error"` (va a blocked)
5. Si alguna pendiente -> `"waiting"` (se queda esperando)

---

### 3.5 Scheduler (Planificador)

#### 3.5.1 Agent Launcher (`agent_launcher.py`)

Gestiona el ciclo de vida de contenedores Docker para los agentes.

##### Enum `AgentType`

Tipos de agente: `PLANNER`, `DEVELOPER`, `QA`, `REVIEWER`, `DOC`.

##### Enum `ContainerStatus`

Estados de contenedor: `PENDING`, `RUNNING`, `COMPLETED`, `FAILED`, `TIMEOUT`, `NOT_FOUND`.

##### Dataclass `AgentResult`

Resultado de ejecucion de un agente:
- `status: ContainerStatus`
- `exit_code: Optional[int]`
- `output: Optional[Dict]` - Output JSON del agente
- `logs: str` - Logs del contenedor
- `error_message: Optional[str]`
- `started_at / finished_at: Optional[datetime]`
- `duration_seconds: float`
- Property `success`: True si `status == COMPLETED` y `exit_code == 0`

##### Dataclass `ContainerInfo`

Informacion de un contenedor en ejecucion: `container_id`, `agent_type`, `issue_number`, `started_at`, `status`.

##### Clase `AgentLauncher`

**`launch_agent(agent_type, issue_number, context, timeout)`**:

1. Construye variables de entorno (`_build_environment`)
2. Construye volumenes (`_build_volumes`)
3. Escribe archivo de input (`_write_input_file`)
4. Crea contenedor Docker con:
   - Imagen: `ai-agent:latest`
   - Red: `ai-agent-network`
   - Limites de CPU y memoria
   - Volumenes montados: `/memory`, `/repo`, `/output`, `/input`
5. Inicia el contenedor
6. Registra en el `ContainerPool`
7. Retorna `container_id`

**`wait_for_completion(container_id, timeout, poll_interval)`**:

1. Loop de polling cada `poll_interval` segundos
2. Consulta `container.status` via Docker API
3. Si el contenedor termino: recoge logs, parsea output, retorna `AgentResult`
4. Si pasa el timeout: mata el contenedor, retorna `AgentResult` con `TIMEOUT`

**`_build_environment(agent_type, issue_number, timeout)`**: Construye el dict de variables de entorno para el contenedor:
```python
{
    "AGENT_TYPE": "developer",
    "PROJECT_ID": "mi-proyecto",
    "ISSUE_NUMBER": "42",
    "GITHUB_TOKEN": "ghp_...",
    "GITHUB_REPO": "owner/repo",
    "LLM_PROVIDER": "anthropic",
    "ANTHROPIC_API_KEY": "sk-ant-...",
    ...
}
```

**`_build_volumes(issue_number)`**: Monta 4 volumenes:
- `./memory` -> `/memory` (lectura/escritura)
- `./repo` -> `/repo` (lectura/escritura)
- `./output/{issue_number}` -> `/output` (escritura)
- `./input/{issue_number}` -> `/input` (lectura)

**`_parse_agent_output(container_id)`**: Lee el archivo `/output/result.json` del contenedor para obtener el output estructurado del agente.

##### Clase `ContainerPool`

Gestiona los contenedores activos:
- `can_launch()` - True si hay espacio para otro contenedor
- `register(container_id, info)` - Registra nuevo contenedor
- `unregister(container_id)` - Elimina registro
- `is_issue_running(issue_number)` - Verifica si ya hay un agente para esa issue

#### 3.5.2 Queue Manager (`queue_manager.py`)

Cola de prioridad para las issues pendientes de procesamiento.

##### Enum `Priority` (IntEnum)

Prioridades: `CRITICAL=0`, `HIGH=1`, `MEDIUM=2` (default), `LOW=3`.

Las issues con label `priority:critical` obtienen prioridad 0 (se procesan primero).

##### Enum `QueueStatus`

Estados de un item en la cola: `PENDING`, `PROCESSING`, `COMPLETED`, `FAILED`, `BLOCKED`.

##### Dataclass `QueueItem`

Un item en la cola, ordenable por prioridad y timestamp:
- `priority: int` - Menor = mayor prioridad
- `timestamp: float` - Momento de encolado (para FIFO dentro de misma prioridad)
- `issue_number: int`
- `issue_data: Dict` - Datos de la issue
- `status: str` - Estado en la cola
- `dependencies: List[int]` - Issues de las que depende
- `retry_count / max_retries: int`

Usa `@dataclass(order=True)` con `compare=True` solo en `priority` y `timestamp`, asi Python puede ordenar automaticamente por prioridad y luego por FIFO.

##### Dataclass `QueueStats`

Estadisticas de la cola: `total_items`, `pending_items`, `processing_items`, `completed_items`, `failed_items`, `blocked_items`, `avg_wait_time`, `avg_process_time`.

##### Backend `MemoryQueueBackend`

Implementacion en memoria usando `heapq`:
- `_heap: List[QueueItem]` - Min-heap para extraer por prioridad
- `_items: Dict[int, QueueItem]` - Lookup O(1) por issue_number
- `_lock: asyncio.Lock` - Para acceso concurrente

`push()` anade al heap. `pop()` extrae el item de menor prioridad que este en estado PENDING.

##### Backend `RedisQueueBackend`

Implementacion con Redis sorted sets:
- Score = `priority * 10^10 + timestamp` (ordena por prioridad, luego FIFO)
- Los datos del item se guardan como JSON en claves separadas

##### Clase `RateLimiter`

Implementa token bucket para rate limiting:
- `_tokens: float` - Tokens disponibles (se regeneran con el tiempo)
- `acquire()` - Intenta consumir un token. Retorna False si no hay disponible.
- `wait_for_token(timeout)` - Espera hasta que haya un token o se agote el timeout.

##### Clase `QueueManager`

La clase principal de gestion de cola:

**`refresh_queue()`**:
1. Llama a `github_client.list_issues(labels=["READY"])`
2. Crea `QueueItem` para cada issue nueva
3. Las encola con `enqueue()`
4. Retorna el numero de issues anadidas

**`enqueue(issue)`**:
1. Calcula prioridad segun labels (`_calculate_priority`)
2. Extrae dependencias (`_extract_dependencies`)
3. Crea `QueueItem` y lo anade al backend

**`get_next()`**:
1. Saca el siguiente item del backend
2. Verifica dependencias (`_check_dependencies`)
3. Si tiene dependencias no resueltas, lo reencola
4. Marca como PROCESSING
5. Retorna `issue_data`

**`_calculate_priority(issue)`**: Mapea labels de prioridad:
- `priority:critical` -> 0
- `priority:high` -> 1
- `priority:medium` -> 2
- `priority:low` -> 3
- Sin label -> 2 (medium)

**`_extract_dependencies(issue)`**: Busca en el body de la issue patrones como "depends on #42", "blocked by #15", "requires #7" y extrae los numeros de issue.

---

## 4. Los Agentes

Todos los agentes viven dentro de contenedores Docker y comparten la misma imagen base. Se diferencian por la variable `AGENT_TYPE`. Todos extienden `AgentInterface`.

### 4.1 Base Compartida

#### 4.1.1 Agent Interface (`agent_interface.py`)

##### Enum `AgentStatus`

Resultados de ejecucion: `SUCCESS`, `FAILURE`, `ERROR`, `TIMEOUT`.

##### Dataclass `AgentResult`

Estructura estandar de resultado que retornan todos los agentes:
- `status: AgentStatus`
- `output: Dict[str, Any]` - Datos principales del resultado
- `message: str` - Resumen legible
- `details: Dict` - Detalles adicionales
- `errors: List[str]` - Lista de errores
- `metrics: Dict` - Metricas (duracion, tokens, archivos, etc.)
- `timestamp: str`

Metodos de fabrica:
- `AgentResult.error(message, errors)` - Crea resultado de error
- `AgentResult.failure(message, output, details)` - Crea resultado de fallo

Serializacion:
- `to_dict()` - Convierte a diccionario
- `from_dict(data)` - Crea desde diccionario

##### Dataclass `AgentContext`

Contiene toda la informacion que necesita un agente:
- `issue_number: int` - Numero de issue
- `project_id: str` - ID del proyecto
- `iteration: int` - Iteracion actual (para dev loop)
- `max_iterations: int`
- `issue_data: Dict` - Datos de la issue (titulo, body, labels)
- `memory: Dict[str, str]` - Contenido de archivos de memoria
- `repository: Dict` - Metadata del repositorio
- `input_data: Dict` - Input del orquestador
- `config: Dict` - Configuracion del agente
- `images: List` - Imagenes extraidas de la issue (mockups, screenshots)

Properties de conveniencia:
- `has_images` - True si hay imagenes disponibles
- `issue_title` / `issue_body` / `issue_labels` - Acceso directo
- `get_memory_file(name)` / `has_memory_file(name)` - Acceso a memoria

##### Clase Abstracta `AgentInterface`

**Constructor `__init__()`**:
- Crea logger
- Inicializa metricas: `llm_calls`, `tokens_input`, `tokens_output`, `files_read`, `files_modified`
- Propiedades lazy: `llm`, `github`, `output_handler`, `context_loader`

**Metodo `run()` (NO override)**:

El punto de entrada principal que orquesta el ciclo de vida:

```python
def run(self) -> AgentResult:
    1. context = self._load_context()      # Carga de ContextLoader
    2. if not self.validate_context(context):
           return error                    # Contexto invalido
    3. result = self.execute(context)       # Logica del agente (override)
    4. self._finalize_result(result)        # Anade metricas
    5. self._write_output(result)           # Escribe a /output
    6. self._update_github(context, result) # Comenta en la issue
    7. return result
```

**Metodos abstractos (deben implementarse):**
- `get_agent_type() -> str` - Retorna "planner", "developer", "qa", "reviewer" o "doc"
- `validate_context(context: AgentContext) -> bool` - Valida que el contexto sea suficiente
- `execute(context: AgentContext) -> AgentResult` - Logica principal del agente

**Metricas:**
- `track_llm_call(tokens_input, tokens_output)` - Registra llamada LLM
- `track_file_read(count)` - Registra lectura de archivos
- `track_file_modified(count)` - Registra modificacion de archivos

#### 4.1.2 LLM Client (`llm_client.py`)

Cliente LLM multi-proveedor que usan todos los agentes.

##### Dataclass `LLMResponse`

Respuesta del LLM: `content`, `model`, `tokens_input`, `tokens_output`, `finish_reason`, `latency_ms`.

##### Dataclass `TextBlock` / `ImageBlock`

Bloques de contenido para mensajes multimodales:
- `TextBlock`: `text: str`, `type: str = "text"`
- `ImageBlock`: `media_type: str`, `base64_data: str`, `alt_text: str`, `type: str = "image"`

##### Dataclass `LLMMessage`

Mensaje en una conversacion. Soporta texto plano y contenido multimodal:

- `role: str` - "system", "user", "assistant"
- `content: Union[str, List[Union[TextBlock, ImageBlock]]]`

Cuando `content` es `str`, funciona como texto plano (backward compatible). Cuando es una lista de blocks, es multimodal.

Properties:
- `is_multimodal` - True si contiene ImageBlocks
- `text_content` - Extrae solo el texto (para logging/estimacion de tokens)

Metodo `to_dict()`: Serializa al formato apropiado para cada proveedor.

##### Proveedores

**`AnthropicProvider`**: Formatea imagenes como:
```json
{"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "..."}}
```

**`OpenAIProvider`**: Formatea imagenes como:
```json
{"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
```

**`OllamaProvider`**: Formatea imagenes como array separado:
```json
{"content": "texto", "images": ["base64data1", "base64data2"]}
```

##### Clase `LLMClient`

**`complete(prompt, system, max_tokens, temperature, images)`**: Genera una respuesta simple.
1. Construye lista de `LLMMessage`
2. Si hay imagenes, usa `create_multimodal_message()` para incluirlas
3. Llama a `chat()`

**`chat(messages, max_tokens, temperature)`**: Envio directo de mensajes al proveedor.

##### Funcion `create_multimodal_message(role, text, images)`

Funcion helper que crea un `LLMMessage`:
- Si `images` es None o vacio: retorna mensaje de texto plano (compatible con todo)
- Si hay imagenes: crea lista de `ImageBlock` + `TextBlock`

##### Funcion `create_llm_client(provider, model, **kwargs) -> LLMClient`

Factory que crea el cliente segun variables de entorno `LLM_PROVIDER`.

##### Funcion `estimate_tokens(text) -> int`

Estima tokens. Intenta usar tiktoken, y si no esta disponible, usa la heuristica ~4 caracteres por token.

#### 4.1.3 Context Loader (`context_loader.py`)

Carga todo el contexto que necesita un agente desde el entorno y los volumenes montados.

##### Clase `ContextLoader`

**`load() -> AgentContext`**:

Ejecuta 7 pasos en orden:

1. **Cargar variables de entorno** (`_load_environment`): Lee `AGENT_TYPE`, `PROJECT_ID`, `ISSUE_NUMBER`, `GITHUB_TOKEN`, `GITHUB_REPO`, `MAX_ITERATIONS`, `ITERATION`, etc.

2. **Cargar memoria** (`_load_memory`): Lee los archivos `.md` de la carpeta `/memory`: `PROJECT.md`, `ARCHITECTURE.md`, `CONVENTIONS.md`, `CONSTRAINTS.md`.

3. **Cargar memoria de feature** (`_load_feature_memory`): Busca un archivo de feature memory para la issue actual en `memory/features/`.

4. **Obtener datos de la issue** (`_fetch_issue`): Usa `GitHubHelper` para llamar a la API de GitHub y obtener titulo, body, labels, comments.

5. **Cargar input del orquestador** (`_load_orchestrator_input`): Lee `/input/input.json` (archivo que el orquestador escribe antes de lanzar el agente).

6. **Cargar info del repositorio** (`_load_repository_info`): Lee metadata del repo (lenguajes, estructura de carpetas).

7. **Extraer imagenes** (`_extract_images`): Procesa el body de la issue buscando imagenes (mockups, screenshots) y las descarga como base64.

Retorna un `AgentContext` completo con toda la informacion.

##### Clase `GitHubHelper`

Cliente simple de GitHub que usan los agentes (distinto al `GitHubClient` del orquestador):

- `get_issue(n)` - Obtiene datos de la issue
- `get_issue_comments(n)` - Obtiene comentarios
- `add_comment(n, body)` - Anade comentario
- `update_labels(n, add, remove)` - Actualiza labels
- `create_issue(title, body, labels)` - Crea issue
- `close_issue(n)` - Cierra issue

Usa `requests.Session` con token de autenticacion.

##### Funciones standalone

- `load_markdown_file(path, max_size)` - Lee archivo markdown con limite de tamano
- `parse_memory_file(content)` - Parsea un archivo de memoria en secciones
- `find_feature_memory_file(memory_path, issue_number)` - Busca el archivo de feature memory por numero de issue
- `extract_acceptance_criteria(body)` - Extrae criterios de aceptacion del markdown

#### 4.1.4 Output Handler (`output_handler.py`)

Escribe resultados de los agentes al volumen de output.

##### Constantes de Schema

Cada tipo de agente tiene un schema JSON que define la estructura esperada de su output:

- `PLANNER_OUTPUT_SCHEMA`: Espera `tasks` (array), `summary`, `dependencies`
- `DEVELOPER_OUTPUT_SCHEMA`: Espera `modified_files`, `commit_message`, `implementation_notes`
- `QA_OUTPUT_SCHEMA`: Espera `qa_result`, `test_results`, `acceptance_checklist`, `feedback`
- `REVIEWER_OUTPUT_SCHEMA`: Espera `review_result`, `comments`, `quality_score`
- `DOC_OUTPUT_SCHEMA`: Espera `updated_files`, `changelog_entry`

##### Funcion `validate_output(output, agent_type) -> bool`

Valida que el output del agente cumpla con su schema correspondiente.

##### Clase `OutputHandler`

- `write_result(result)` - Escribe `result.json` al directorio de output
- `write_artifact(name, content)` - Escribe un artefacto (archivo auxiliar)
- `write_metrics(metrics)` - Escribe metricas a `metrics.json`
- `write_log(message, level)` - Anade entrada al log
- `read_previous_result()` - Lee resultado previo (para reintentos)

Usa escritura atomica (`.tmp` + rename) para evitar archivos corruptos.

##### Clase `ResultFormatter`

Formatea resultados para diferentes contextos:
- `to_json(result)` - Formato JSON
- `to_markdown(result, agent_type)` - Formato Markdown (para comentarios de GitHub)
- `to_text(result)` - Formato texto plano

#### 4.1.5 Image Utils (`image_utils.py`)

Procesa imagenes embebidas en issues de GitHub (mockups, screenshots, diagramas).

##### Dataclass `ImageContent`

Imagen procesada lista para enviar al LLM:
- `url: str` - URL original
- `alt_text: str` - Texto alternativo
- `media_type: str` - MIME type (image/png, image/jpeg, etc.)
- `base64_data: str` - Datos codificados en base64
- `source: str` - Origen ("issue_body", "comment")
- `size_bytes: int` - Tamano original

##### Funcion `extract_image_urls(markdown_text) -> List[Dict]`

Extrae URLs de imagenes de texto markdown usando tres patrones regex:

1. **Sintaxis markdown**: `![alt text](url)` -> Extrae alt_text y url
2. **URLs sueltas**: URLs que terminan en `.png`, `.jpg`, `.gif`, `.webp`
3. **GitHub user-content**: URLs de `user-images.githubusercontent.com` (pueden no tener extension)

Retorna lista deduplicada de `{"url": "...", "alt_text": "..."}`.

##### Funcion `download_image(url, session, timeout, max_size_bytes) -> Tuple[bytes, str]`

Descarga una imagen:
1. Hace GET con streaming
2. Detecta media type del header `Content-Type` o infiere de la extension
3. Lee chunks verificando que no exceda `max_size_bytes` (5MB por defecto)
4. Retorna (bytes crudos, media type)

##### Funcion `process_issue_images(markdown_text, session, max_images, max_size_bytes) -> List[ImageContent]`

Pipeline completa:
1. Extrae URLs con `extract_image_urls()`
2. Limita a `max_images` (5 por defecto)
3. Para cada imagen: descarga -> base64 encode -> crea `ImageContent`
4. Errores individuales se loguean y se saltan (no bloquean)
5. Retorna lista de imagenes procesadas

---

### 4.2 Planner Agent

**Archivo**: `agents/planner/planner_agent.py`
**Clase**: `PlannerAgent(AgentInterface)`

El Planner descompone features complejas en tareas mas pequenas.

**`get_agent_type()`**: Retorna `"planner"`.

**`validate_context(context)`**: Verifica que exista el body de la issue y al menos un archivo de memoria.

**`execute(context)`**:

1. **Extraer info de la feature** (`_extract_feature_info`):
   - Parsea titulo, descripcion, criterios de aceptacion, restricciones

2. **Preparar prompt** (`_prepare_prompt`):
   - Construye un prompt con la feature info + contexto del proyecto + arquitectura + convenciones
   - Incluye instrucciones para producir JSON con la descomposicion

3. **Obtener descomposicion** (`_get_decomposition`):
   - Llama al LLM con el prompt
   - Si hay imagenes en el contexto, las incluye en la llamada (multimodal)
   - Parsea la respuesta JSON

4. **Validar** (`_validate_decomposition`):
   - Verifica que haya al menos 1 tarea
   - Verifica que no excedan `max_sub_issues` (10)
   - Valida dependencias entre tareas

5. **Crear issues** (`_create_issues`):
   - Para cada tarea, crea una issue en GitHub
   - Anade labels `subtask`, `READY`, y la prioridad
   - Vincula al padre con "Subtask of #N"

6. **Comentar en el padre** (`_add_summary_comment`):
   - Lista las subtareas creadas con links

7. **Crear feature memory** (`_create_feature_memory`):
   - Genera archivo markdown en `memory/features/{slug}.md`
   - Incluye metadata, criterios, subtareas, enfoque tecnico

**Retorna**: `AgentResult` con `output = {tasks, summary, feature_memory_file}`.

---

### 4.3 Developer Agent

**Archivo**: `agents/developer/developer_agent.py`
**Clase**: `DeveloperAgent(AgentInterface)`

El Developer implementa codigo basandose en los requisitos de una issue.

**`execute(context)`**:

1. **Parsear requisitos** (`_parse_task_requirements -> TaskInfo`):
   - Extrae titulo, descripcion, criterios de aceptacion, hints de implementacion
   - Identifica archivos mencionados en el body
   - Detecta si es subtarea de una feature

2. **Cargar QA feedback** (`_load_qa_feedback`):
   - Si es un reintento, lee el feedback previo del QA

3. **Encontrar codigo relevante** (`_find_relevant_code`):
   - Lista archivos del repositorio (`_get_repository_files`)
   - Pide al LLM que identifique los archivos relevantes (`_format_file_identification_prompt`)
   - Lee el contenido de esos archivos

4. **Crear plan de implementacion** (`_create_implementation_plan`):
   - Construye prompt con requisitos + codigo existente + contexto + QA feedback
   - Si hay imagenes (mockups), las incluye en la llamada al LLM
   - El LLM retorna un plan con: enfoque, archivos a modificar/crear, tests, riesgos
   - System prompt incluye instrucciones para analizar imagenes si `has_images=True`

5. **Generar cambios de codigo** (`_generate_code_changes`):
   - Para cada archivo del plan, genera el contenido completo
   - Usa prompts especificos por archivo con contexto del contenido original
   - Parsea el codigo de la respuesta del LLM

6. **Aplicar cambios** (`_apply_changes`):
   - Escribe los archivos al sistema de archivos del repo montado

7. **Verificar** (`_verify_changes`):
   - Ejecuta linter (ruff) si esta configurado
   - Si hay errores, intenta corregirlos automaticamente (`_attempt_fix_errors`)

8. **Operaciones Git** (`_handle_git_operations`):
   - Crea branch `agent/{issue_number}`
   - Hace commit con mensaje generado

**Retorna**: `AgentResult` con `output = {modified_files, commit_message, implementation_notes, tests_added, branch}`.

##### Dataclasses auxiliares

- **`TaskInfo`**: Requisitos parseados (titulo, descripcion, criterios, hints, archivos mencionados)
- **`FileChange`**: Un cambio de archivo (path, action create/modify, content, original_content, explanation)
- **`ImplementationPlan`**: Plan del LLM (approach, files, tests, risks)

---

### 4.4 QA Agent

**Archivo**: `agents/qa/qa_agent.py`
**Clase**: `QAAgent(AgentInterface)`

El QA valida que la implementacion cumple los criterios de aceptacion.

**`execute(context)`**:

1. **Extraer criterios** (`_extract_criteria -> List[AcceptanceCriterion]`):
   - Parsea el body buscando checkboxes
   - Categoriza cada criterio (functional, integration, error_handling, etc.)

2. **Ejecutar tests** (`_run_tests -> List[TestResult]`):
   - Obtiene comandos de test del contexto o usa defaults (`pytest`, `ruff check`)
   - Ejecuta cada comando con `subprocess` y captura stdout/stderr/exit_code
   - Mide duracion

3. **Cargar archivos modificados** (`_load_file_contents`):
   - Lee los archivos que el developer modifico

4. **Verificar cada criterio** (`_verify_criterion -> CriterionVerification`):
   - Para cada criterio de aceptacion, pide al LLM que analice el codigo y determine si se cumple
   - El LLM recibe: el criterio, los archivos relevantes, resultados de tests
   - Retorna: PASS, FAIL, o UNCLEAR con evidencia

5. **Generar feedback** (`_generate_feedback`):
   - Resume los resultados para el developer

6. **Generar sugerencias de fix** (`_generate_fix_suggestions -> List[FixSuggestion]`):
   - Si fallo, pide al LLM sugerencias especificas de como arreglar cada problema

7. **Generar resumen** (`_generate_summary`):
   - Resumen ejecutivo del QA

**Veredicto final**: PASS si todos los criterios pasan y todos los tests pasan. FAIL en caso contrario.

**Retorna**: `AgentResult` con `output = {qa_result, criteria_results, test_results, feedback, suggestions, summary}`.

##### Dataclasses auxiliares

- **`AcceptanceCriterion`**: Criterio individual (id, text, category)
- **`CriterionVerification`**: Resultado de verificar un criterio (result, evidence, reason)
- **`TestResult`**: Resultado de un test (command, exit_code, stdout, stderr, passed)
- **`FixSuggestion`**: Sugerencia de arreglo (issue, location, suggestion, priority)
- **`QAReport`**: Reporte completo (verdict, criteria_results, test_results, feedback, suggestions)

---

### 4.5 Reviewer Agent

**Archivo**: `agents/reviewer/reviewer_agent.py`
**Clase**: `ReviewerAgent(AgentInterface)`

El Reviewer revisa la calidad del codigo, adherencia a convenciones y alineacion con la arquitectura.

**`execute(context)`**:

1. **Cargar archivos** (`_load_files`): Lee los archivos modificados, filtrando binarios.

2. **Revisar cada archivo** (`_review_file -> FileReview`):
   - Pide al LLM que revise el archivo contra convenciones y arquitectura
   - El LLM produce comentarios con severidad (CRITICAL, MAJOR, MINOR, SUGGESTION, PRAISE)
   - Puntua en 7 categorias: code_quality, conventions, architecture, maintainability, documentation, security, performance

3. **Calcular puntuaciones** (`_calculate_overall_scores`): Promedio ponderado de todas las reviews.

4. **Calcular score global** (`_calculate_weighted_score`): Aplica pesos por categoria:
   - code_quality: 30%, conventions: 20%, architecture: 20%, maintainability: 15%, documentation: 15%

5. **Identificar bloqueantes** (`_get_blocking_issues`): Comentarios con severidad CRITICAL o MAJOR.

6. **Decidir** (`_determine_decision`):
   - Si score >= threshold (70) y no hay bloqueantes: `APPROVED`
   - Si hay bloqueantes: `CHANGES_REQUESTED`
   - Si score bajo: `CHANGES_REQUESTED`

**Retorna**: `AgentResult` con `output = {review_result, file_reviews, overall_score, scores_breakdown, summary, blocking_issues, improvement_areas}`.

##### Dataclasses auxiliares

- **`ReviewComment`**: Comentario de review (file, line, category, severity, message, suggestion)
- **`FileReview`**: Review de un archivo (file_path, comments, scores, summary)
- **`ReviewReport`**: Reporte completo (decision, file_reviews, overall_score, summary)

##### Enums

- **`ReviewDecision`**: APPROVED, CHANGES_REQUESTED, NEEDS_DISCUSSION
- **`CommentSeverity`**: CRITICAL, MAJOR, MINOR, SUGGESTION, PRAISE
- **`ReviewCategory`**: CODE_QUALITY, CONVENTIONS, ARCHITECTURE, MAINTAINABILITY, DOCUMENTATION, SECURITY, PERFORMANCE

---

### 4.6 Doc Agent

**Archivo**: `agents/doc/doc_agent.py`
**Clase**: `DocAgent(AgentInterface)`

El Doc actualiza la documentacion y memoria del proyecto despues de una implementacion.

**`execute(context)`**:

1. **Recopilar detalles** (`_gather_implementation_details`): Lee el output del developer (archivos modificados, commit, notas) y del QA (feedback).

2. **Generar resumen** (`_generate_implementation_summary`): Pide al LLM que resuma la implementacion.

3. **Actualizar feature memory** (`_update_feature_memory`):
   - Busca el archivo de feature memory existente
   - Si existe, lo parsea y actualiza con los detalles nuevos
   - Anade entrada al historial
   - Actualiza estado, archivos modificados, resumen de implementacion

4. **Generar changelog** (`_generate_changelog_entry`):
   - Crea una entrada de changelog con: categoria (added/changed/fixed/etc.), resumen, detalles

5. **Escribir documentacion** (`_write_documentation`): Escribe los archivos actualizados al disco.

**Retorna**: `AgentResult` con `output = {updated_files, changelog_entry, documentation_updates, summary}`.

##### Dataclasses auxiliares

- **`HistoryEntry`**: Entrada de historial (timestamp, entry_type, agent, action, result, details)
- **`FeatureMemory`**: Representacion en memoria de un feature memory file (con metodo `to_markdown()`)
- **`ChangelogEntry`**: Entrada de changelog (version, date, issue_number, title, category, summary)
- **`DocumentationUpdate`**: Actualizacion a escribir (doc_type, file_path, content, action)

##### Enums

- **`DocumentationType`**: FEATURE_MEMORY, CHANGELOG, ARCHITECTURE, CONVENTIONS, PROJECT
- **`HistoryEntryType`**: CREATED, PLANNING, DEVELOPMENT, QA_PASSED, QA_FAILED, REVIEW, COMPLETED, BLOCKED

---

## 5. Monitorizacion

### 5.1 Logger (`monitoring/logger.py`)

Sistema de logging estructurado con soporte para audit trail.

#### Funcion `setup_logging(level, fmt, log_file, log_dir, mask_sensitive, max_bytes, backup_count)`

Configura el logging del sistema:
- Si `fmt="json"`: Usa `JSONFormatter` para logs estructurados
- Si `fmt="text"`: Usa formato estandar
- Si `structlog` esta disponible: Configura structlog con procesadores
- Si `mask_sensitive=True`: Filtra tokens, API keys, passwords de los logs
- Soporta rotacion de archivos (`RotatingFileHandler`)

#### Funcion `mask_sensitive_data(logger, method_name, event_dict)`

Procesador de structlog que enmascara datos sensibles. Busca claves que contengan: `token`, `key`, `secret`, `password`, `credential`, `authorization`, `api_key`.

Resultado: `"ghp_abc123def456"` -> `"ghp_***456"`

#### Funcion `mask_dict(data)`

Version recursiva para diccionarios: recorre todos los niveles y enmascara valores de claves sensibles.

#### Clase `JSONFormatter`

Formatter de logging que produce JSON:
```json
{"timestamp": "2024-01-01T00:00:00Z", "level": "INFO", "logger": "orchestrator", "message": "Processing issue", "issue_number": 42}
```

#### Clase `LogContext` / Funcion `log_context(**kwargs)`

Context manager que inyecta campos extra en todos los logs dentro de su scope:

```python
with log_context(issue_number=42, agent="developer"):
    logger.info("Processing")  # Incluye issue_number=42, agent="developer"
```

#### Clase `AuditLogger`

Escribe eventos de auditoria a un archivo JSONL (una linea JSON por evento).

Metodos de registro:

| Metodo | Que registra |
|--------|-------------|
| `log_state_transition(issue, from, to, trigger, agent)` | Transiciones de estado |
| `log_agent_execution(agent_type, issue, result, duration)` | Ejecucion de agentes |
| `log_code_change(issue, files, agent, commit)` | Cambios de codigo |
| `log_github_api(endpoint, method, status, issue)` | Llamadas a GitHub API |
| `log_decision(issue, type, decision, reasoning)` | Decisiones del LLM |
| `log_llm_call(model, agent, tokens_in, tokens_out, duration, cost)` | Llamadas al LLM |
| `log_error(component, error_type, message, issue)` | Errores |

Cada entrada incluye automaticamente `timestamp`, `event_type`, y un `id` unico.

### 5.2 Metrics (`monitoring/metrics.py`)

Sistema de metricas con soporte para Prometheus y fallback en memoria.

#### Clase `MetricsCollector`

La clase principal. Si `prometheus_client` esta instalado, usa metricas reales de Prometheus. Si no, usa clases fallback en memoria.

**Metricas registradas:**

| Metrica | Tipo | Labels | Descripcion |
|---------|------|--------|-------------|
| `issues_processed_total` | Counter | type, result | Issues procesadas |
| `issue_duration_seconds` | Histogram | - | Duracion de procesamiento |
| `issue_iterations` | Histogram | - | Iteraciones por issue |
| `issues_in_progress` | Gauge | state | Issues activas por estado |
| `agent_execution_duration` | Histogram | type | Duracion de agentes |
| `agent_running_count` | Gauge | type | Agentes activos |
| `qa_results_total` | Counter | result | Resultados de QA |
| `llm_calls_total` | Counter | model, agent | Llamadas LLM |
| `llm_tokens_total` | Counter | model, direction | Tokens consumidos |
| `llm_duration_seconds` | Histogram | model | Latencia LLM |
| `llm_cost_dollars` | Counter | model | Coste estimado |
| `github_requests_total` | Counter | endpoint, method, status | Requests a GitHub |
| `github_rate_limit_remaining` | Gauge | - | Rate limit restante |
| `queue_depth` | Gauge | - | Profundidad de cola |
| `errors_total` | Counter | component, type | Errores |
| `state_transitions_total` | Counter | from, to | Transiciones de estado |
| `system_info` | Info | - | Version, proyecto, etc. |

**Metodo `snapshot() -> Dict`**: Exporta todas las metricas como diccionario plano (util para debug o APIs no-Prometheus).

**Metodo `start_http_server(port)`**: Arranca el servidor HTTP de Prometheus en el puerto especificado.

#### Funcion `estimate_cost(model, input_tokens, output_tokens) -> float`

Estima el coste en dolares de una llamada LLM. Pricing incluido:

| Modelo | Input ($/1M tokens) | Output ($/1M tokens) |
|--------|---------------------|----------------------|
| Claude Opus 4 | 15.00 | 75.00 |
| Claude Sonnet 4 | 3.00 | 15.00 |
| Claude Haiku 4 | 0.80 | 4.00 |
| GPT-4o | 5.00 | 15.00 |
| GPT-4o-mini | 0.15 | 0.60 |

#### Clase `AlertManager`

Sistema de alertas basado en reglas:

- `add_rule(name, check_fn, severity, message)` - Registra una regla
- `add_callback(callback)` - Registra callback para cuando una alerta se dispara
- `check_all()` - Evalua todas las reglas y dispara alertas

Incluye cooldown para evitar alertas repetidas.

#### Clase `AlertRule`

Una regla de alerta: `name`, `check_fn` (funcion que retorna True si la alerta debe dispararse), `severity`, `message`.

#### Clase `HealthCheck`

Agregador de health checks:

- `register(name, check_fn, critical)` - Registra un componente
- `check_all()` - Ejecuta todos los checks y retorna estado agregado
- `check_one(name)` - Verifica un componente especifico

Retorna `{"healthy": True/False, "components": {...}}`.

#### Clases fallback en memoria

Cuando `prometheus_client` no esta disponible, se usan implementaciones en memoria:
- `_InMemoryCounter` - Contador simple con labels
- `_InMemoryGauge` - Gauge con set/inc/dec
- `_InMemoryHistogram` - Histograma con count/sum/min/max
- `_InMemoryInfo` - Informacion de sistema

Estas clases replican la interfaz de Prometheus para que el codigo funcione identicamente.

---

## 6. Infraestructura Docker

### 6.1 Dockerfile del Orquestador

```dockerfile
FROM python:3.11-slim
# Instala: curl (health), git, docker-cli (para lanzar agentes)
# Copia: orchestrator/, config/
# CMD: python -m orchestrator.main
```

El orquestador necesita acceso al Docker socket (`/var/run/docker.sock`) para lanzar contenedores de agentes.

### 6.2 Dockerfile de Agentes

```dockerfile
FROM python:3.11-slim
# Instala: git, build-essential, curl
# Copia: agents/
# Configura git user para commits
# Crea usuario non-root "agent"
# ENTRYPOINT: /entrypoint.sh
```

Todos los agentes usan la **misma imagen**. El `entrypoint.sh` lee `AGENT_TYPE` y ejecuta `python -m agents.{type}.{type}_agent`.

### 6.3 Entrypoint (`entrypoint.sh`)

1. Valida variables requeridas: `AGENT_TYPE`, `PROJECT_ID`, `ISSUE_NUMBER`, `GITHUB_TOKEN`, `GITHUB_REPO`
2. Valida que `AGENT_TYPE` sea uno de: planner, developer, qa, reviewer, doc
3. Crea directorios de output
4. Ejecuta `python -m agents.${AGENT_TYPE}.${AGENT_TYPE}_agent`
5. Captura y retorna el exit code

### 6.4 Docker Compose

Define 7 servicios:

| Servicio | Perfil | Descripcion |
|----------|--------|-------------|
| `orchestrator` | (siempre) | Orquestador principal |
| `agent-runner` | `agent` | Template de agentes (no se arranca solo) |
| `redis` | (siempre) | Backend de estado |
| `postgres` | `full` | Base de datos (opcional) |
| `webhook-tunnel` | `dev` | Tunel ngrok para desarrollo |
| `prometheus` | `monitoring` | Recolector de metricas |
| `grafana` | `monitoring` | Dashboard de metricas |

### 6.5 Volumenes

| Volumen Host | Volumen Container | Quien lo usa |
|-------------|------------------|-------------|
| `./memory` | `/memory` | Orquestador + Agentes |
| `./repo` | `/repo` | Agentes |
| `./output` | `/output` | Agentes (escritura) |
| `./logs` | `/logs` | Orquestador |
| Docker socket | `/var/run/docker.sock` | Orquestador (para lanzar agentes) |

---

## 7. Flujo de Datos Completo

A continuacion se describe el flujo completo desde que se crea una issue hasta que se completa, indicando que componente interviene en cada paso y que datos fluyen entre ellos.

### Ejemplo: Procesamiento de una Task

```
1. HUMANO crea issue en GitHub con labels "task" + "READY"

2. ORQUESTADOR (main loop)
   |-- queue_manager.refresh_queue()
   |   |-- github_client.list_issues(labels=["READY"])  --> [Issue #42]
   |   |-- Crea QueueItem(priority=2, issue_number=42)
   |   |-- backend.push(item)
   |
   |-- _process_queue()
       |-- queue_manager.get_next()  --> Issue #42
       |-- asyncio.create_task(process_issue(42))

3. WORKFLOW ENGINE (run)
   |-- _initialize_state(42)
   |   |-- state_manager.create_initial_state(42, "")
   |   |-- github_client.get_issue(42) --> {title, body, labels}
   |
   |-- TRIAGE NODE
   |   |-- langchain_engine.triage_issue(42, title, body, labels)
   |   |   |-- LLM analiza la issue --> {"issue_type": "task"}
   |   |-- state.issue_type = "task"
   |   |-- triage_router(state) --> "task" --> go to DEVELOPMENT
   |
   |-- DEVELOPMENT NODE
   |   |-- issue_manager.transition_to_in_progress(42, "developer")
   |   |   |-- GitHub: quita READY, pone IN_PROGRESS + agent:developer
   |   |
   |   |-- agent_launcher.launch_agent("developer", 42, context)
   |   |   |-- Docker: crea container "ai-agent" con AGENT_TYPE=developer
   |   |   |-- Monta: /memory, /repo, /output, /input
   |   |
   |   |-- DEVELOPER AGENT (dentro del contenedor)
   |   |   |-- context_loader.load()
   |   |   |   |-- Lee variables de entorno
   |   |   |   |-- Lee /memory/PROJECT.md, ARCHITECTURE.md, etc.
   |   |   |   |-- GitHub API: get_issue(42)
   |   |   |   |-- Descarga imagenes de la issue (si hay mockups)
   |   |   |   |-- Lee /input/input.json
   |   |   |   --> AgentContext completo
   |   |   |
   |   |   |-- _parse_task_requirements(context) --> TaskInfo
   |   |   |-- _find_relevant_code(task_info) --> {archivos relevantes}
   |   |   |-- _create_implementation_plan(task_info, code, context)
   |   |   |   |-- LLM: analiza requisitos + codigo + mockups --> plan
   |   |   |-- _generate_code_changes(plan) --> [FileChange, ...]
   |   |   |   |-- LLM: genera codigo para cada archivo
   |   |   |-- _apply_changes(changes) --> escribe archivos al disco
   |   |   |-- _verify_changes(changes) --> ruff check
   |   |   |-- _handle_git_operations(42, changes) --> commit + branch
   |   |   |-- output_handler.write_result(result)
   |   |   |   |-- Escribe /output/result.json
   |   |   |-- Contenedor termina con exit code 0
   |   |
   |   |-- agent_launcher.wait_for_completion(container_id)
   |   |   |-- Lee /output/result.json --> AgentResult
   |   |-- state.last_agent_output = result.output
   |
   |-- QA NODE
   |   |-- issue_manager.transition_to_qa(42)
   |   |   |-- GitHub: pone label QA
   |   |
   |   |-- agent_launcher.launch_agent("qa", 42, context)
   |   |
   |   |-- QA AGENT (dentro del contenedor)
   |   |   |-- _extract_criteria(body) --> [criterios de aceptacion]
   |   |   |-- _run_tests(commands) --> [TestResult, ...]
   |   |   |   |-- subprocess: pytest, ruff check
   |   |   |-- _verify_criterion(each) --> PASS/FAIL por criterio
   |   |   |   |-- LLM: analiza codigo vs criterio
   |   |   |-- Si falla: _generate_fix_suggestions()
   |   |   |-- Escribe resultado
   |   |
   |   |-- state.qa_result = "PASS" (o "FAIL")
   |   |-- qa_router(state):
   |       |-- Si PASS --> go to REVIEW
   |       |-- Si FAIL y iteraciones < max --> go to QA_FAILED
   |       |-- Si FAIL y max iteraciones --> go to BLOCKED
   |
   |-- [Si QA_FAILED]
   |   |-- qa_failed_node: incrementa iteration_count, vuelve a DEVELOPMENT
   |   |-- (El Developer recibe QA feedback en el reintento)
   |
   |-- REVIEW NODE
   |   |-- issue_manager.transition_to_review(42)
   |   |-- agent_launcher.launch_agent("reviewer", 42, context)
   |   |
   |   |-- REVIEWER AGENT
   |   |   |-- _review_file(cada archivo) --> FileReview con comentarios
   |   |   |-- _calculate_weighted_score() --> score global
   |   |   |-- _determine_decision(score, blocking_issues) --> APPROVED
   |   |
   |   |-- review_router(state):
   |       |-- Si APPROVED --> go to DOCUMENTATION
   |       |-- Si CHANGES_REQUESTED --> go to DEVELOPMENT (reintento)
   |
   |-- DOCUMENTATION NODE
   |   |-- agent_launcher.launch_agent("doc", 42, context)
   |   |
   |   |-- DOC AGENT
   |   |   |-- _update_feature_memory() --> actualiza memory/features/
   |   |   |-- _generate_changelog_entry() --> entrada de changelog
   |   |
   |   --> go to DONE (incluso si falla doc)
   |
   |-- DONE NODE
       |-- issue_manager.transition_to_done(42, summary)
       |   |-- GitHub: pone label DONE, cierra issue
       |   |-- Comenta resumen final
       |-- state_manager.save_state(42, state)  --> Persiste estado final

4. ORQUESTADOR
   |-- queue_manager.mark_complete(42)
   |-- Logs: "Issue #42 completed successfully"
```

### Datos que fluyen entre componentes

```
GitHub Issue
    |
    v
Orchestrator (queue_manager.refresh_queue)
    |  issue_data: {number, title, body, labels}
    v
WorkflowEngine (run)
    |  GraphState: {issue_number, issue_state, ...}
    v
Agent Launcher (launch_agent)
    |  Env vars: AGENT_TYPE, ISSUE_NUMBER, tokens...
    |  Volumes: /memory, /repo, /input, /output
    |  /input/input.json: {previous_output, qa_feedback, ...}
    v
Agent Container
    |  ContextLoader.load() --> AgentContext
    |  AgentInterface.execute(context) --> AgentResult
    |  OutputHandler.write_result() --> /output/result.json
    v
Agent Launcher (wait_for_completion)
    |  AgentResult: {status, output, logs, duration}
    v
WorkflowEngine (node handler)
    |  Actualiza GraphState con output del agente
    |  Router decide siguiente nodo
    v
StateManager (save_state)
    |  Persiste WorkflowState a File/Redis/PostgreSQL
    v
IssueManager (transition, comment)
    |  Actualiza labels y comenta en GitHub
    v
GitHub Issue (actualizada con labels y comentarios)
```