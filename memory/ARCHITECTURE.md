# Architecture Memory

## System Architecture Overview

Proyecto estático de una sola página HTML. Sin backend, sin base de datos, sin APIs.

## High-Level Architecture

```
┌─────────────────────────────────┐
│          NAVEGADOR              │
│                                 │
│   index.html (página única)     │
│   ├── Header / Hero             │
│   ├── Servicios                 │
│   ├── Precios                   │
│   ├── Horarios                  │
│   ├── Ubicación / Contacto      │
│   └── Footer                    │
│                                 │
└─────────────────────────────────┘
```

## Directory Structure

```
project/
├── index.html              # Página principal (único archivo)
├── images/                 # Imágenes del sitio
│   ├── logo.png
│   ├── hero.jpg
│   └── ...
├── memory/                 # Memoria del proyecto (agentes)
├── config/                 # Configuración del sistema de agentes
└── CLAUDE.md               # Instrucciones AI
```

## Secciones de la Página

### Header / Navegación
- Logo de la peluquería
- Nombre del negocio
- Navegación interna con anclas (#servicios, #precios, etc.)

### Hero
- Imagen principal o banner
- Título llamativo
- Breve descripción del negocio

### Servicios
- Lista de servicios ofrecidos (corte, tinte, peinado, etc.)
- Descripción breve de cada servicio

### Precios
- Tabla de precios por servicio

### Horarios y Ubicación
- Días y horas de apertura
- Dirección física
- Mapa o indicaciones

### Contacto
- Teléfono
- Email
- Redes sociales (enlaces)

### Footer
- Copyright
- Links rápidos

## Architectural Decisions Record (ADR)

### ADR-001: HTML puro sin CSS/JS externo
- **Date**: 2026-02-08
- **Status**: Accepted
- **Context**: Se necesita la landing más simple posible
- **Decision**: Todo en un solo archivo HTML con estilos inline si es necesario
- **Consequences**: Máxima portabilidad, sin dependencias, fácil de desplegar

---

*Last updated: 2026-02-08*
*Maintained by: Agents and Human Architects*
