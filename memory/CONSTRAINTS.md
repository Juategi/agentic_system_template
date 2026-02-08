# Technical Constraints

## Technology Constraints

### Required Technologies
- HTML5: Único lenguaje permitido

### Prohibited Technologies
- JavaScript: No se usa en este proyecto
- CSS externo: No se usa (inline si es estrictamente necesario)
- Frameworks: Ninguno (no Bootstrap, no Tailwind, etc.)
- Build tools: Ninguno (no webpack, no vite, etc.)

## Performance Constraints

### Page Load
- La página debe cargar sin dependencias externas
- Imágenes optimizadas (formatos web: jpg, png, webp)
- Sin requests a CDNs externos

## Compatibility Constraints

### Browser Support
| Browser | Minimum Version |
|---------|-----------------|
| Chrome | Últimas 2 versiones |
| Firefox | Últimas 2 versiones |
| Safari | Últimas 2 versiones |
| Edge | Últimas 2 versiones |

### Device Support
- Desktop: Sí
- Mobile: Sí (meta viewport obligatorio)
- Tablet: Sí

## Development Constraints

### Code Constraints
- Un solo archivo HTML principal (index.html)
- Imágenes en carpeta /images
- HTML válido según W3C

### Deployment
- Despliegue en cualquier hosting estático
- Sin proceso de build necesario
- Copiar archivos directamente

## Agent-Specific Constraints

### What Agents CAN Do
- Crear y modificar index.html
- Añadir imágenes a /images
- Modificar memoria en /memory

### What Agents CANNOT Do
- Añadir JavaScript
- Añadir CSS externo
- Instalar dependencias
- Crear múltiples páginas HTML

---

*Last updated: 2026-02-08*
*Review these constraints before implementing any feature*
