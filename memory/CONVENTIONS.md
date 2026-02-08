# Coding Conventions

## General Principles

1. **HTML semántico**: Usar etiquetas con significado (`<header>`, `<nav>`, `<main>`, `<section>`, `<footer>`)
2. **Accesibilidad**: Atributos `alt` en imágenes, estructura de headings correcta (h1 > h2 > h3)
3. **Simplicidad**: Sin frameworks, sin dependencias externas

## HTML

### Estructura Base
```html
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Peluquería - Nombre</title>
</head>
<body>
    <header>...</header>
    <main>
        <section id="servicios">...</section>
        <section id="precios">...</section>
        <section id="contacto">...</section>
    </main>
    <footer>...</footer>
</body>
</html>
```

### Naming
- IDs de secciones: `kebab-case` (ej: `id="nuestros-servicios"`)
- Clases si se usan: `kebab-case`
- Imágenes: `kebab-case.ext` (ej: `hero-banner.jpg`)

### Buenas Prácticas
- Indentación: 2 espacios
- Atributos en orden: `id`, `class`, resto
- Cerrar todas las etiquetas
- Comillas dobles para atributos
- Una sección por bloque de contenido

## Git Conventions

### Commit Messages
```
<type>: <subject>

Types:
- feat: Nueva sección o funcionalidad
- fix: Corrección
- style: Cambios visuales
- docs: Documentación
- chore: Mantenimiento

Example:
feat: add servicios section with pricing table
```

---

*Last updated: 2026-02-08*
*Agents must follow these conventions*
