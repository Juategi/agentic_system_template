# Changelog Entry Generator

Generate a changelog entry for the completed work.

## Issue Information

**Number:** {{issue_number}}
**Title:** {{issue_title}}

**Description:**
{{issue_body}}

## Implementation Summary

{{implementation_summary}}

## Files Changed

{{files_modified}}

## Your Task

Generate a changelog entry following the Keep a Changelog format.

### Categories

Choose the appropriate category:

- **Added**: New features or capabilities
- **Changed**: Changes to existing functionality
- **Fixed**: Bug fixes
- **Removed**: Removed features or capabilities
- **Security**: Security-related changes
- **Deprecated**: Features marked for future removal

### Guidelines

1. **Be User-Focused**: Write for users, not developers
2. **Be Concise**: One line summary when possible
3. **Be Specific**: Mention what changed, not how
4. **Link Issues**: Reference the issue number

### Examples

**Good:**
```
### Added
- User authentication with email/password login (#123)
```

**Bad:**
```
### Added
- Implemented JWT token validation in auth_controller.py
```

## Output Format

Return a single changelog entry:

```markdown
### [Category]
- [Description of change] (#{{issue_number}})
```

If multiple distinct changes, list each:

```markdown
### Added
- Feature A (#{{issue_number}})
- Feature B (#{{issue_number}})

### Fixed
- Bug that caused X (#{{issue_number}})
```
