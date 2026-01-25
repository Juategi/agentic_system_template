# Security-Focused Code Review

Perform a security-focused review of the following code.

## File Information

**Path:** {{file_path}}
**Language:** {{language}}

## File Content

```
{{file_content}}
```

## Security Review Checklist

### Input Validation
- [ ] All user inputs are validated
- [ ] Input lengths are checked
- [ ] Input types are verified
- [ ] Special characters are handled

### Injection Prevention
- [ ] SQL queries use parameterized statements
- [ ] No string concatenation for queries
- [ ] Command injection is prevented
- [ ] XSS vulnerabilities are mitigated

### Authentication & Authorization
- [ ] Authentication is properly implemented
- [ ] Authorization checks are in place
- [ ] Session management is secure
- [ ] Passwords are properly hashed

### Data Protection
- [ ] Sensitive data is encrypted
- [ ] Secrets are not hardcoded
- [ ] PII is handled appropriately
- [ ] Data is sanitized before logging

### Error Handling
- [ ] Errors don't leak sensitive info
- [ ] Stack traces are not exposed
- [ ] Error messages are generic for users

### Dependencies
- [ ] No known vulnerable dependencies
- [ ] Dependencies are from trusted sources
- [ ] Minimum necessary permissions

## Output Format

```json
{
  "security_score": 85,
  "vulnerabilities": [
    {
      "severity": "critical" | "high" | "medium" | "low",
      "category": "injection" | "auth" | "data_exposure" | "config" | "other",
      "line": 42,
      "description": "SQL query uses string concatenation",
      "impact": "Could allow SQL injection attacks",
      "recommendation": "Use parameterized queries instead",
      "cwe": "CWE-89"
    }
  ],
  "good_practices": [
    "Proper password hashing with bcrypt",
    "Input validation on all endpoints"
  ],
  "recommendations": [
    "Add rate limiting to authentication endpoints",
    "Implement CSRF protection"
  ],
  "summary": "Overall security assessment"
}
```

## Severity Definitions

- **critical**: Immediate exploitation risk, data breach possible
- **high**: Significant vulnerability, requires prompt attention
- **medium**: Moderate risk, should be addressed soon
- **low**: Minor issue, best practice improvement

---

Now perform the security review.
