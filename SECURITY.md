# Security Policy

## Reporting a vulnerability

Please report security issues **privately** - do not open a public issue. Use GitHub's
private advisories (the **Report a vulnerability** button on the repository's **Security**
tab) or contact the maintainer directly. You will get an acknowledgement, and a fix will
be coordinated before any public disclosure.

## Secrets and safe configuration

- **Never commit secrets.** API keys, the Telegram bot token, and proxy credentials
  belong only in `.env`, which is git-ignored. Use `.env.example` as the template.
- **`GTRADE_SSL_VERIFY=0`** disables TLS certificate verification and exists only for a
  trusted, local TLS-intercepting proxy. Never use it in a public or shared deployment.
- This project is **signals only** - it does not connect to a broker or move real money.
  Do not wire it to live execution without a separate, audited safety and auth layer.

## Scope

Reports about the code in this repository are in scope. Third-party dependencies should
be reported to their respective projects.
