"""Reusable helpers ported from the QF637 gold-alarm notebooks (NB01-NB09).

Modules:
    data       - load/refresh prices and market variables (NB01, NB02)
    signals    - gold abnormality signal components (NB06)
    alarm      - conditioned gold alarm + dashboard state + event utilities (NB06, NB09)
    var        - historical simulation VaR/ES and Kupiec POF test (NB08)
    riskbook   - Brent-only trade-ledger physical book
    stress     - fixed Brent event shocks and reverse stress
    dashboard  - dashboard metrics assembly and lead-time/false-alarm tables (NB09)
    pipeline   - build_all(): runs the full chain end to end
"""
