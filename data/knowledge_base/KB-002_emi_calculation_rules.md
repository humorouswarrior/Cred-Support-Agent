---
doc_id: KB-002
topic: emi_calculation_rules
title: How EMI is calculated
---

Cred computes the equated monthly instalment using the standard reducing-balance formula
EMI = P * r * (1 + r)^n / ((1 + r)^n - 1), where P is the sanctioned principal, r is the
monthly interest rate, and n is the tenure in months. The monthly rate is the annual
rate divided by twelve, and interest always accrues on the outstanding balance rather
than the original principal. When disbursement happens mid-month, the days between
disbursal and the first instalment date are billed separately as pre-EMI interest. The
EMI shown at application time is indicative and is refreshed on the sanction letter once
the final rate is locked.
