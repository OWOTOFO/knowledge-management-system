"""Generate synthetic (no real data, no PII) sample content for the demo.

* 2 PDFs + 2 Word documents of internal policy-style text  -> unstructured store
* a small SQLite database of products, branches, loan volumes and complaints -> structured store
"""

from __future__ import annotations

import random
import sqlite3
from datetime import date, timedelta
from pathlib import Path

# ---------------------------------------------------------------------------
# Unstructured content.  Each document = list of (heading, [paragraphs]).
# ---------------------------------------------------------------------------

DISPUTE_POLICY = (
    "Credit Card Dispute Policy",
    "Policy CCD-2025-03 | Owner: Card Operations | Effective 1 January 2025",
    [
        ("1. Purpose and scope", [
            "This policy defines how the bank handles disputes raised by cardholders on credit card "
            "transactions, including unauthorised transactions, billing errors and merchant disputes.",
            "It applies to all consumer and small business credit card products issued by the bank.",
        ]),
        ("2. Filing a dispute", [
            "Cardholders must raise a dispute within 60 days of the statement date on which the "
            "transaction first appeared.",
            "Disputes can be raised through the mobile app, online banking, by phone or in any branch. "
            "Branch staff must log the dispute in the Card Dispute System on the same business day.",
        ]),
        ("3. Service levels (SLA)", [
            "Card Operations must acknowledge every dispute within 2 business days.",
            "Standard disputes must be resolved within 30 calendar days of receipt.",
            "Complex disputes involving a merchant chargeback may take up to 90 calendar days, and the "
            "cardholder must be informed in writing if the 30 day target will be missed.",
        ]),
        ("4. Provisional credit", [
            "For disputed amounts above 50 dollars, a provisional credit is applied to the account within "
            "10 business days while the investigation is ongoing.",
            "If the investigation finds the transaction was valid, the provisional credit is reversed with "
            "5 business days written notice to the cardholder.",
        ]),
        ("5. Handling procedure", [
            "Step one: verify the cardholder identity and block the card if fraud is suspected.",
            "Step two: record the dispute reason code and supporting evidence in the Card Dispute System.",
            "Step three: request transaction details from the card network and the merchant.",
            "Step four: decide the outcome, notify the cardholder and close the case.",
        ]),
    ],
)

MORTGAGE_GUIDE = (
    "Mortgage Product Guide",
    "Retail Lending | Owner: Mortgage Product Team | Version 4.2",
    [
        ("1. Product overview", [
            "The bank offers a 30-Year Fixed Mortgage and a 5/1 ARM Mortgage for owner-occupied homes.",
            "Current interest rates are published daily in the product rate table maintained by Treasury.",
        ]),
        ("2. Eligibility", [
            "Applicants need a minimum credit score of 660.",
            "The maximum loan-to-value ratio is 80 percent without mortgage insurance and 95 percent with "
            "mortgage insurance.",
            "The debt-to-income ratio must not exceed 43 percent.",
        ]),
        ("3. Required documents", [
            "A mortgage application requires government photo ID, two most recent pay slips, two years of "
            "tax returns, three months of bank statements and a signed purchase agreement.",
        ]),
        ("4. Early repayment", [
            "An early repayment charge of 2 percent of the outstanding balance applies during the first 3 "
            "years of a 30-Year Fixed Mortgage.",
            "The 5/1 ARM Mortgage has no early repayment charge.",
            "Borrowers may overpay up to 10 percent of the balance each year without any charge.",
        ]),
    ],
)

REMOTE_WORK_STANDARD = (
    "Remote Work IT Security Standard",
    "Standard ITS-07 | Owner: Information Security Office",
    [
        ("Network access", [
            "Employees working remotely must connect through the corporate VPN before accessing any "
            "internal system.",
            "Multi-factor authentication is mandatory for VPN, email and all customer data systems.",
            "Public Wi-Fi may only be used together with the VPN.",
        ]),
        ("Devices", [
            "Only bank-issued laptops with full-disk encryption may be used to access customer data.",
            "Personal email accounts and personal cloud storage must never be used for bank information.",
        ]),
        ("Lost or stolen devices", [
            "A lost or stolen laptop or phone must be reported to the IT Service Desk within 24 hours.",
            "The IT Service Desk will remotely lock and wipe the device and the employee must also "
            "inform their line manager.",
        ]),
    ],
)

EXPENSE_POLICY = (
    "Expense Reimbursement Policy",
    "Policy FIN-12 | Owner: Group Finance",
    [
        ("Submitting expenses", [
            "Expense reports must be submitted in the expense system within 30 days of the expense date.",
            "Itemised receipts are required for every expense above 25 dollars.",
            "Reports submitted after 90 days will not be reimbursed.",
        ]),
        ("Approvals", [
            "All expense reports require line manager approval.",
            "Any single expense above 5000 dollars also requires approval from a Vice President.",
        ]),
    ],
)

EXPENSE_LIMITS_TABLE = [
    ("Category", "Limit"),
    ("Hotel (per night)", "250 dollars"),
    ("Meals (per day)", "75 dollars"),
    ("Economy airfare", "Required for flights under 6 hours"),
]


def _write_pdf(path: Path, doc: tuple) -> None:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer

    title, subtitle, sections = doc
    styles = getSampleStyleSheet()
    story = [Paragraph(title, styles["Title"]), Paragraph(subtitle, styles["Italic"]), Spacer(1, 12)]
    for i, (heading, paras) in enumerate(sections):
        if i and i % 2 == 0:  # force multiple pages so page metadata is meaningful
            story.append(PageBreak())
        story.append(Paragraph(heading, styles["Heading2"]))
        story.extend(Paragraph(p, styles["BodyText"]) for p in paras)
    SimpleDocTemplate(str(path), pagesize=A4, title=title).build(story)


def _write_docx(path: Path, doc: tuple, table: list[tuple] | None = None) -> None:
    import docx

    title, subtitle, sections = doc
    d = docx.Document()
    d.add_heading(title, level=0)
    d.add_paragraph(subtitle)
    for heading, paras in sections:
        d.add_heading(heading, level=1)
        for p in paras:
            d.add_paragraph(p)
    if table:
        d.add_heading("Expense limits", level=1)
        t = d.add_table(rows=0, cols=len(table[0]))
        for row in table:
            cells = t.add_row().cells
            for c, v in zip(cells, row):
                c.text = v
    d.save(str(path))


# ---------------------------------------------------------------------------
# Structured content
# ---------------------------------------------------------------------------

SCHEMA = """
CREATE TABLE products (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    type TEXT NOT NULL,              -- credit_card | mortgage | deposit | loan
    interest_rate REAL NOT NULL,     -- annual %, APR for cards and loans, APY for deposits
    min_balance REAL NOT NULL,       -- dollars
    annual_fee REAL NOT NULL         -- dollars
);
CREATE TABLE branches (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    region TEXT NOT NULL,            -- North | South | East | West
    city TEXT NOT NULL,
    opened_date TEXT NOT NULL
);
CREATE TABLE monthly_loan_volume (
    branch_id INTEGER NOT NULL REFERENCES branches(id),
    month TEXT NOT NULL,             -- YYYY-MM
    product_id INTEGER NOT NULL REFERENCES products(id),
    num_loans INTEGER NOT NULL,
    amount REAL NOT NULL,            -- dollars originated
    PRIMARY KEY (branch_id, month, product_id)
);
CREATE TABLE complaints (
    id INTEGER PRIMARY KEY,
    product_id INTEGER NOT NULL REFERENCES products(id),
    category TEXT NOT NULL,
    opened_date TEXT NOT NULL,
    resolved_date TEXT,
    days_to_resolve INTEGER
);
"""

PRODUCTS = [
    (1, "Platinum Rewards Card", "credit_card", 21.99, 0, 95),
    (2, "Everyday Cashback Card", "credit_card", 24.49, 0, 0),
    (3, "30-Year Fixed Mortgage", "mortgage", 6.25, 0, 0),
    (4, "5/1 ARM Mortgage", "mortgage", 5.65, 0, 0),
    (5, "Premier Checking", "deposit", 0.05, 1500, 0),
    (6, "High-Yield Savings", "deposit", 4.10, 500, 0),
    (7, "Small Business Loan", "loan", 8.75, 0, 250),
]

BRANCHES = [
    (1, "Riverside", "North", "Lakeview", "2008-04-01"),
    (2, "Old Town", "North", "Hillcrest", "1996-09-15"),
    (3, "Harbour Point", "South", "Port Ellis", "2012-02-20"),
    (4, "Market Square", "South", "Granton", "2001-06-11"),
    (5, "Sunrise Plaza", "East", "Easton", "2015-03-03"),
    (6, "Tech Park", "East", "Newbridge", "2019-10-28"),
    (7, "Canyon Road", "West", "Redmesa", "2005-01-17"),
    (8, "Pacific Gate", "West", "Seabrook", "2010-07-07"),
]

COMPLAINT_CATEGORIES = ["Billing dispute", "Fraud", "Fees", "Customer service", "Application delay"]


def _build_db(path: Path, seed: int = 42) -> None:
    rng = random.Random(seed)
    if path.exists():
        path.unlink()
    con = sqlite3.connect(path)
    con.executescript(SCHEMA)
    con.executemany("INSERT INTO products VALUES (?,?,?,?,?,?)", PRODUCTS)
    con.executemany("INSERT INTO branches VALUES (?,?,?,?,?)", BRANCHES)

    avg_size = {3: 380_000, 4: 420_000, 7: 85_000}
    rows = []
    for b in BRANCHES:
        for m in range(1, 13):
            for pid, size in avg_size.items():
                n = rng.randint(2, 25)
                rows.append((b[0], f"2025-{m:02d}", pid, n, round(n * size * rng.uniform(0.8, 1.2), 2)))
    con.executemany("INSERT INTO monthly_loan_volume VALUES (?,?,?,?,?)", rows)

    complaints = []
    for cid in range(1, 301):
        pid = rng.choices([p[0] for p in PRODUCTS], weights=[30, 25, 10, 8, 12, 8, 7])[0]
        cat = rng.choice(COMPLAINT_CATEGORIES)
        opened = date(2025, 1, 1) + timedelta(days=rng.randint(0, 330))
        days = int(rng.triangular(2, 60, 18)) if pid in (1, 2) else int(rng.triangular(1, 45, 10))
        complaints.append((cid, pid, cat, opened.isoformat(), (opened + timedelta(days=days)).isoformat(), days))
    con.executemany("INSERT INTO complaints VALUES (?,?,?,?,?,?)", complaints)
    con.commit()
    con.close()


def generate(docs_dir: Path, db_path: Path) -> dict:
    docs_dir.mkdir(parents=True, exist_ok=True)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    _write_pdf(docs_dir / "Credit_Card_Dispute_Policy.pdf", DISPUTE_POLICY)
    _write_pdf(docs_dir / "Mortgage_Product_Guide.pdf", MORTGAGE_GUIDE)
    _write_docx(docs_dir / "Remote_Work_IT_Security_Standard.docx", REMOTE_WORK_STANDARD)
    _write_docx(docs_dir / "Expense_Reimbursement_Policy.docx", EXPENSE_POLICY, EXPENSE_LIMITS_TABLE)
    _build_db(db_path)
    return {"docs": sorted(p.name for p in docs_dir.iterdir()), "db": str(db_path)}
