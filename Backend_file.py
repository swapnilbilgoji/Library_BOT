# backend_main.py

from fastapi import FastAPI, HTTPException, Depends, Request, Query
from pydantic import BaseModel
from typing import Optional
from datetime import datetime, date, timedelta

from sqlalchemy import (
    create_engine, Column, Integer, String, Date, DateTime
)
from sqlalchemy.orm import declarative_base, sessionmaker, Session

import os
from fpdf import FPDF
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from twilio.rest import Client
from dotenv import load_dotenv

# ---------------------------
# CONFIG & DB SETUP
# ---------------------------
load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL is not set in environment or .env")

LIBRARIAN_EMAIL = os.getenv("LIBRARIAN_EMAIL", "librarian@example.com")

# Email (Gmail) config
SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "465"))
SMTP_USER = os.getenv("SMTP_USER", "your_email@gmail.com")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "your_app_password")

# Twilio WhatsApp config
TWILIO_SID = os.getenv("TWILIO_SID", "")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN", "")
TWILIO_WHATSAPP_FROM = os.getenv("TWILIO_WHATSAPP_FROM", "whatsapp:+14155238886")

Base = declarative_base()
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

app = FastAPI(title="Super Library Backend")

# ---------------------------
# DB MODELS
# ---------------------------

class Student(Base):
    __tablename__ = "students"
    id = Column(Integer, primary_key=True, index=True)
    usn = Column(String, unique=True, index=True)
    name = Column(String)
    semester = Column(String)
    branch = Column(String)
    phone = Column(String)
    email = Column(String, nullable=True)


class Book(Base):
    __tablename__ = "books"
    id = Column(Integer, primary_key=True, index=True)
    accession_no = Column(String, unique=True, index=True)
    title = Column(String)
    author = Column(String)
    subject = Column(String)
    total_copies = Column(Integer, default=1)
    available_copies = Column(Integer, default=1)
    rack_location = Column(String, nullable=True)


class Attendance(Base):
    __tablename__ = "attendance"
    id = Column(Integer, primary_key=True, index=True)
    student_usn = Column(String, index=True)
    date = Column(Date, default=date.today)
    time_in = Column(DateTime, default=datetime.utcnow)


class Issue(Base):
    __tablename__ = "issues"
    id = Column(Integer, primary_key=True, index=True)
    student_usn = Column(String, index=True)
    book_accession_no = Column(String, index=True)
    issue_date = Column(Date, default=date.today)
    due_date = Column(Date)
    status = Column(String, default="issued")  # issued, renewed, pending_return, returned
    renew_count = Column(Integer, default=0)


class Wishlist(Base):
    __tablename__ = "wishlist"
    id = Column(Integer, primary_key=True, index=True)
    student_usn = Column(String, index=True)
    book_accession_no = Column(String, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)


Base.metadata.create_all(bind=engine)

# ---------------------------
# SCHEMAS
# ---------------------------

class AttendanceIn(BaseModel):
    usn: str

class IssueRequestIn(BaseModel):
    usn: str
    accession_no: str

class RenewIn(BaseModel):
    usn: str
    accession_no: str
    extra_days: int = 30

class ReturnIn(BaseModel):
    usn: str
    accession_no: str

class WishlistIn(BaseModel):
    usn: str
    accession_no: str

class ReturnRequestIn(BaseModel):
    usn: str
    accession_no: str

# ---------------------------
# DEPENDENCY
# ---------------------------

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ---------------------------
# UTIL: EMAIL + WHATSAPP + PDF
# ---------------------------

def send_email(to_email: str, subject: str, body: str, attachment_path: Optional[str] = None):
    msg = MIMEMultipart()
    msg["From"] = SMTP_USER
    msg["To"] = to_email
    msg["Subject"] = subject

    msg.attach(MIMEText(body, "plain"))

    if attachment_path and os.path.exists(attachment_path):
        from email.mime.base import MIMEBase
        from email import encoders
        with open(attachment_path, "rb") as f:
            part = MIMEBase("application", "octet-stream")
            part.set_payload(f.read())
        encoders.encode_base64(part)
        part.add_header("Content-Disposition", f"attachment; filename={os.path.basename(attachment_path)}")
        msg.attach(part)

    try:
        with smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT) as server:
            server.login(SMTP_USER, SMTP_PASSWORD)
            server.send_message(msg)
    except Exception as e:
        print("Email sending failed:", e)


def send_whatsapp_message(to_number: str, message: str):
    """
    to_number should be in format: +91XXXXXXXXXX
    Twilio will send to 'whatsapp:+91XXXXXXXXXX'
    """
    if not TWILIO_SID or not TWILIO_AUTH_TOKEN:
        print("Twilio credentials not set; skipping WhatsApp send.")
        return

    try:
        client = Client(TWILIO_SID, TWILIO_AUTH_TOKEN)
        client.messages.create(
            from_=TWILIO_WHATSAPP_FROM,
            to=f"whatsapp:{to_number}",
            body=message
        )
    except Exception as e:
        print("WhatsApp send failed:", e)


def generate_attendance_pdf(attendance_list, filename: str):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, txt="Daily Library Attendance Report", ln=True)

    for att in attendance_list:
        line = f"{att.date}  |  {att.student_usn}  |  {att.time_in.strftime('%H:%M:%S')}"
        pdf.cell(0, 10, txt=line, ln=True)

    pdf.output(filename)


def generate_issues_pdf(issues_list, filename: str, title: str):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, txt=title, ln=True)

    for iss in issues_list:
        line = f"{iss.issue_date} | {iss.student_usn} | {iss.book_accession_no} | status={iss.status}"
        pdf.cell(0, 10, txt=line, ln=True)

    pdf.output(filename)

# ---------------------------
# BASIC ENDPOINTS
# ---------------------------

@app.get("/")
def root():
    return {"message": "Super Library Backend running"}

# ---------------------------
# ATTENDANCE
# ---------------------------

@app.post("/attendance/mark")
def mark_attendance(data: AttendanceIn, db: Session = Depends(get_db)):
    student = db.query(Student).filter(Student.usn == data.usn).first()
    if not student:
        raise HTTPException(status_code=404, detail="Student not found in database. Please register first.")

    today = date.today()
    existing = (
        db.query(Attendance)
        .filter(Attendance.student_usn == data.usn, Attendance.date == today)
        .first()
    )
    if existing:
        return {"message": "Attendance already marked for today."}

    att = Attendance(student_usn=data.usn, date=today, time_in=datetime.utcnow())
    db.add(att)
    db.commit()
    db.refresh(att)

    return {"message": f"Attendance marked for {data.usn} on {today}."}

# ---------------------------
# BOOK STATUS (for UI availability)
# ---------------------------

@app.get("/books/status/{accession_no}")
def book_status(accession_no: str, db: Session = Depends(get_db)):
    book = db.query(Book).filter(Book.accession_no == accession_no).first()
    if not book:
        raise HTTPException(status_code=404, detail="Book not found in backend.")
    return {
        "accession_no": book.accession_no,
        "title": book.title,
        "available_copies": book.available_copies,
        "total_copies": book.total_copies,
        "rack_location": book.rack_location,
    }

# ---------------------------
# ISSUE BOOK
# ---------------------------

@app.post("/books/issue/request")
def request_issue(data: IssueRequestIn, db: Session = Depends(get_db)):
    student = db.query(Student).filter(Student.usn == data.usn).first()
    if not student:
        raise HTTPException(status_code=404, detail="Student not found.")

    book = db.query(Book).filter(Book.accession_no == data.accession_no).first()
    if not book:
        raise HTTPException(status_code=404, detail="Book not found.")
    if book.available_copies <= 0:
        raise HTTPException(status_code=400, detail="No copies available.")

    issue_date = date.today()
    due_date = issue_date + timedelta(days=30)

    issue = Issue(
        student_usn=data.usn,
        book_accession_no=data.accession_no,
        issue_date=issue_date,
        due_date=due_date,
        status="issued",
        renew_count=0,
    )
    book.available_copies -= 1

    db.add(issue)
    db.add(book)
    db.commit()
    db.refresh(issue)

    # Notify librarian via email
    subject = "New Book Issued"
    body = f"USN: {data.usn} \nBook Accession: {data.accession_no}\nIssue Date: {issue_date}\nDue Date: {due_date}"
    send_email(LIBRARIAN_EMAIL, subject, body)

    # Notify student (if phone)
    if student.phone:
        msg = (
            f"📚 Book issued successfully!\n"
            f"Accession: {data.accession_no}\n"
            f"Due Date: {due_date}"
        )
        send_whatsapp_message(student.phone, msg)

    return {"message": "Book issued successfully and librarian notified."}

# ---------------------------
# RENEW BOOK (with MAX 3 RENEWALS LIMIT)
# ---------------------------

@app.post("/books/renew")
def renew_book(data: RenewIn, db: Session = Depends(get_db)):
    issue = (
        db.query(Issue)
        .filter(
            Issue.student_usn == data.usn,
            Issue.book_accession_no == data.accession_no,
            Issue.status.in_(["issued", "renewed"])
        )
        .first()
    )
    if not issue:
        raise HTTPException(status_code=404, detail="Active issue record not found.")

    # ✅ RENEWAL LIMIT CHECK: max 3 renewals
    if issue.renew_count >= 3:
        # Find student for WhatsApp notification
        student = db.query(Student).filter(Student.usn == data.usn).first()
        if student and student.phone:
            msg = (
                "⚠ Renewal Failed\n"
                "You have reached the maximum renewal limit (3 times) for this book.\n"
                "Please return the book to the library to continue using it."
            )
            send_whatsapp_message(student.phone, msg)

        raise HTTPException(
            status_code=400,
            detail="Maximum renewal limit reached (3). Please return the book."
        )

    # ✅ Allowed renewal: update due date & renew_count
    issue.due_date = issue.due_date + timedelta(days=data.extra_days)
    issue.status = "renewed"
    issue.renew_count += 1
    db.add(issue)
    db.commit()

    subject = "Book Renewal Notification"
    body = (
        f"USN: {data.usn}\nBook Accession: {data.accession_no}\n"
        f"New Due Date: {issue.due_date}\nRenew Count: {issue.renew_count}"
    )
    send_email(LIBRARIAN_EMAIL, subject, body)

    student = db.query(Student).filter(Student.usn == data.usn).first()
    if student and student.phone:
        msg = (
            f"✅ Your book (Accession {data.accession_no}) has been renewed.\n"
            f"New due date: {issue.due_date}\n"
            f"Renewals used: {issue.renew_count} / 3"
        )
        send_whatsapp_message(student.phone, msg)

    return {"message": "Book renewed and librarian notified."}

# ---------------------------
# DIRECT RETURN (optional, for admin/tools)
# ---------------------------

@app.post("/books/return")
def return_book(data: ReturnIn, db: Session = Depends(get_db)):
    issue = (
        db.query(Issue)
        .filter(
            Issue.student_usn == data.usn,
            Issue.book_accession_no == data.accession_no,
            Issue.status.in_(["issued", "renewed", "pending_return"])
        )
        .first()
    )
    if not issue:
        raise HTTPException(status_code=404, detail="Issued book record not found.")

    issue.status = "returned"
    db.add(issue)

    book = db.query(Book).filter(Book.accession_no == data.accession_no).first()
    if book:
        book.available_copies += 1
        db.add(book)

    db.commit()

    # Notify librarian
    subject = "Book Returned"
    body = f"USN: {data.usn}\nBook Accession: {data.accession_no}\nReturn Date: {date.today()}"
    send_email(LIBRARIAN_EMAIL, subject, body)

    # Notify wishlist students
    wishers = db.query(Wishlist).filter(Wishlist.book_accession_no == data.accession_no).all()
    for w in wishers:
        student = db.query(Student).filter(Student.usn == w.student_usn).first()
        if student and student.phone:
            msg = (
                f"🎉 Good news! A book from your wishlist is now available.\n"
                f"Accession No: {data.accession_no}\n"
                f"Visit the library or use the chatbot to issue it."
            )
            send_whatsapp_message(student.phone, msg)

    # Notify returning student
    student = db.query(Student).filter(Student.usn == data.usn).first()
    if student and student.phone:
        msg = (
            f"📚 Your book return has been recorded.\n"
            f"Accession: {data.accession_no}"
        )
        send_whatsapp_message(student.phone, msg)

    return {"message": "Book return recorded, stock updated, and notifications sent."}

# ---------------------------
# RETURN REQUEST (student) + CONFIRM (librarian)
# ---------------------------

@app.post("/books/return/request")
def request_return(data: ReturnRequestIn, request: Request, db: Session = Depends(get_db)):
    """
    Student triggers return request. Status -> pending_return.
    Email with approval link sent to librarian.
    """
    issue = (
        db.query(Issue)
        .filter(
            Issue.student_usn == data.usn,
            Issue.book_accession_no == data.accession_no,
            Issue.status.in_(["issued", "renewed"])
        )
        .first()
    )
    if not issue:
        raise HTTPException(status_code=404, detail="Active issue record not found for this book & student.")

    issue.status = "pending_return"
    db.add(issue)
    db.commit()
    db.refresh(issue)

    host = request.headers.get("host", "localhost:8000")
    scheme = request.url.scheme or "http"
    base_url = f"{scheme}://{host}"
    approval_link = f"{base_url}/books/return/confirm?issue_id={issue.id}"

    subject = "Book Return Request - Approval Needed"
    body = (
        f"Student USN: {data.usn}\n"
        f"Book Accession: {data.accession_no}\n"
        f"Issue Date: {issue.issue_date}\n"
        f"Due Date: {issue.due_date}\n\n"
        f"To confirm this return, click the link below:\n{approval_link}"
    )
    send_email(LIBRARIAN_EMAIL, subject, body)

    return {"message": "Return request sent to librarian for approval."}


@app.get("/books/return/confirm")
def confirm_return(issue_id: int = Query(...), db: Session = Depends(get_db)):
    """
    Librarian clicks this link from email to confirm return.
    """
    issue = db.query(Issue).filter(Issue.id == issue_id).first()
    if not issue:
        raise HTTPException(status_code=404, detail="Issue record not found.")

    if issue.status == "returned":
        return {"message": "This book is already marked as returned."}

    if issue.status not in ["pending_return", "issued", "renewed"]:
        raise HTTPException(status_code=400, detail=f"Cannot return book with status '{issue.status}'.")

    issue.status = "returned"

    # Update book copies
    book = db.query(Book).filter(Book.accession_no == issue.book_accession_no).first()
    if book:
        book.available_copies += 1
        db.add(book)

    db.add(issue)
    db.commit()

    # Notify wishlist students
    wishers = db.query(Wishlist).filter(Wishlist.book_accession_no == issue.book_accession_no).all()
    for w in wishers:
        st = db.query(Student).filter(Student.usn == w.student_usn).first()
        if st and st.phone:
            msg = (
                f"🎉 Good news! A book from your wishlist is now available.\n"
                f"Accession No: {issue.book_accession_no}\n"
                f"Visit the library or use the chatbot to issue it."
            )
            send_whatsapp_message(st.phone, msg)

    # Notify returning student
    student = db.query(Student).filter(Student.usn == issue.student_usn).first()
    if student and student.phone:
        msg = (
            f"📚 Your book return has been accepted.\n"
            f"Accession: {issue.book_accession_no}\n"
            f"Thank you for returning the book!"
        )
        send_whatsapp_message(student.phone, msg)

    # Notify librarian confirmation done
    send_email(
        LIBRARIAN_EMAIL,
        "Book Return Confirmed",
        f"Return confirmed for USN {issue.student_usn}, Accession {issue.book_accession_no}."
    )

    return {"message": "Book return confirmed, stock updated, and student notified."}

# ---------------------------
# WISHLIST
# ---------------------------

@app.post("/wishlist/add")
def add_to_wishlist(data: WishlistIn, db: Session = Depends(get_db)):
    student = db.query(Student).filter(Student.usn == data.usn).first()
    if not student:
        raise HTTPException(status_code=404, detail="Student not found.")

    book = db.query(Book).filter(Book.accession_no == data.accession_no).first()
    if not book:
        raise HTTPException(status_code=404, detail="Book not found.")

    existing = db.query(Wishlist).filter(
        Wishlist.student_usn == data.usn,
        Wishlist.book_accession_no == data.accession_no
    ).first()
    if existing:
        return {"message": "Book already in wishlist."}

    w = Wishlist(student_usn=data.usn, book_accession_no=data.accession_no)
    db.add(w)
    db.commit()
    db.refresh(w)

    return {"message": "Book added to wishlist."}

# ---------------------------
# STUDENT PROFILE (MY LIBRARY)
# ---------------------------

@app.get("/student/profile/{usn}")
def student_profile(usn: str, db: Session = Depends(get_db)):
    student = db.query(Student).filter(Student.usn == usn).first()
    if not student:
        raise HTTPException(status_code=404, detail="Student not found.")

    active_issues = db.query(Issue).filter(
        Issue.student_usn == usn,
        Issue.status.in_(["issued", "renewed", "pending_return"])
    ).all()

    wishlist_items = db.query(Wishlist).filter(Wishlist.student_usn == usn).all()

    active_books = []
    for iss in active_issues:
        b = db.query(Book).filter(Book.accession_no == iss.book_accession_no).first()
        active_books.append({
            "accession_no": iss.book_accession_no,
            "title": b.title if b else "",
            "issue_date": str(iss.issue_date),
            "due_date": str(iss.due_date),
            "status": iss.status,
        })

    wishlist_books = []
    for w in wishlist_items:
        b = db.query(Book).filter(Book.accession_no == w.book_accession_no).first()
        wishlist_books.append({
            "accession_no": w.book_accession_no,
            "title": b.title if b else "",
            "added_on": w.created_at.strftime("%Y-%m-%d %H:%M"),
        })

    return {
        "student": {
            "usn": student.usn,
            "name": student.name,
            "semester": student.semester,
            "branch": student.branch,
            "phone": student.phone,
            "email": student.email,
        },
        "active_books": active_books,
        "wishlist": wishlist_books,
    }

# ---------------------------
# DAILY REPORTS
# ---------------------------

@app.post("/reports/daily")
def daily_reports(db: Session = Depends(get_db)):
    today = date.today()

    today_attendance = db.query(Attendance).filter(Attendance.date == today).all()
    today_issues = db.query(Issue).filter(Issue.issue_date == today).all()

    att_pdf = f"attendance_{today}.pdf"
    iss_pdf = f"issues_{today}.pdf"

    generate_attendance_pdf(today_attendance, att_pdf)
    generate_issues_pdf(today_issues, iss_pdf, title="Books Issued Today")

    subject = f"Daily Library Reports - {today}"
    body = "Please find attached today's attendance and issued book reports."

    send_email(LIBRARIAN_EMAIL, subject, body, attachment_path=att_pdf)
    send_email(LIBRARIAN_EMAIL, subject, body, attachment_path=iss_pdf)

    return {"message": "Daily reports generated and emailed."}

# ---------------------------
# REMINDERS: DUE / OVERDUE
# ---------------------------

@app.post("/reminders/due")
def send_due_reminders(db: Session = Depends(get_db)):
    today = date.today()

    upcoming = db.query(Issue).filter(
        Issue.due_date == today + timedelta(days=5),
        Issue.status.in_(["issued", "renewed"])
    ).all()

    overdue = db.query(Issue).filter(
        Issue.due_date < today,
        Issue.status.in_(["issued", "renewed"])
    ).all()

    for iss in upcoming:
        student = db.query(Student).filter(Student.usn == iss.student_usn).first()
        if student and student.phone:
            msg = (
                f"Reminder: Your library book (Accession {iss.book_accession_no}) "
                f"is due on {iss.due_date}. Please return or renew."
            )
            send_whatsapp_message(student.phone, msg)

    for iss in overdue:
        student = db.query(Student).filter(Student.usn == iss.student_usn).first()
        if student and student.phone:
            msg = (
                f"OVERDUE: Your library book (Accession {iss.book_accession_no}) "
                f"was due on {iss.due_date}. Kindly return it."
            )
            send_whatsapp_message(student.phone, msg)

    subject = f"Due/Overdue Book Summary - {today}"
    body_lines = ["Upcoming due in 5 days:"]
    for iss in upcoming:
        body_lines.append(f"{iss.student_usn} | {iss.book_accession_no} | due {iss.due_date}")

    body_lines.append("\nOverdue books:")
    for iss in overdue:
        body_lines.append(f"{iss.student_usn} | {iss.book_accession_no} | due {iss.due_date}")

    body = "\n".join(body_lines)
    send_email(LIBRARIAN_EMAIL, subject, body)

    return {"message": "Due/overdue reminders sent."}
