# import os
import smtplib, ssl
from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email_config import sender_email, apppwd, recipients

port = 465 # for ssl
password = apppwd

attachment_dir = '/home/ckampa/Dropbox/monitoring/hpc/'

def daily_email(date):
    # date: a datetime.datetime object.
    subject = f'HPC Daily: Summary {date}'
    subdir = f'daily/{date.year}/{date.month:0>2}/'
    fname = f'{date}_HPC_Picturebook.pdf'
    attachment_file = attachment_dir+subdir+fname
    body = """\
    Good morning!

    Please view a snapshot of the last 24 hours of slow controls data in the atttachment.

    If there are any questions, talk to Cole! (colekampa2024@u.northwestern.edu)
    """

    message = MIMEMultipart("alternative")
    message["Subject"] = subject
    message["From"] = sender_email
    message["To"] = ", ".join(recipients)
    message["Bcc"] = ", ".join(recipients)  # Recommended for mass emails

    # Add body to email
    message.attach(MIMEText(body, "plain"))

    # Open PDF file in binary mode
    with open(attachment_file, "rb") as attachment:
        # Add file as application/octet-stream
        # Email client can usually download this automatically as attachment
        part = MIMEBase("application", "octet-stream")
        part.set_payload(attachment.read())

    # Encode file in ASCII characters to send by email
    encoders.encode_base64(part)

    # Add header as key/value pair to attachment part
    part.add_header(
        "Content-Disposition",
        f"attachment; filename= {fname}",
    )

    # Add attachment to message and convert message to string
    message.attach(part)
    text = message.as_string()

    # Create a secure SSL context
    context = ssl.create_default_context()

    with smtplib.SMTP_SSL("smtp.gmail.com", port, context=context) as server:
        server.login(sender_email, password)
        server.sendmail(sender_email, recipients, text)

def NMR_error_email(date, df, m_bad):
    df_ = df[m_bad].copy()
    df_ = df_[['NMR [T]', 'Magnet Current [A]']]
    # date: a datetime.datetime object.
    subject = f'HPC ERROR: NMR {date}'
    #subdir = f'daily/{date.year}/{date.month:0>2}/'
    fname = f'picturebook_recent.pdf'
    attachment_file = attachment_dir+fname
    body = f"""\
    Urgent!

    Please check on the Hall probe calibration system. Something looks weird with the NMR data. If you also receive a Magnet Current email, likely the magnet tripped. The "recent" picturebook when the error was flagged is attached.

    Flagged data:
    {df_}

    If there are any questions, talk to Cole! (colekampa2024@u.northwestern.edu)
    """

    message = MIMEMultipart("alternative")
    message["Subject"] = subject
    message["From"] = sender_email
    message["To"] = ", ".join(recipients)
    message["Bcc"] = ", ".join(recipients)  # Recommended for mass emails

    # Add body to email
    message.attach(MIMEText(body, "plain"))

    # Open PDF file in binary mode
    with open(attachment_file, "rb") as attachment:
        # Add file as application/octet-stream
        # Email client can usually download this automatically as attachment
        part = MIMEBase("application", "octet-stream")
        part.set_payload(attachment.read())

    # Encode file in ASCII characters to send by email
    encoders.encode_base64(part)

    # Add header as key/value pair to attachment part
    part.add_header(
        "Content-Disposition",
        f"attachment; filename= {fname}",
    )

    # Add attachment to message and convert message to string
    message.attach(part)
    text = message.as_string()

    # Create a secure SSL context
    context = ssl.create_default_context()

    with smtplib.SMTP_SSL("smtp.gmail.com", port, context=context) as server:
        server.login(sender_email, password)
        server.sendmail(sender_email, recipients, text)

def Magnet_Current_error_email(date, df, m_bad):
    df_ = df[m_bad].copy()
    if 'NMR [T]' in df_.columns:
        cols = ['Magnet Current [A]', 'Magnet Voltage [V]', 'NMR [T]']
    else:
        cols = ['Magnet Current [A]', 'Magnet Voltage [V]']
    df_ = df_[cols]
    # date: a datetime.datetime object.
    subject = f'HPC ERROR: Magnet Current {date}'
    #subdir = f'daily/{date.year}/{date.month:0>2}/'
    fname = f'picturebook_recent.pdf'
    attachment_file = attachment_dir+fname
    body = f"""\
    Urgent!

    Please check on the Hall probe calibration system. Something looks weird with the Magnet Current data. The "recent" picturebook when the error was flagged is attached.

    Flagged data:
    {df_}

    If there are any questions, talk to Cole! (colekampa2024@u.northwestern.edu)
    """

    message = MIMEMultipart("alternative")
    message["Subject"] = subject
    message["From"] = sender_email
    message["To"] = ", ".join(recipients)
    message["Bcc"] = ", ".join(recipients)  # Recommended for mass emails

    # Add body to email
    message.attach(MIMEText(body, "plain"))

    # Open PDF file in binary mode
    with open(attachment_file, "rb") as attachment:
        # Add file as application/octet-stream
        # Email client can usually download this automatically as attachment
        part = MIMEBase("application", "octet-stream")
        part.set_payload(attachment.read())

    # Encode file in ASCII characters to send by email
    encoders.encode_base64(part)

    # Add header as key/value pair to attachment part
    part.add_header(
        "Content-Disposition",
        f"attachment; filename= {fname}",
    )

    # Add attachment to message and convert message to string
    message.attach(part)
    text = message.as_string()

    # Create a secure SSL context
    context = ssl.create_default_context()

    with smtplib.SMTP_SSL("smtp.gmail.com", port, context=context) as server:
        server.login(sender_email, password)
        server.sendmail(sender_email, recipients, text)

def DataStream_error_email(date, df, min_since_data):
    # date: a datetime.datetime object.
    subject = f'HPC ERROR: Data Stream {date}'
    fname = f'picturebook_recent.pdf'
    attachment_file = attachment_dir+fname
    body = f"""\
    Urgent!

    Please check on the Hall probe calibration system. There has not been an update to the data file in {min_since_data:0.1f} minutes. Sometimes Dropbox has not synced, check that first. The "recent" picturebook is attached.

    Most recent data:
    {df.tail()}

    If there are any questions, talk to Cole! (colekampa2024@u.northwestern.edu)
    """

    message = MIMEMultipart("alternative")
    message["Subject"] = subject
    message["From"] = sender_email
    message["To"] = ", ".join(recipients)
    message["Bcc"] = ", ".join(recipients)  # Recommended for mass emails

    # Add body to email
    message.attach(MIMEText(body, "plain"))

    # Open PDF file in binary mode
    with open(attachment_file, "rb") as attachment:
        # Add file as application/octet-stream
        # Email client can usually download this automatically as attachment
        part = MIMEBase("application", "octet-stream")
        part.set_payload(attachment.read())

    # Encode file in ASCII characters to send by email
    encoders.encode_base64(part)

    # Add header as key/value pair to attachment part
    part.add_header(
        "Content-Disposition",
        f"attachment; filename= {fname}",
    )

    # Add attachment to message and convert message to string
    message.attach(part)
    text = message.as_string()

    # Create a secure SSL context
    context = ssl.create_default_context()

    with smtplib.SMTP_SSL("smtp.gmail.com", port, context=context) as server:
        server.login(sender_email, password)
        server.sendmail(sender_email, recipients, text)
