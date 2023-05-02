# import os
import smtplib, ssl
from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email_config import apppwd

port = 465 # for ssl
sender_email = 'NU.Mu2e.Monitoring@gmail.com'
password = apppwd
receiver_email = "colekampa2024@u.northwestern.edu"

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
    message["To"] = receiver_email
    message["Bcc"] = receiver_email  # Recommended for mass emails

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
        server.sendmail(sender_email, receiver_email, text)
