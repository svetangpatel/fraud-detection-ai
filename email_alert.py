import smtplib
from email.mime.text import MIMEText

def send_fraud_alert():

    sender = "bankalert@gmail.com"
    password = "abcdefghijklmnop"  # Use an app password for Gmail
    receiver = "receiveremail@gmail.com"

    msg = MIMEText("🚨 Fraud Transaction Detected!")
    msg["Subject"] = "Fraud Alert"
    msg["From"] = sender
    msg["To"] = receiver

    server = smtplib.SMTP_SSL("smtp.gmail.com", 465)
    server.login(sender, password)
    server.send_message(msg)
    server.quit()

    print("Email sent successfully")