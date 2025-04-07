import requests
import json

def send_email_template(emaildata:dict):
    reqUrl = "https://api.zeptomail.com/v1.1/email/template"

    headersList = {
        "Accept": "*/*",
        "User-Agent": "Thunder Client (https://www.thunderclient.com)",
        "Authorization": "Zoho-enczapikey wSsVR61+qRbzWP95zjL4cec+mlQBAwz2QUR00FWh6Hb9GPnDpcc8nkLPVwKlFfAYRzM7FjJH97wqzEhR1zsLhtkvm1xRCSiF9mqRe1U4J3x17qnvhDzDX2xbkxqOL4MIxA1tkmJhEc8g+g==",
        "Content-Type": "application/json"
    }
    payload={
        "template_alias": "taskzverify",
        "bounce_address": "alerts@bounce.treeone.in",
        "from": {
            "address": "noreply@treeone.co.in",
            "name": "TreeOne Taskz+"
        },
        "to": [
            {
                "email_address": {
                    "address": emaildata['to'],
                    "name": emaildata['user_name']
                }
            }
        ],
        "subject":"Agent",
        "merge_info": {
            "product_name": "AGENT-CHAT",
            "name": emaildata['user_name'],
            "LINK": emaildata['file_url'],
            "team": "Treeone Team"
        }
    }

    # Convert the payload to JSON format if it's a dictionary
    if isinstance(payload, dict):
        payload = json.dumps(payload)
    try:
        response = requests.post(reqUrl, data=payload, headers=headersList)

        # Check the response status code
        if response.status_code == 201:
            print("Email sent successfully!")
            print("Response Content:", response.json())
            return "Email sent successfully!"
        else:
            print("Failed to send email. Status Code:", response.status_code)
            print("Response Content:", response.text)
            return "Failed to send email"

    except requests.exceptions.RequestException as e:
        print("An error occurred while sending the email:", e)



def send_python_email(emaildata:dict):
    print("Email dataaaaaaaaaaa",emaildata)
    url = "https://api.zeptomail.com/v1.1/email"

    # Load API key from environment variable
    api_key = "wSsVR61+qRbzWP95zjL4cec+mlQBAwz2QUR00FWh6Hb9GPnDpcc8nkLPVwKlFfAYRzM7FjJH97wqzEhR1zsLhtkvm1xRCSiF9mqRe1U4J3x17qnvhDzDX2xbkxqOL4MIxA1tkmJhEc8g+g=="
    if not api_key:
        raise ValueError("API key is missing. Set ZEPTO_API_KEY as an environment variable.")

    email_html = f'''
    <div style="font-family: Arial, sans-serif; padding: 20px; background-color: #f4f4f4; text-align: center;">
        <div style="max-width: 600px; background: white; padding: 20px; border-radius: 10px; box-shadow: 0 0 10px rgba(0,0,0,0.1); margin: auto;">
            <h2 style="color: #333;">🎉 Test Email Sent Successfully! 🎉</h2>
            <p style="color: #666; font-size: 16px;">Dear {emaildata['user_name']},</p>
            <p style="color: #666; font-size: 16px;">Your requested file is ready. Click the button below to access it:</p>
            <a href="{emaildata['file_url']}" style="display: inline-block; background-color: #28a745; color: white; padding: 12px 20px; text-decoration: none; font-size: 16px; border-radius: 5px; margin-top: 10px;">📁 View File</a>
            <p style="color: #666; font-size: 14px; margin-top: 20px;">If the button does not work, copy and paste the following link into your browser:</p>
            <p style="color: #007bff; word-break: break-word;">{emaildata['file_url']}</p>
        </div>
    </div>
    '''
    payload = {
        "from": {"address": "noreply@treeone.in"},
        "to": [{"email_address": {"address":emaildata['to'], "name": emaildata['user_name']}}],
        "subject": "AI-AGENT-BOT",
        "htmlbody": email_html
    }

    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "authorization": f"Zoho-enczapikey {api_key}"
    }

    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()  # Raise an exception for HTTP errors
        print("Email sent successfully:", response.json())
    except requests.exceptions.RequestException as e:
        print("Error sending email:", e)

