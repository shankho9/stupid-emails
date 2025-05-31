import os.path
import pickle
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
import base64
from email.mime.text import MIMEText
import json
from openai import OpenAI
import os
import tiktoken  # Add this to imports at top
import re  # Add this to imports at top
from datetime import datetime, timedelta  # Update this import line
import time  # Add to imports at top

# Load secrets from secrets.json
def load_secrets():
    try:
        with open('secrets.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError("secrets.json file not found. Please create it with your OpenAI API key.")
    except json.JSONDecodeError:
        raise ValueError("secrets.json is not properly formatted JSON.")

# Initialize OpenAI client with API key from secrets
secrets = load_secrets()
client = OpenAI(api_key=secrets['openai_api_key'])

# If modifying these scopes, delete the file token.pickle.
SCOPES = ['https://www.googleapis.com/auth/gmail.readonly', 'https://www.googleapis.com/auth/gmail.modify']

# Constants
CLASSIFICATION_LABELS = ["ATTN", "FK-U", "MARKETING", "TAKE-A-LOOK", "HMMMM"]

def load_prompt_template():
    """Loads the prompt template from prompt.txt"""
    try:
        with open('prompt.txt', 'r') as f:
            return f.read()
    except FileNotFoundError:
        raise FileNotFoundError("prompt.txt file not found. Please create it with your classification instructions.")

def classify_email(sender, subject, content, email_date):
    """Classifies email using GPT-4."""
    try:
        # Load the prompt template
        prompt_template = load_prompt_template()
        
        # Format the prompt with the email details
        prompt = prompt_template.format(
            sender=sender,
            subject=subject,
            content=content,
            email_date=email_date
        )
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            response_format={ "type": "json_object" },
            messages=[
                {"role": "system", "content": "You are an email classifier that categorizes emails."},
                {"role": "user", "content": prompt}
            ]
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"Debug - Error details: {str(e)}")
        return {"classification": "ERROR", "reason": f"Failed to classify: {str(e)}"}

def get_gmail_service():
    """Gets or creates Gmail API service."""
    creds = None
    
    # The file token.pickle stores the user's access and refresh tokens
    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)
    
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        
        # Save the credentials for the next run
        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)

    return build('gmail', 'v1', credentials=creds)

def count_tokens(text, model="gpt-4"):
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

def clean_email_content(text):
    """Cleans email content while preserving quotes and important formatting."""
    
    # Remove style and script tags and their contents
    text = re.sub(r'<style[^>]*>[\s\S]*?</style>', '', text)
    text = re.sub(r'<script[^>]*>[\s\S]*?</script>', '', text)
    
    # Remove image tags but keep alt text
    text = re.sub(r'<img[^>]*alt="([^"]*)"[^>]*>', r'\1', text)
    text = re.sub(r'<img[^>]*>', '', text)
    
    # Remove common marketing/footer patterns while preserving quote structure
    patterns_to_remove = [
        r'Copyright ©.*?(?=\n|$)',
        r'You are receiving this email because.*?(?=\n|$)',
        r'To connect with us.*?(?=\n|$)',
        r'Our mailing address.*?(?=\n|$)',
        r'Unsubscribe.*?(?=\n|$)',
        r'Add .* to your address book.*?(?=\n|$)',
    ]
    
    for pattern in patterns_to_remove:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.MULTILINE)
    
    # Remove HTML attributes that don't affect content
    text = re.sub(r'style="[^"]*"', '', text)
    text = re.sub(r'class="[^"]*"', '', text)
    text = re.sub(r'width="[^"]*"', '', text)
    text = re.sub(r'height="[^"]*"', '', text)
    text = re.sub(r'align="[^"]*"', '', text)
    
    # Remove URLs and base64 images while preserving link text
    text = re.sub(r'<a[^>]*href="[^"]*"[^>]*>([^<]+)</a>', r'\1', text)
    text = re.sub(r'data:image/[^;]+;base64,[a-zA-Z0-9+/]+={0,2}', '', text)
    
    # Remove remaining HTML tags but preserve their content
    text = re.sub(r'<[^>]+>', ' ', text)
    
    # Clean up whitespace while preserving email quote structure
    text = re.sub(r' +', ' ', text)  # Multiple spaces to single space
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)  # Multiple blank lines to double line
    
    # Preserve common quote markers
    text = re.sub(r'^\s*>+\s*', '> ', text, flags=re.MULTILINE)  # Standardize quote markers
    
    # Better quote handling - collapse multiple '>' into single '>'
    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        # If line starts with multiple '>', reduce to single '>'
        if line.strip().startswith('>'):
            # Remove extra spaces around '>' symbols
            line = re.sub(r'\s*>\s*>\s*', '> ', line)
            # Ensure only one '>' at start with one space after
            line = re.sub(r'^\s*>+\s*', '> ', line)
        cleaned_lines.append(line)
    
    text = '\n'.join(cleaned_lines)
    
    # Clean up any remaining multiple blank lines
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
    
    return text.strip()

def create_labels_if_needed(service):
    """Creates our classification labels if they don't exist."""
    results = service.users().labels().list(userId='me').execute()
    existing_labels = {label['name']: label['id'] for label in results.get('labels', [])}
    
    # Create any missing labels
    for label_name in CLASSIFICATION_LABELS:
        if label_name not in existing_labels:
            label_body = {
                'name': label_name,
                'messageListVisibility': 'show',
                'labelListVisibility': 'labelShow'
            }
            service.users().labels().create(userId='me', body=label_body).execute()

def update_email_label(service, message_id, classification):
    """Updates the label of an email based on its classification, ensuring only one label at a time."""
    try:
        # Get current labels for the message
        msg = service.users().messages().get(userId='me', id=message_id).execute()
        
        # Get label name mappings
        label_results = service.users().labels().list(userId='me').execute()
        
        # Remove existing classification labels
        classification_label_ids = [
            label['id'] for label in label_results.get('labels', [])
            if label['name'] in CLASSIFICATION_LABELS
        ]
        
        if classification_label_ids:
            service.users().messages().modify(
                userId='me',
                id=message_id,
                body={'removeLabelIds': classification_label_ids}
            ).execute()
        
        # Add the new label
        new_label_id = next(
            (label['id'] for label in label_results.get('labels', [])
            if label['name'] == classification),
            None
        )
        
        if new_label_id:
            service.users().messages().modify(
                userId='me',
                id=message_id,
                body={'addLabelIds': [new_label_id]}
            ).execute()
        else:
            print(f"Warning: Could not find label ID for classification '{classification}'")
            fallback_label_id = next(
                (label['id'] for label in label_results.get('labels', [])
                if label['name'] == "HMMMM"),
                None
            )
            if fallback_label_id:
                service.users().messages().modify(
                    userId='me',
                    id=message_id,
                    body={'addLabelIds': [fallback_label_id]}
                ).execute()
            else:
                print("Error: Could not find HMMMM label")
        
    except Exception as e:
        print(f"Error in update_email_label: {str(e)}")
        raise

def read_emails_in_date_range():
    """Reads and processes emails from Nov 1 2023 to present."""
    service = get_gmail_service()
    
    # Create labels if they don't exist
    create_labels_if_needed(service)
    
    start_date = (datetime.now() - timedelta(days=14)).strftime('%Y/%m/%d')
    query = f'in:inbox -in:spam after:{start_date}'
    
    try:
        results = service.users().messages().list(userId='me', q=query).execute()
        messages = results.get('messages', [])
        
        while 'nextPageToken' in results:
            time.sleep(2)
            results = service.users().messages().list(
                userId='me',
                q=query,
                pageToken=results['nextPageToken']
            ).execute()
            messages.extend(results.get('messages', []))
            print(f"Fetched {len(messages)} messages so far...")

        if not messages:
            print('No messages found.')
            return
        
        print(f'Processing {len(messages)} emails:')
        print('-' * 50)
        
        for i, message in enumerate(messages):
            if i > 0 and i % 5 == 0:
                print(f"Processed {i} messages. Pausing briefly...")
                time.sleep(3)
            
            # Get full message details including labels
            msg = service.users().messages().get(userId='me', id=message['id']).execute()
            
            # Get current labels
            current_labels = set(msg.get('labelIds', []))
            
            # Get label name to ID mapping
            results = service.users().labels().list(userId='me').execute()
            label_map = {label['name']: label['id'] for label in results.get('labels', [])}
            
            # Check if message has any of our classification labels already
            should_skip = any(
                label_map.get(label) in current_labels 
                for label in CLASSIFICATION_LABELS
            )
            
            if should_skip:
                print(f"Skipping message - already has a classification label")
                continue
            
            # Process headers and content as before
            headers = msg['payload']['headers']
            subject = next((header['value'] for header in headers if header['name'].lower() == 'subject'), 'No Subject')
            sender = next((header['value'] for header in headers if header['name'].lower() == 'from'), 'No Sender')
            email_date = next((header['value'] for header in headers if header['name'].lower() == 'date'), 'No Date')
            
            if 'parts' in msg['payload']:
                parts = msg['payload']['parts']
                data = parts[0]['body'].get('data', '')
            else:
                data = msg['payload']['body'].get('data', '')
                
            if data:
                text = base64.urlsafe_b64decode(data).decode('utf-8')
                text = clean_email_content(text)
            else:
                text = 'No content'
            
            token_count = count_tokens(text)
            print(f'Token count: {token_count}')
            
            classification = classify_email(sender, subject, text, email_date)
            
            if classification['classification'] not in ['ERROR']:
                update_email_label(service, message['id'], classification['classification'])
            
            print(f'From: {sender}')
            print(f'Subject: {subject}')
            print('Content:')
            print(text[:300] + '...' if len(text) > 300 else text)
            print('\nClassification:')
            print(f"Type: {classification['classification']}")
            print(f"Reason: {classification['reason']}")
            print('-' * 50)

    except Exception as e:
        print(f"An error occurred: {str(e)}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

def test_specific_email():
    """Test function to process and tag emails only from ddolin@a16z.com"""
    service = get_gmail_service()
    
    # Create labels if they don't exist
    create_labels_if_needed(service)
    
    # Specific query for emails from ddolin@a16z.com
    query = 'from:ddolin@a16z.com'
    
    try:
        results = service.users().messages().list(userId='me', q=query).execute()
        messages = results.get('messages', [])
        
        if not messages:
            print('No messages found from ddolin@a16z.com')
            return
        
        print(f'Processing {len(messages)} emails from ddolin@a16z.com:')
        print('-' * 50)
        
        for message in messages:
            msg = service.users().messages().get(userId='me', id=message['id']).execute()
            
            headers = msg['payload']['headers']
            subject = next((header['value'] for header in headers if header['name'].lower() == 'subject'), 'No Subject')
            sender = next((header['value'] for header in headers if header['name'].lower() == 'from'), 'No Sender')
            email_date = next((header['value'] for header in headers if header['name'].lower() == 'date'), 'No Date')
            
            if 'parts' in msg['payload']:
                parts = msg['payload']['parts']
                data = parts[0]['body'].get('data', '')
            else:
                data = msg['payload']['body'].get('data', '')
                
            if data:
                text = base64.urlsafe_b64decode(data).decode('utf-8')
                text = clean_email_content(text)
            else:
                text = 'No content'
            
            # Get classification from GPT
            classification = classify_email(sender, subject, text, email_date)
            
            print(f'Date: {email_date}')
            print(f'Subject: {subject}')
            print('Content:')
            print(text[:300] + '...' if len(text) > 300 else text)
            print('\nClassification:')
            print(f"Type: {classification['classification']}")
            print(f"Reason: {classification['reason']}")
            
            # Apply the label
            if classification['classification'] not in ['ERROR']:
                update_email_label(service, message['id'], classification['classification'])
                print(f"Applied label: {classification['classification']}")
            
            print('-' * 50)

    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == '__main__':
    # test_specific_email()
    read_emails_in_date_range()
