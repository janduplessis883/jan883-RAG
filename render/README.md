# Render email-ingestion service

This directory contains the cloud-side email receiver and its Render deployment
configuration. It is intentionally separate from the local RAG application and
the macOS OneDrive watcher.

The service will receive Resend `email.received` webhooks, retrieve the full
message and attachments from Resend, and create a page in the Notion **Work
Email Archive** data source.

## Render configuration

Configure the Render service with this directory as its Root Directory:

- Build command: `pip install -r requirements.txt`
- Start command: `uvicorn app:app --host 0.0.0.0 --port $PORT`

Copy the variables in `.env.example` into Render's environment settings. Do not
commit a real `.env` file or any API keys.

Required environment variables include:

- `RESEND_API_KEY`
- `RESEND_WEBHOOK_SECRET`
- `NOTION_TOKEN`
- `NOTION_DATA_SOURCE_ID`
- `NOTION_CALENDAR_DATA_SOURCE_ID`

If the email subject contains `calendar` (case-insensitive), the
service creates a page in the Calendar data source using the subject as the
`Event` title, the current UTC time as the ISO 8601 `Date`, and the body as page
content. Calendar messages bypass duplicate checking and attachment
processing. Other messages retain the existing Work Email Archive behavior.
