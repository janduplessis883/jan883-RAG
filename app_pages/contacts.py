import streamlit as st

from app_pages.common import render_runtime_sidebar, runtime
from app_pages.contact_sync import contacts_sync_panel
from local_rag.contacts import hydrate_contacts

st.title('Contacts')
st.caption('Your Notion contacts, available for exact lookups and description search in Chat.')
render_runtime_sidebar()
config, database, *_ = runtime()
contacts_sync_panel()
query = st.text_input('Search contacts', placeholder='Name, company, email, telephone, address or description')
missing = st.selectbox('Missing details', ['Any', 'Email', 'Telephone', 'Address', 'Website'])
records = database.list_contacts()
if query.strip():
    needle = query.strip().casefold()
    records = [row for row in records if needle in ' '.join(str(row[field]) for field in (
        'name', 'description', 'address', 'email', 'tel', 'url', 'email_normalized', 'tel_normalized')).casefold()]
field = {'Email': 'email', 'Telephone': 'tel', 'Address': 'address', 'Website': 'url'}.get(missing)
if field:
    records = [row for row in records if not row[field]]
st.caption(f'{len(records)} contacts')
if not records:
    st.info('No contacts found. Sync your Notion contacts or adjust the filters.')
else:
    rows = [{'Name / Company': row['name'], 'Description': row['description'], 'Address': row['address'],
             'Email': row['email'], 'Tel': row['tel'], 'Website': row['url'], 'Notion': row['notion_url'],
             'Attachments': len(row['attachments'])} for row in records]
    st.dataframe(rows, hide_index=True, column_config={
        'Website': st.column_config.LinkColumn('Website'), 'Notion': st.column_config.LinkColumn('Open in Notion')})
    selected_id = st.selectbox('Contact details', [row['notion_page_id'] for row in records],
                              format_func=lambda value: next(row['name'] for row in records if row['notion_page_id'] == value))
    selected = next(row for row in records if row['notion_page_id'] == selected_id)
    selected = hydrate_contacts(config, database, [selected])[0]
    with st.container(border=True):
        st.subheader(selected['name'])
        st.write(selected['description'] or 'No description provided.')
        for label, field in [('Telephone', 'tel'), ('Email', 'email'), ('Address', 'address')]:
            st.write(f"{label}: {selected[field] or 'Not specified'}")
        st.link_button('Open in Notion', selected['notion_url'], icon=':material/open_in_new:')
        for attachment in selected['attachments']:
            suffix = ' (open in Notion)' if attachment.get('open_in_notion') else ''
            st.link_button(attachment['name'] + suffix, attachment['url'], icon=':material/attach_file:')
        st.caption('Attachment links only; their contents are not indexed.')
