import io
import streamlit as st
from xhtml2pdf import pisa

def create_pdf(content):
    try:
        pdf_buffer = io.BytesIO()
        pisa_status = pisa.CreatePDF(io.StringIO(content), dest=pdf_buffer)
        if pisa_status.err:
            st.error("Failed to create PDF")
            return None
        pdf_buffer.seek(0)
        return pdf_buffer
    except Exception as e:
        st.error(f"Error creating PDF: {e}")
        return None